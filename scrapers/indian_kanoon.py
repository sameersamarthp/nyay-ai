"""
Indian Kanoon scraper.

Primary source for legal documents. Supports both API access (preferred)
and HTML scraping as fallback.

API documentation: https://api.indiankanoon.org/

OPTIMIZED: Uses POST for both search and document fetch via API.
This avoids rate limiting issues with HTML scraping.
Supports concurrent fetching with multiple threads.
"""

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from queue import Queue, Empty
from typing import Generator
from urllib.parse import urljoin, urlencode

from bs4 import BeautifulSoup
from dateutil.parser import parse as parse_date
from tqdm import tqdm

from config.settings import settings
from storage.document_store import DocumentStore
from storage.schemas import DocumentSource, LegalDocument, ScrapingProgress
from scrapers.base_scraper import BaseScraper
from utils.logger import get_logger
from utils.retry import ParseError, FetchError

logger = get_logger(__name__)


class IndianKanoonScraper(BaseScraper):
    """Scraper for Indian Kanoon legal database."""

    source = DocumentSource.INDIAN_KANOON

    def __init__(
        self,
        store: DocumentStore | None = None,
        target_count: int | None = None,
        api_token: str | None = None,
        court_filter: str | None = None,
        use_api: bool | None = None,
        num_threads: int = 1,
    ):
        """Initialize Indian Kanoon scraper.

        Args:
            store: Document store instance.
            target_count: Number of documents to collect.
            api_token: Indian Kanoon API token (if available).
            court_filter: Filter for specific court (e.g., "supremecourt").
            use_api: Force API (True), force HTML (False), or auto-detect (None).
            num_threads: Number of concurrent threads for fetching (default: 1).
        """
        # Set attributes before super().__init__() since _get_headers needs them
        self.api_token = api_token or settings.INDIAN_KANOON_API_TOKEN
        self.court_filter = court_filter
        self.base_url = settings.INDIAN_KANOON_BASE_URL
        self.api_url = settings.INDIAN_KANOON_API_URL
        self._current_page = 0

        # Determine mode: explicit choice or auto-detect based on token
        if use_api is None:
            self.use_api = bool(self.api_token)
        else:
            self.use_api = use_api

        super().__init__(store, target_count, num_threads)

        if self.use_api:
            if self.api_token:
                logger.info("Mode: API (token available)")
            else:
                logger.warning("Mode: API requested but no token - will likely fail")
        else:
            logger.info("Mode: HTML scraping (API disabled or no token)")

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers including API token if available."""
        headers = super()._get_headers()
        if self.api_token:
            headers["Authorization"] = f"Token {self.api_token}"
        return headers

    def get_document_urls(self) -> Generator[str, None, None]:
        """Generate document URLs from search results.

        Yields:
            Document URLs.
        """
        if self.use_api:
            yield from self._get_urls_via_api()
        else:
            yield from self._get_urls_via_html()

    def _api_search(self, form_input: str, page: int = 0) -> dict | None:
        """Make a POST request to the Indian Kanoon search API.

        Args:
            form_input: Search query string.
            page: Page number (0-indexed).

        Returns:
            JSON response dict or None on failure.
        """
        search_url = f"{self.api_url}/search/"

        data = {
            "formInput": form_input,
            "pagenum": page,
        }

        # Explicitly set headers for API request
        # Note: The API rejects bot-like User-Agents, so we use a standard one
        headers = {
            "Authorization": f"Token {self.api_token}",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        }

        with self.rate_limiter.acquire():
            try:
                response = self.session.post(
                    search_url,
                    data=data,
                    headers=headers,
                    timeout=settings.REQUEST_TIMEOUT,
                )

                if response.status_code == 429:
                    self.rate_limiter.record_error(is_rate_limit=True)
                    self.rate_limiter.backoff()
                    return None

                response.raise_for_status()
                self.rate_limiter.record_success()
                return response.json()

            except Exception as e:
                self.rate_limiter.record_error()
                logger.error(f"API search failed: {e}")
                return None

    def _fetch_doc_via_api(self, doc_id: str) -> dict | None:
        """Fetch a document via API using POST (more reliable than HTML scraping).

        Args:
            doc_id: Document ID (tid).

        Returns:
            JSON response dict or None on failure.
        """
        doc_url = f"{self.api_url}/doc/{doc_id}/"

        headers = {
            "Authorization": f"Token {self.api_token}",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        }

        with self.rate_limiter.acquire():
            try:
                response = self.session.post(
                    doc_url,
                    headers=headers,
                    timeout=settings.REQUEST_TIMEOUT,
                )

                if response.status_code == 429:
                    self.rate_limiter.record_error(is_rate_limit=True)
                    self.rate_limiter.backoff()
                    return None

                response.raise_for_status()
                self.rate_limiter.record_success()
                return response.json()

            except Exception as e:
                self.rate_limiter.record_error()
                logger.error(f"API doc fetch failed for {doc_id}: {e}")
                return None

    def _parse_api_document(self, doc_data: dict) -> LegalDocument | None:
        """Parse document from API JSON response.

        Args:
            doc_data: API response dict containing document data.

        Returns:
            LegalDocument if parsing succeeded, None otherwise.
        """
        try:
            doc_id = doc_data.get("tid")
            title = doc_data.get("title", "")
            doc_html = doc_data.get("doc", "")
            docsource = doc_data.get("docsource", "")

            if not doc_html or len(doc_html) < 100:
                logger.warning(f"Document {doc_id} too short, skipping")
                return None

            # Parse the HTML content from API response
            soup = self.parse_html(doc_html)
            url = f"{self.base_url}/doc/{doc_id}/"

            # Extract full text from the HTML
            full_text = self._clean_text(soup.get_text())

            if len(full_text) < 100:
                logger.warning(f"Document {doc_id} text too short after cleaning")
                return None

            # Extract metadata
            court = self._extract_court_from_docsource(docsource, title)
            date_decided = self._extract_date(soup)
            citation = self._extract_citation(soup, title)
            judges = self._extract_judges(soup)
            petitioner, respondent = self._extract_parties(title)
            acts_referred = self._extract_acts(soup, full_text)
            cases_cited = self._extract_cited_cases(soup)

            return LegalDocument(
                source=self.source,
                url=url,
                case_title=title,
                court=court,
                citation=citation,
                date_decided=date_decided,
                petitioner=petitioner,
                respondent=respondent,
                judges=judges,
                acts_referred=acts_referred,
                cases_cited=cases_cited,
                full_text=full_text,
            )

        except Exception as e:
            logger.error(f"Failed to parse API document: {e}")
            return None

    def _extract_court_from_docsource(self, docsource: str, title: str) -> str:
        """Extract court name from API docsource field."""
        if not docsource:
            return self._extract_court(BeautifulSoup("", "lxml"), title)

        docsource_lower = docsource.lower()
        if "supreme court" in docsource_lower:
            return "Supreme Court of India"
        if "delhi" in docsource_lower:
            return "Delhi High Court"
        if "bombay" in docsource_lower:
            return "Bombay High Court"
        if "karnataka" in docsource_lower:
            return "Karnataka High Court"
        if "high court" in docsource_lower:
            return docsource  # Use as-is
        return docsource or "Unknown Court"

    def _get_doc_ids_via_api(self) -> Generator[str, None, None]:
        """Get document IDs using the API (optimized for API-based fetching).

        Yields:
            Document IDs (tids).
        """
        # Build search queries based on court filter
        if self.court_filter:
            court_queries = {
                "supremecourt": ["judgment doctypes:supremecourt"],
                "delhi": ["judgment doctypes:delhi"],
                "bombay": ["judgment doctypes:bombay"],
                "karnataka": ["judgment doctypes:karnataka"],
                "highcourts": [
                    "judgment doctypes:delhi",
                    "judgment doctypes:bombay",
                    "judgment doctypes:karnataka",
                ],
            }
            queries = court_queries.get(self.court_filter, [f"judgment doctypes:{self.court_filter}"])
        else:
            # Expanded queries with year ranges and diverse topics for better coverage
            queries = [
                # Core legal topics
                "criminal appeal doctypes:judgments",
                "civil suit doctypes:judgments",
                "constitutional doctypes:judgments",
                "contract doctypes:judgments",
                "property doctypes:judgments",
                "writ petition doctypes:judgments",
                "arbitration doctypes:judgments",
                "tax appeal doctypes:judgments",
                # Additional topics for diversity
                "murder doctypes:judgments",
                "cheating doctypes:judgments",
                "defamation doctypes:judgments",
                "divorce doctypes:judgments",
                "land acquisition doctypes:judgments",
                "motor accident doctypes:judgments",
                "labour dispute doctypes:judgments",
                "service matter doctypes:judgments",
                "insurance claim doctypes:judgments",
                "banking doctypes:judgments",
                "company law doctypes:judgments",
                "intellectual property doctypes:judgments",
                "environmental doctypes:judgments",
                "consumer protection doctypes:judgments",
                # Year-specific queries for recent judgments
                "fromdate:2024-01-01 todate:2024-12-31 doctypes:judgments",
                "fromdate:2023-01-01 todate:2023-12-31 doctypes:judgments",
                "fromdate:2022-01-01 todate:2022-12-31 doctypes:judgments",
                "fromdate:2021-01-01 todate:2021-12-31 doctypes:judgments",
                "fromdate:2020-01-01 todate:2020-12-31 doctypes:judgments",
                "fromdate:2019-01-01 todate:2019-12-31 doctypes:judgments",
                # Court-specific queries
                "doctypes:supremecourt",
                "doctypes:delhi",
                "doctypes:bombay",
                "doctypes:allahabad",
                "doctypes:madras",
                "doctypes:calcutta",
            ]

        seen_doc_ids = set()

        for query in queries:
            page = 0
            max_pages = 200  # Increased for better coverage

            while page < max_pages:
                try:
                    data = self._api_search(query, page)

                    if not data:
                        logger.warning(f"API search returned no data for '{query}' at page {page}")
                        break

                    docs = data.get("docs", [])
                    if not docs:
                        logger.info(f"No more results for '{query}' at page {page}")
                        break

                    for doc in docs:
                        doc_id = doc.get("tid")
                        if doc_id and doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            yield str(doc_id)

                    page += 1

                except Exception as e:
                    logger.error(f"API search failed for '{query}' at page {page}: {e}")
                    break

    def _process_doc_id(self, doc_id: str) -> tuple[str, LegalDocument | None]:
        """Process a single document ID and return the document (thread-safe).

        Args:
            doc_id: Document ID to process.

        Returns:
            Tuple of (url, LegalDocument) if successful, (url, None) otherwise.
        """
        url = f"{self.base_url}/doc/{doc_id}/"
        try:
            doc_data = self._fetch_doc_via_api(doc_id)
            if not doc_data:
                return url, None
            doc = self._parse_api_document(doc_data)
            return url, doc
        except Exception as e:
            logger.error(f"Error processing doc {doc_id}: {e}")
            return url, None

    def scrape_via_api(self, resume: bool = True) -> int:
        """Optimized scrape using API for both search and document fetch.

        This method uses POST requests for both operations, which is more
        reliable and less likely to be rate limited.
        Supports concurrent fetching with multiple threads.

        Args:
            resume: Whether to resume from previous progress.

        Returns:
            Number of documents collected.
        """
        if not self.api_token:
            logger.warning("No API token, falling back to HTML scraping")
            return super().scrape(resume)

        # Load or create progress
        progress = None
        if resume:
            progress = self.store.get_progress(self.source)

        if not progress:
            progress = ScrapingProgress(
                source=self.source,
                total_target=self.target_count,
            )

        if progress.is_complete:
            logger.info(f"{self.source.value}: Already complete ({progress.documents_collected} docs)")
            return progress.documents_collected

        collected = progress.documents_collected
        logger.info(
            f"Starting {self.source.value} API scraper: "
            f"{collected}/{self.target_count} collected"
            f" (threads: {self.num_threads})"
        )

        # Progress bar
        pbar = tqdm(
            initial=collected,
            total=self.target_count,
            desc=f"{self.source.value} (API)",
            unit="docs",
        )

        # Pre-load existing doc URLs for fast duplicate checking
        logger.info("Loading existing document URLs for duplicate detection...")
        existing_urls = self.store.get_existing_urls(self.source)
        existing_urls_lock = threading.Lock()
        logger.info(f"Found {len(existing_urls)} existing documents")
        skipped_duplicates = 0

        if self.num_threads <= 1:
            # Sequential processing
            collected, skipped_duplicates = self._scrape_api_sequential(
                progress, collected, pbar, existing_urls
            )
        else:
            # Concurrent processing
            collected, skipped_duplicates = self._scrape_api_concurrent(
                progress, collected, pbar, existing_urls, existing_urls_lock
            )

        pbar.close()

        # Save final progress
        progress.documents_collected = collected
        progress.last_updated = datetime.now()
        progress.is_complete = collected >= self.target_count
        self.store.save_progress(progress)

        logger.info(
            f"{self.source.value}: Collected {collected}/{self.target_count} documents"
            f" (skipped {skipped_duplicates} duplicates)"
        )

        return collected

    def _scrape_api_sequential(
        self,
        progress: ScrapingProgress,
        collected: int,
        pbar: tqdm,
        existing_urls: set[str],
    ) -> tuple[int, int]:
        """Sequential API scraping.

        Returns:
            Tuple of (collected count, skipped duplicates count).
        """
        skipped_duplicates = 0

        try:
            for doc_id in self._get_doc_ids_via_api():
                if self._interrupted:
                    logger.warning("Interrupted! Saving progress...")
                    break

                if collected >= self.target_count:
                    break

                # Skip if we already have this document (fast check)
                url = f"{self.base_url}/doc/{doc_id}/"
                if url in existing_urls:
                    skipped_duplicates += 1
                    if skipped_duplicates % 100 == 0:
                        logger.info(f"Skipped {skipped_duplicates} duplicates so far...")
                    continue

                url, doc = self._process_doc_id(doc_id)

                if doc:
                    if self.store.save_document(doc):
                        collected += 1
                        existing_urls.add(url)
                        pbar.update(1)
                        if collected % 10 == 0:
                            logger.info(f"Progress: {collected} new docs collected")
                    else:
                        existing_urls.add(url)

                # Checkpoint
                if collected > 0 and collected % settings.CHECKPOINT_INTERVAL == 0:
                    progress.documents_collected = collected
                    progress.last_url = url
                    progress.last_updated = datetime.now()
                    self.store.save_progress(progress)
                    logger.info(f"Checkpoint: {collected} documents saved")

        except Exception as e:
            logger.error(f"API scraping error: {e}")

        return collected, skipped_duplicates

    def _scrape_api_concurrent(
        self,
        progress: ScrapingProgress,
        collected: int,
        pbar: tqdm,
        existing_urls: set[str],
        existing_urls_lock: threading.Lock,
    ) -> tuple[int, int]:
        """Concurrent API scraping using ThreadPoolExecutor.

        Returns:
            Tuple of (collected count, skipped duplicates count).
        """
        skipped_duplicates = 0
        doc_id_queue: Queue[str] = Queue(maxsize=self.num_threads * 2)
        last_url = ""

        def doc_id_producer():
            """Produce document IDs to the queue."""
            nonlocal last_url, skipped_duplicates
            for doc_id in self._get_doc_ids_via_api():
                if self._interrupted or collected >= self.target_count:
                    break

                # Skip duplicates before queuing
                url = f"{self.base_url}/doc/{doc_id}/"
                with existing_urls_lock:
                    if url in existing_urls:
                        skipped_duplicates += 1
                        if skipped_duplicates % 100 == 0:
                            logger.info(f"Skipped {skipped_duplicates} duplicates so far...")
                        continue

                doc_id_queue.put(doc_id)
                last_url = url

            # Signal end with None
            for _ in range(self.num_threads):
                doc_id_queue.put(None)

        # Start producer in background
        producer_thread = threading.Thread(target=doc_id_producer, daemon=True)
        producer_thread.start()

        try:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {}
                active_count = 0

                while True:
                    if self._interrupted:
                        logger.warning("Interrupted! Saving progress...")
                        break

                    if collected >= self.target_count:
                        break

                    # Submit new tasks while we have capacity
                    while active_count < self.num_threads:
                        try:
                            doc_id = doc_id_queue.get(timeout=0.1)
                            if doc_id is None:
                                break
                            future = executor.submit(self._process_doc_id, doc_id)
                            futures[future] = doc_id
                            active_count += 1
                        except Empty:
                            break

                    if not futures:
                        break

                    # Process completed futures
                    done_futures = [f for f in futures if f.done()]
                    for future in done_futures:
                        doc_id = futures.pop(future)
                        active_count -= 1

                        try:
                            url, doc = future.result()
                            if doc:
                                if self.store.save_document(doc):
                                    with self._counter_lock:
                                        collected += 1
                                        pbar.update(1)
                                    with existing_urls_lock:
                                        existing_urls.add(url)
                                    if collected % 10 == 0:
                                        logger.info(f"Progress: {collected} new docs collected")
                                else:
                                    with existing_urls_lock:
                                        existing_urls.add(url)
                        except Exception as e:
                            logger.error(f"Error processing doc {doc_id}: {e}")

                    # Checkpoint (thread-safe)
                    with self._counter_lock:
                        if collected > 0 and collected % settings.CHECKPOINT_INTERVAL == 0:
                            progress.documents_collected = collected
                            progress.last_url = last_url
                            progress.last_updated = datetime.now()
                            self.store.save_progress(progress)
                            logger.info(f"Checkpoint: {collected} documents saved")

        except Exception as e:
            logger.error(f"Concurrent API scraping error: {e}")

        return collected, skipped_duplicates

    def scrape(self, resume: bool = True) -> int:
        """Run the scraper - uses API or HTML based on mode setting.

        Args:
            resume: Whether to resume from previous progress.

        Returns:
            Number of documents collected.
        """
        if self.use_api:
            logger.info("Using optimized API-based scraping (POST requests)")
            return self.scrape_via_api(resume)
        else:
            logger.info("Using HTML scraping mode")
            return super().scrape(resume)

    def _get_urls_via_api(self) -> Generator[str, None, None]:
        """Get document URLs using the API."""
        # Build search queries based on court filter
        # Use +doctypes: filter to filter by court type
        if self.court_filter:
            # Map court filters to search queries with doctypes filter
            # Note: Use space before doctypes:, not + (API quirk)
            court_queries = {
                "supremecourt": ["judgment doctypes:supremecourt"],
                "delhi": ["judgment doctypes:delhi"],
                "bombay": ["judgment doctypes:bombay"],
                "karnataka": ["judgment doctypes:karnataka"],
                "highcourts": [
                    "judgment doctypes:delhi",
                    "judgment doctypes:bombay",
                    "judgment doctypes:karnataka",
                ],
            }
            queries = court_queries.get(self.court_filter, [f"judgment doctypes:{self.court_filter}"])
        else:
            # Default: search by legal topics to get diverse results
            # Use doctypes:judgments to filter out laws/sections
            queries = [
                "criminal appeal doctypes:judgments",
                "civil suit doctypes:judgments",
                "constitutional doctypes:judgments",
                "contract doctypes:judgments",
                "property doctypes:judgments",
                "writ petition doctypes:judgments",
                "arbitration doctypes:judgments",
                "tax appeal doctypes:judgments",
            ]

        seen_doc_ids = set()

        for query in queries:
            page = 0
            max_pages = 100  # Limit pages per query

            while page < max_pages:
                try:
                    data = self._api_search(query, page)

                    if not data:
                        logger.warning(f"API search returned no data for '{query}' at page {page}")
                        break

                    docs = data.get("docs", [])
                    if not docs:
                        logger.info(f"No more results for '{query}' at page {page}")
                        break

                    for doc in docs:
                        doc_id = doc.get("tid")
                        if doc_id and doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            yield f"{self.base_url}/doc/{doc_id}/"

                    page += 1

                except Exception as e:
                    logger.error(f"API search failed for '{query}' at page {page}: {e}")
                    break

    def _get_urls_via_html(self) -> Generator[str, None, None]:
        """Get document URLs by scraping search pages."""
        # Search queries - use diverse keywords for better coverage
        if self.court_filter:
            search_queries = [
                f"doctypes:{self.court_filter}",
                f"judgment doctypes:{self.court_filter}",
            ]
        else:
            search_queries = [
                "criminal appeal",
                "civil suit",
                "writ petition",
                "murder",
                "contract",
                "property dispute",
                "divorce",
                "cheating",
                "constitutional",
                "tax appeal",
            ]

        for query in search_queries:
            page = 0

            while True:
                search_url = f"{self.base_url}/search/?formInput={query}&pagenum={page}"

                try:
                    response = self.fetch_page(search_url)
                    soup = self.parse_html(response.text)

                    # Find result links (h4.result_title or div.result_title depending on page version)
                    results = soup.select(".result_title a")
                    if not results:
                        # Also try direct doc links
                        results = soup.select('a[href*="/doc/"]')
                    if not results:
                        logger.info(f"No more results for '{query}' at page {page}")
                        break

                    for result in results:
                        href = result.get("href")
                        if href and "/doc/" in href:
                            yield urljoin(self.base_url, href)

                    page += 1

                except Exception as e:
                    logger.error(f"Search failed at page {page}: {e}")
                    break

    def parse_document(self, url: str, html: str) -> LegalDocument | None:
        """Parse a judgment document.

        Args:
            url: Document URL.
            html: HTML content.

        Returns:
            LegalDocument if parsing succeeded, None otherwise.
        """
        try:
            soup = self.parse_html(html)
            return self._parse_judgment(url, soup)
        except Exception as e:
            logger.error(f"Failed to parse {url}: {e}")
            return None

    def _parse_judgment(self, url: str, soup: BeautifulSoup) -> LegalDocument | None:
        """Parse judgment HTML into LegalDocument."""
        # Get case title - try multiple selectors for different page versions
        title_elem = soup.select_one("h2.doc_title, div.doc_title, h3.doc_title, title")
        if not title_elem:
            # Try the main heading in akn sections
            title_elem = soup.select_one("main#main-content h3, main#main-content h2")
        if not title_elem:
            raise ParseError("Could not find case title")

        case_title = title_elem.get_text(strip=True)
        # Clean up title if it's from the <title> tag
        case_title = re.sub(r"\s*\|\s*Indian Kanoon$", "", case_title)

        # Get full text - try multiple selectors for different page versions
        judgment_elem = soup.select_one(
            "div.judgments, article.judgments, div.doc_content, "
            "main#main-content, div#content, section.akn-section"
        )
        if not judgment_elem:
            # Fallback: get all akn-content spans
            akn_content = soup.select(".akn-content, .akn-p")
            if akn_content:
                full_text = " ".join(elem.get_text(strip=True) for elem in akn_content)
            else:
                raise ParseError("Could not find judgment text")
        else:
            full_text = self._clean_text(judgment_elem.get_text())

        if len(full_text) < 100:
            logger.warning(f"Document too short: {url}")
            return None

        # Extract metadata
        court = self._extract_court(soup, case_title)
        date_decided = self._extract_date(soup)
        citation = self._extract_citation(soup, case_title)
        judges = self._extract_judges(soup)
        petitioner, respondent = self._extract_parties(case_title)
        acts_referred = self._extract_acts(soup, full_text)
        cases_cited = self._extract_cited_cases(soup)

        return LegalDocument(
            source=self.source,
            url=url,
            case_title=case_title,
            court=court,
            citation=citation,
            date_decided=date_decided,
            petitioner=petitioner,
            respondent=respondent,
            judges=judges,
            acts_referred=acts_referred,
            cases_cited=cases_cited,
            full_text=full_text,
        )

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(line for line in lines if line)

    def _extract_court(self, soup: BeautifulSoup, title: str) -> str:
        """Extract court name."""
        # Try to find court info in metadata
        # Note: Some pages use h1.docsource_main instead of div.docsource_main
        court_elem = soup.select_one("h1.docsource_main, div.docsource_main, span.doc_court")
        if court_elem:
            return court_elem.get_text(strip=True)

        # Infer from title or URL
        title_lower = title.lower()
        if "supreme court" in title_lower:
            return "Supreme Court of India"
        if "delhi" in title_lower and "high court" in title_lower:
            return "Delhi High Court"
        if "bombay" in title_lower and "high court" in title_lower:
            return "Bombay High Court"
        if "karnataka" in title_lower and "high court" in title_lower:
            return "Karnataka High Court"
        if "high court" in title_lower:
            return "High Court"

        return "Unknown Court"

    def _extract_date(self, soup: BeautifulSoup) -> date | None:
        """Extract judgment date."""
        # Look for date in metadata
        date_elem = soup.select_one("div.doc_date, span.doc_date, div.docsource_date")
        if date_elem:
            try:
                date_text = date_elem.get_text(strip=True)
                return parse_date(date_text, fuzzy=True).date()
            except Exception:
                pass

        # Look for date pattern in text
        text = soup.get_text()
        date_patterns = [
            r"(?:dated|decided|judgment)\s*:?\s*(\d{1,2}[\s/-]\w+[\s/-]\d{4})",
            r"(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s*,?\s*\d{4})",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return parse_date(match.group(1), fuzzy=True).date()
                except Exception:
                    continue

        return None

    def _extract_citation(self, soup: BeautifulSoup, title: str) -> str | None:
        """Extract case citation."""
        # Common citation patterns
        citation_patterns = [
            r"\d{4}\s+SCC\s+\d+",
            r"\d{4}\s+\(\d+\)\s+SCC\s+\d+",
            r"AIR\s+\d{4}\s+\w+\s+\d+",
            r"\d{4}\s+\(\d+\)\s+SCR\s+\d+",
            r"\d{4}\s+Cri\.?L\.?J\.?\s+\d+",
        ]

        text = soup.get_text()
        for pattern in citation_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)

        return None

    def _extract_judges(self, soup: BeautifulSoup) -> list[str]:
        """Extract judge names."""
        judges = []

        # Look for bench/coram information
        # Note: Some pages use h3.doc_bench instead of div.doc_bench
        bench_elem = soup.select_one("h3.doc_bench, div.doc_bench, h3.doc_author, div.doc_author")
        if bench_elem:
            text = bench_elem.get_text()
            # Remove "Author:" or "Bench:" prefix
            text = re.sub(r"^(Author|Bench)\s*:\s*", "", text, flags=re.IGNORECASE)
            # Split by common separators
            parts = re.split(r"[,&]|and", text, flags=re.IGNORECASE)
            for part in parts:
                judge = part.strip()
                if judge and len(judge) > 2:
                    # Clean up common prefixes
                    judge = re.sub(r"^(Hon'?ble\s+|Mr\.?\s+|Justice\s+|J\.\s*)", "", judge, flags=re.IGNORECASE)
                    if judge:
                        judges.append(judge.strip())

        return judges

    def _extract_parties(self, title: str) -> tuple[str | None, str | None]:
        """Extract petitioner and respondent from case title."""
        # Common patterns: "X v. Y", "X vs Y", "X versus Y"
        match = re.search(r"(.+?)\s+(?:v\.?|vs\.?|versus)\s+(.+)", title, re.IGNORECASE)
        if match:
            petitioner = match.group(1).strip()
            respondent = match.group(2).strip()
            # Remove trailing date like "on 28 March, 2018"
            respondent = re.sub(r"\s+on\s+\d{1,2}\s+\w+,?\s*\d{4}.*$", "", respondent, flags=re.IGNORECASE)
            return petitioner, respondent
        return None, None

    def _extract_acts(self, soup: BeautifulSoup, full_text: str) -> list[str]:
        """Extract referenced acts/statutes."""
        acts = set()

        # Common act patterns
        act_patterns = [
            r"(?:the\s+)?(\w+(?:\s+\w+)*?\s+Act,?\s*\d{4})",
            r"(?:the\s+)?(\w+(?:\s+\w+)*?\s+Code,?\s*\d{4})",
            r"(?:under|section\s+\d+\s+of)\s+(?:the\s+)?(\w+(?:\s+\w+)*?\s+Act)",
        ]

        for pattern in act_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for match in matches:
                act = match.strip()
                if len(act) > 5 and len(act) < 100:
                    acts.add(act)

        return list(acts)[:20]  # Limit to 20 acts

    def _extract_cited_cases(self, soup: BeautifulSoup) -> list[str]:
        """Extract cited case names."""
        cited = set()

        # Look for links to other cases
        case_links = soup.select("a[href*='/doc/']")
        for link in case_links:
            case_name = link.get_text(strip=True)
            if case_name and " v" in case_name.lower():
                cited.add(case_name)

        return list(cited)[:50]  # Limit to 50 citations
