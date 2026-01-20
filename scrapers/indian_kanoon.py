"""
Indian Kanoon scraper.

Primary source for legal documents. Supports both API access (preferred)
and HTML scraping as fallback.

API documentation: https://api.indiankanoon.org/
"""

import re
from datetime import date
from typing import Generator
from urllib.parse import urljoin, urlencode

from bs4 import BeautifulSoup
from dateutil.parser import parse as parse_date

from config.settings import settings
from storage.document_store import DocumentStore
from storage.schemas import DocumentSource, LegalDocument
from scrapers.base_scraper import BaseScraper
from utils.logger import get_logger
from utils.retry import ParseError

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
    ):
        """Initialize Indian Kanoon scraper.

        Args:
            store: Document store instance.
            target_count: Number of documents to collect.
            api_token: Indian Kanoon API token (if available).
            court_filter: Filter for specific court (e.g., "supremecourt").
        """
        # Set attributes before super().__init__() since _get_headers needs them
        self.api_token = api_token or settings.INDIAN_KANOON_API_TOKEN
        self.court_filter = court_filter
        self.base_url = settings.INDIAN_KANOON_BASE_URL
        self.api_url = settings.INDIAN_KANOON_API_URL
        self._current_page = 0

        super().__init__(store, target_count)

        if self.api_token:
            logger.info("Using Indian Kanoon API")
        else:
            logger.info("Using Indian Kanoon HTML scraping (no API token)")

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
        if self.api_token:
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
        # Search queries for different categories
        search_queries = [
            f"fromdate:{settings.DATE_START} todate:{settings.DATE_END}",
        ]

        if self.court_filter:
            search_queries = [f"doctypes:{self.court_filter}"]

        for query in search_queries:
            page = 0

            while True:
                search_url = f"{self.base_url}/search/?formInput={query}&pagenum={page}"

                try:
                    response = self.fetch_page(search_url)
                    soup = self.parse_html(response.text)

                    # Find result links
                    results = soup.select("div.result_title a")
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
        # Get case title
        title_elem = soup.select_one("h2.doc_title, div.doc_title, title")
        if not title_elem:
            raise ParseError("Could not find case title")

        case_title = title_elem.get_text(strip=True)

        # Get full text
        # Note: Mobile version uses <article class="judgments"> instead of <div class="judgments">
        judgment_elem = soup.select_one("div.judgments, article.judgments, div.doc_content, div#content")
        if not judgment_elem:
            raise ParseError("Could not find judgment text")

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
