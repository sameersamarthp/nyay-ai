"""
Abstract base class for all scrapers.

Supports concurrent fetching using ThreadPoolExecutor.
"""

import signal
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from queue import Queue, Empty
from typing import Generator

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tqdm import tqdm

from config.settings import settings
from storage.document_store import DocumentStore
from storage.schemas import DocumentSource, LegalDocument, ScrapingProgress
from utils.logger import get_logger
from utils.rate_limiter import AdaptiveRateLimiter
from utils.retry import with_retry, FetchError

logger = get_logger(__name__)


class BaseScraper(ABC):
    """Abstract base class for legal document scrapers."""

    source: DocumentSource  # Must be set by subclasses

    def __init__(
        self,
        store: DocumentStore | None = None,
        target_count: int | None = None,
        num_threads: int = 1,
    ):
        """Initialize scraper.

        Args:
            store: Document store instance. Creates new if not provided.
            target_count: Number of documents to collect.
            num_threads: Number of concurrent threads for fetching (default: 1).
        """
        self.store = store or DocumentStore()
        self.target_count = target_count or self._get_default_target()
        self.num_threads = max(1, num_threads)
        self.rate_limiter = AdaptiveRateLimiter()
        self.session = self._create_session()
        self._interrupted = False
        self._setup_signal_handlers()

        # Thread-local storage for sessions (each thread gets its own session)
        self._thread_local = threading.local()

        # Lock for thread-safe counter updates
        self._counter_lock = threading.Lock()

        # Initialize UserAgent for rotation
        self._ua = None
        if settings.ROTATE_USER_AGENT:
            try:
                self._ua = UserAgent()
            except Exception:
                logger.warning("Failed to initialize UserAgent, using static UA")

        if self.num_threads > 1:
            logger.info(f"Concurrent fetching enabled with {self.num_threads} threads")

    def _get_default_target(self) -> int:
        """Get default target count for this source."""
        targets = {
            DocumentSource.INDIAN_KANOON: settings.TARGET_INDIAN_KANOON,
            DocumentSource.SUPREME_COURT: settings.TARGET_SUPREME_COURT,
            DocumentSource.HIGH_COURTS: settings.TARGET_HIGH_COURTS,
            DocumentSource.INDIA_CODE: settings.TARGET_INDIA_CODE,
        }
        return targets.get(self.source, 1000)

    def _create_session(self) -> requests.Session:
        """Create a requests session with default headers."""
        session = requests.Session()
        session.headers.update(self._get_headers())
        return session

    def _get_thread_session(self) -> requests.Session:
        """Get thread-local session (creates one if needed).

        Returns:
            Thread-local requests.Session instance.
        """
        if not hasattr(self._thread_local, "session"):
            self._thread_local.session = self._create_session()
        return self._thread_local.session

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for requests."""
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        if settings.ROTATE_USER_AGENT:
            try:
                ua = UserAgent()
                headers["User-Agent"] = ua.random
            except Exception:
                headers["User-Agent"] = (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
        else:
            # Use standard browser User-Agent (many legal sites block bot UAs)
            headers["User-Agent"] = (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )

        return headers

    def _setup_signal_handlers(self) -> None:
        """Setup handlers for graceful shutdown."""

        def handler(signum, frame):
            logger.warning("Interrupt received, saving progress...")
            self._interrupted = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _rotate_user_agent(self, session: requests.Session | None = None) -> str:
        """Rotate User-Agent for the next request.

        Args:
            session: Session to update. Uses thread-local session if None.

        Returns:
            The new User-Agent string.
        """
        if session is None:
            session = self._get_thread_session()

        if self._ua:
            new_ua = self._ua.random
            session.headers["User-Agent"] = new_ua
            return new_ua
        return session.headers.get("User-Agent", "Unknown")

    @with_retry()
    def fetch_page(self, url: str) -> requests.Response:
        """Fetch a page with rate limiting and retry (thread-safe).

        Args:
            url: URL to fetch.

        Returns:
            Response object.

        Raises:
            FetchError: If fetching fails after retries.
        """
        # Get thread-local session
        session = self._get_thread_session()

        with self.rate_limiter.acquire():
            # Rotate User-Agent before each request
            user_agent = self._rotate_user_agent(session)
            logger.debug(f"User-Agent: {user_agent}")

            try:
                response = session.get(url, timeout=settings.REQUEST_TIMEOUT)

                if response.status_code == 429:
                    self.rate_limiter.record_error(is_rate_limit=True)
                    self.rate_limiter.backoff()
                    raise FetchError(f"Rate limited: {url}")

                response.raise_for_status()
                self.rate_limiter.record_success()
                return response

            except requests.RequestException as e:
                self.rate_limiter.record_error()
                logger.error(f"Failed to fetch {url}: {e}")
                raise FetchError(f"Failed to fetch {url}: {e}") from e

    def parse_html(self, html: str) -> BeautifulSoup:
        """Parse HTML content.

        Args:
            html: HTML string to parse.

        Returns:
            BeautifulSoup object.
        """
        return BeautifulSoup(html, "lxml")

    @abstractmethod
    def get_document_urls(self) -> Generator[str, None, None]:
        """Generate URLs of documents to scrape.

        Yields:
            Document URLs.
        """
        pass

    @abstractmethod
    def parse_document(self, url: str, html: str) -> LegalDocument | None:
        """Parse a document from HTML.

        Args:
            url: Document URL.
            html: HTML content.

        Returns:
            LegalDocument if parsing succeeded, None otherwise.
        """
        pass

    def _process_single_url(self, url: str) -> LegalDocument | None:
        """Process a single URL and return the document (thread-safe).

        Args:
            url: URL to process.

        Returns:
            LegalDocument if successful, None otherwise.
        """
        try:
            response = self.fetch_page(url)
            return self.parse_document(url, response.text)
        except FetchError as e:
            logger.warning(f"Skipping {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return None

    def scrape(self, resume: bool = True) -> int:
        """Run the scraper with optional concurrent fetching.

        Args:
            resume: Whether to resume from previous progress.

        Returns:
            Number of documents collected.
        """
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
            f"Starting {self.source.value} scraper: "
            f"{collected}/{self.target_count} collected"
            f" (threads: {self.num_threads})"
        )

        # Progress bar
        pbar = tqdm(
            initial=collected,
            total=self.target_count,
            desc=self.source.value,
            unit="docs",
        )

        # Use single-threaded or concurrent based on num_threads
        if self.num_threads <= 1:
            collected = self._scrape_sequential(progress, collected, pbar)
        else:
            collected = self._scrape_concurrent(progress, collected, pbar)

        pbar.close()

        # Save final progress
        progress.documents_collected = collected
        progress.last_updated = datetime.now()
        progress.is_complete = collected >= self.target_count
        self.store.save_progress(progress)

        logger.info(
            f"{self.source.value}: Collected {collected}/{self.target_count} documents"
        )

        return collected

    def _scrape_sequential(
        self,
        progress: ScrapingProgress,
        collected: int,
        pbar: tqdm,
    ) -> int:
        """Sequential scraping (single thread).

        Args:
            progress: Scraping progress object.
            collected: Current count of collected documents.
            pbar: Progress bar.

        Returns:
            Updated collected count.
        """
        try:
            for url in self.get_document_urls():
                if self._interrupted:
                    logger.warning("Interrupted! Saving progress...")
                    break

                if collected >= self.target_count:
                    break

                doc = self._process_single_url(url)

                if doc:
                    if self.store.save_document(doc):
                        collected += 1
                        pbar.update(1)
                    else:
                        logger.debug(f"Duplicate: {url}")

                # Checkpoint
                if collected > 0 and collected % settings.CHECKPOINT_INTERVAL == 0:
                    progress.documents_collected = collected
                    progress.last_url = url
                    progress.last_updated = datetime.now()
                    self.store.save_progress(progress)
                    logger.info(f"Checkpoint: {collected} documents saved")

        except Exception as e:
            logger.error(f"Scraping error: {e}")

        return collected

    def _scrape_concurrent(
        self,
        progress: ScrapingProgress,
        collected: int,
        pbar: tqdm,
    ) -> int:
        """Concurrent scraping using ThreadPoolExecutor.

        Args:
            progress: Scraping progress object.
            collected: Current count of collected documents.
            pbar: Progress bar.

        Returns:
            Updated collected count.
        """
        # URL queue for feeding threads
        url_queue: Queue[str] = Queue(maxsize=self.num_threads * 2)
        last_url = ""

        def url_producer():
            """Produce URLs to the queue."""
            nonlocal last_url
            for url in self.get_document_urls():
                if self._interrupted or collected >= self.target_count:
                    break
                url_queue.put(url)
                last_url = url
            # Signal end with None
            for _ in range(self.num_threads):
                url_queue.put(None)

        # Start URL producer in background
        producer_thread = threading.Thread(target=url_producer, daemon=True)
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
                            url = url_queue.get(timeout=0.1)
                            if url is None:
                                # End of URLs
                                break
                            future = executor.submit(self._process_single_url, url)
                            futures[future] = url
                            active_count += 1
                        except Empty:
                            break

                    if not futures:
                        # No more work
                        break

                    # Process completed futures
                    done_futures = [f for f in futures if f.done()]
                    for future in done_futures:
                        url = futures.pop(future)
                        active_count -= 1

                        try:
                            doc = future.result()
                            if doc:
                                if self.store.save_document(doc):
                                    with self._counter_lock:
                                        collected += 1
                                        pbar.update(1)
                                else:
                                    logger.debug(f"Duplicate: {url}")
                        except Exception as e:
                            logger.error(f"Error processing {url}: {e}")

                    # Checkpoint (thread-safe)
                    with self._counter_lock:
                        if collected > 0 and collected % settings.CHECKPOINT_INTERVAL == 0:
                            progress.documents_collected = collected
                            progress.last_url = last_url
                            progress.last_updated = datetime.now()
                            self.store.save_progress(progress)
                            logger.info(f"Checkpoint: {collected} documents saved")

        except Exception as e:
            logger.error(f"Concurrent scraping error: {e}")

        return collected

    def dry_run(self, count: int = 10) -> list[LegalDocument]:
        """Test scraper with a small number of documents.

        Args:
            count: Number of documents to fetch.

        Returns:
            List of parsed documents.
        """
        documents = []
        logger.info(
            f"Dry run: Fetching {count} documents from {self.source.value}"
            f" (threads: {self.num_threads})"
        )

        if self.num_threads <= 1:
            # Sequential dry run
            for i, url in enumerate(self.get_document_urls()):
                if i >= count:
                    break

                try:
                    response = self.fetch_page(url)
                    doc = self.parse_document(url, response.text)

                    if doc:
                        documents.append(doc)
                        logger.info(f"[{len(documents)}/{count}] Parsed: {doc.case_title[:50]}...")

                except Exception as e:
                    logger.error(f"Error: {e}")
        else:
            # Concurrent dry run
            urls = []
            for i, url in enumerate(self.get_document_urls()):
                if i >= count * 2:  # Fetch extra in case some fail
                    break
                urls.append(url)

            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(self._process_single_url, url): url
                    for url in urls
                }

                for future in as_completed(futures):
                    if len(documents) >= count:
                        break

                    url = futures[future]
                    try:
                        doc = future.result()
                        if doc:
                            documents.append(doc)
                            logger.info(f"[{len(documents)}/{count}] Parsed: {doc.case_title[:50]}...")
                    except Exception as e:
                        logger.error(f"Error fetching {url}: {e}")

        logger.info(f"Dry run complete: {len(documents)}/{count} documents parsed")
        return documents
