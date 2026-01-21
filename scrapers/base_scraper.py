"""
Abstract base class for all scrapers.
"""

import signal
from abc import ABC, abstractmethod
from datetime import datetime
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
    ):
        """Initialize scraper.

        Args:
            store: Document store instance. Creates new if not provided.
            target_count: Number of documents to collect.
        """
        self.store = store or DocumentStore()
        self.target_count = target_count or self._get_default_target()
        self.rate_limiter = AdaptiveRateLimiter()
        self.session = self._create_session()
        self._interrupted = False
        self._setup_signal_handlers()

        # Initialize UserAgent for rotation
        self._ua = None
        if settings.ROTATE_USER_AGENT:
            try:
                self._ua = UserAgent()
            except Exception:
                logger.warning("Failed to initialize UserAgent, using static UA")

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

    def _rotate_user_agent(self) -> str:
        """Rotate User-Agent for the next request.

        Returns:
            The new User-Agent string.
        """
        if self._ua:
            new_ua = self._ua.random
            self.session.headers["User-Agent"] = new_ua
            return new_ua
        return self.session.headers.get("User-Agent", "Unknown")

    @with_retry()
    def fetch_page(self, url: str) -> requests.Response:
        """Fetch a page with rate limiting and retry.

        Args:
            url: URL to fetch.

        Returns:
            Response object.

        Raises:
            FetchError: If fetching fails after retries.
        """
        with self.rate_limiter.acquire():
            # Rotate User-Agent before each request
            user_agent = self._rotate_user_agent()
            logger.debug(f"User-Agent: {user_agent}")

            try:
                response = self.session.get(url, timeout=settings.REQUEST_TIMEOUT)

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

    def scrape(self, resume: bool = True) -> int:
        """Run the scraper.

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
        )

        # Progress bar
        pbar = tqdm(
            initial=collected,
            total=self.target_count,
            desc=self.source.value,
            unit="docs",
        )

        try:
            for url in self.get_document_urls():
                if self._interrupted:
                    logger.warning("Interrupted! Saving progress...")
                    break

                if collected >= self.target_count:
                    break

                try:
                    response = self.fetch_page(url)
                    doc = self.parse_document(url, response.text)

                    if doc:
                        if self.store.save_document(doc):
                            collected += 1
                            pbar.update(1)
                        else:
                            logger.debug(f"Duplicate: {url}")

                except FetchError as e:
                    logger.warning(f"Skipping {url}: {e}")
                    continue

                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
                    continue

                # Checkpoint
                if collected % settings.CHECKPOINT_INTERVAL == 0:
                    progress.documents_collected = collected
                    progress.last_url = url
                    progress.last_updated = datetime.now()
                    self.store.save_progress(progress)
                    logger.info(f"Checkpoint: {collected} documents saved")

        finally:
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

    def dry_run(self, count: int = 10) -> list[LegalDocument]:
        """Test scraper with a small number of documents.

        Args:
            count: Number of documents to fetch.

        Returns:
            List of parsed documents.
        """
        documents = []
        logger.info(f"Dry run: Fetching {count} documents from {self.source.value}")

        for i, url in enumerate(self.get_document_urls()):
            if i >= count:
                break

            try:
                response = self.fetch_page(url)
                doc = self.parse_document(url, response.text)

                if doc:
                    documents.append(doc)
                    logger.info(f"[{i + 1}/{count}] Parsed: {doc.case_title[:50]}...")

            except Exception as e:
                logger.error(f"Error: {e}")

        logger.info(f"Dry run complete: {len(documents)}/{count} documents parsed")
        return documents
