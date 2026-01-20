"""
High Courts scraper.

Uses Indian Kanoon as the primary source for High Court judgments.
Focuses on Delhi, Bombay, and Karnataka High Courts.
"""

from typing import Generator

from config.settings import settings
from storage.document_store import DocumentStore
from storage.schemas import DocumentSource, LegalDocument
from scrapers.indian_kanoon import IndianKanoonScraper
from utils.logger import get_logger

logger = get_logger(__name__)


# Mapping of court codes to full names
HIGH_COURT_CODES = {
    "delhi": "Delhi High Court",
    "bombay": "Bombay High Court",
    "karnataka": "Karnataka High Court",
    "calcutta": "Calcutta High Court",
    "madras": "Madras High Court",
    "allahabad": "Allahabad High Court",
}


class HighCourtsScraper(IndianKanoonScraper):
    """Scraper for High Court judgments via Indian Kanoon."""

    source = DocumentSource.HIGH_COURTS

    def __init__(
        self,
        store: DocumentStore | None = None,
        target_count: int | None = None,
        api_token: str | None = None,
        courts: list[str] | None = None,
    ):
        """Initialize High Courts scraper.

        Args:
            store: Document store instance.
            target_count: Number of documents to collect.
            api_token: Indian Kanoon API token (if available).
            courts: List of court codes to scrape (e.g., ["delhi", "bombay"]).
                   Defaults to settings.HIGH_COURTS.
        """
        super().__init__(
            store=store,
            target_count=target_count or settings.TARGET_HIGH_COURTS,
            api_token=api_token,
            court_filter="highcourts",
        )

        # Courts to scrape
        self.courts = courts or ["delhi", "bombay", "karnataka"]
        self._current_court_index = 0

        logger.info(
            f"High Courts scraper initialized for: {', '.join(self.courts)} "
            "(via Indian Kanoon)"
        )

    def get_document_urls(self) -> Generator[str, None, None]:
        """Generate document URLs from multiple high courts.

        Rotates through courts to get balanced coverage.

        Yields:
            Document URLs.
        """
        # Calculate docs per court
        docs_per_court = self.target_count // len(self.courts)
        extra = self.target_count % len(self.courts)

        for i, court_code in enumerate(self.courts):
            target = docs_per_court + (1 if i < extra else 0)
            count = 0

            logger.info(f"Scraping {HIGH_COURT_CODES.get(court_code, court_code)}: target {target}")

            # Update court filter for this specific court
            original_filter = self.court_filter
            self.court_filter = court_code

            for url in super().get_document_urls():
                if count >= target:
                    break
                yield url
                count += 1

            # Restore original filter
            self.court_filter = original_filter

    def parse_document(self, url: str, html: str) -> LegalDocument | None:
        """Parse a High Court judgment.

        Args:
            url: Document URL.
            html: HTML content.

        Returns:
            LegalDocument with HIGH_COURTS source.
        """
        doc = super().parse_document(url, html)

        if doc:
            # Ensure correct source
            doc.source = DocumentSource.HIGH_COURTS

            # Try to identify specific high court
            if doc.court == "Unknown Court" or doc.court == "High Court":
                doc.court = self._identify_court(url, doc.case_title)

        return doc

    def _identify_court(self, url: str, title: str) -> str:
        """Identify which High Court from URL or title."""
        text = f"{url} {title}".lower()

        for code, name in HIGH_COURT_CODES.items():
            if code in text:
                return name

        return "High Court"
