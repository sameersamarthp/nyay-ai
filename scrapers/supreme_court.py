"""
Supreme Court of India scraper.

Uses Indian Kanoon as the primary source for Supreme Court judgments.
This provides better metadata extraction and more reliable access.
"""

from typing import Generator

from config.settings import settings
from storage.document_store import DocumentStore
from storage.schemas import DocumentSource, LegalDocument
from scrapers.indian_kanoon import IndianKanoonScraper
from utils.logger import get_logger

logger = get_logger(__name__)


class SupremeCourtScraper(IndianKanoonScraper):
    """Scraper for Supreme Court of India judgments via Indian Kanoon."""

    source = DocumentSource.SUPREME_COURT

    def __init__(
        self,
        store: DocumentStore | None = None,
        target_count: int | None = None,
        api_token: str | None = None,
        use_api: bool | None = None,
    ):
        """Initialize Supreme Court scraper.

        Args:
            store: Document store instance.
            target_count: Number of documents to collect.
            api_token: Indian Kanoon API token (if available).
            use_api: Force API (True), force HTML (False), or auto-detect (None).
        """
        super().__init__(
            store=store,
            target_count=target_count or settings.TARGET_SUPREME_COURT,
            api_token=api_token,
            court_filter="supremecourt",
            use_api=use_api,
        )
        logger.info("Supreme Court scraper initialized (via Indian Kanoon)")

    def parse_document(self, url: str, html: str) -> LegalDocument | None:
        """Parse a Supreme Court judgment.

        Overrides parent to ensure proper source tagging.

        Args:
            url: Document URL.
            html: HTML content.

        Returns:
            LegalDocument with SUPREME_COURT source.
        """
        doc = super().parse_document(url, html)

        if doc:
            # Ensure correct source and court
            doc.source = DocumentSource.SUPREME_COURT
            if doc.court == "Unknown Court":
                doc.court = "Supreme Court of India"

        return doc
