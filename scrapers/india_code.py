"""
India Code scraper for central acts and statutes.

Source: https://www.indiacode.nic.in/
Official repository of all Central Acts.
"""

import re
from datetime import date
from typing import Generator
from urllib.parse import urljoin, quote

from bs4 import BeautifulSoup
from dateutil.parser import parse as parse_date

from config.settings import settings
from storage.document_store import DocumentStore
from storage.schemas import DocumentSource, LegalDocument
from scrapers.base_scraper import BaseScraper
from utils.logger import get_logger
from utils.retry import ParseError

logger = get_logger(__name__)


class IndiaCodeScraper(BaseScraper):
    """Scraper for India Code (central acts and statutes)."""

    source = DocumentSource.INDIA_CODE

    def __init__(
        self,
        store: DocumentStore | None = None,
        target_count: int | None = None,
    ):
        """Initialize India Code scraper.

        Args:
            store: Document store instance.
            target_count: Number of acts to collect.
        """
        super().__init__(store, target_count or settings.TARGET_INDIA_CODE)
        self.base_url = settings.INDIA_CODE_BASE_URL

        logger.info("India Code scraper initialized")

    def get_document_urls(self) -> Generator[str, None, None]:
        """Generate URLs for acts from India Code.

        Yields:
            Act page URLs.
        """
        # India Code organizes acts by year
        # We'll focus on major acts and recent legislation

        # Start with the acts listing page
        listing_url = f"{self.base_url}/handle/123456789/1362"

        try:
            response = self.fetch_page(listing_url)
            soup = self.parse_html(response.text)

            # Find links to acts
            act_links = soup.select("a[href*='/handle/']")

            for link in act_links:
                href = link.get("href")
                if href and "/handle/" in href:
                    yield urljoin(self.base_url, href)

        except Exception as e:
            logger.warning(f"Failed to get listing page: {e}")

        # Also try browsing by year for recent acts
        for year in range(2024, 2018, -1):
            browse_url = f"{self.base_url}/browse?type=year&value={year}"

            try:
                response = self.fetch_page(browse_url)
                soup = self.parse_html(response.text)

                act_links = soup.select("a[href*='/handle/']")
                for link in act_links:
                    href = link.get("href")
                    if href:
                        yield urljoin(self.base_url, href)

            except Exception as e:
                logger.warning(f"Failed to browse year {year}: {e}")
                continue

        # Add some well-known major acts (these are commonly searched)
        major_acts = [
            "Indian Penal Code",
            "Code of Civil Procedure",
            "Code of Criminal Procedure",
            "Indian Contract Act",
            "Indian Evidence Act",
            "Constitution of India",
            "Companies Act",
            "Income Tax Act",
            "Motor Vehicles Act",
            "Negotiable Instruments Act",
        ]

        for act_name in major_acts:
            search_url = f"{self.base_url}/search?query={quote(act_name)}"
            try:
                response = self.fetch_page(search_url)
                soup = self.parse_html(response.text)

                act_links = soup.select("a[href*='/handle/']")
                for link in act_links[:3]:  # Top 3 results
                    href = link.get("href")
                    if href:
                        yield urljoin(self.base_url, href)

            except Exception as e:
                logger.warning(f"Failed to search for {act_name}: {e}")

    def parse_document(self, url: str, html: str) -> LegalDocument | None:
        """Parse an act/statute page.

        Args:
            url: Act URL.
            html: HTML content.

        Returns:
            LegalDocument if parsing succeeded, None otherwise.
        """
        try:
            soup = self.parse_html(html)
            return self._parse_act(url, soup)
        except Exception as e:
            logger.error(f"Failed to parse {url}: {e}")
            return None

    def _parse_act(self, url: str, soup: BeautifulSoup) -> LegalDocument | None:
        """Parse act HTML into LegalDocument."""
        # Get act title
        title_elem = soup.select_one(
            "h1, h2.artifact-title, div.item-summary-view-metadata h1, title"
        )
        if not title_elem:
            raise ParseError("Could not find act title")

        act_title = title_elem.get_text(strip=True)

        # Clean up title
        act_title = re.sub(r"\s+", " ", act_title).strip()

        # Get full text
        content_elem = soup.select_one(
            "div.fulltext, div.item-page-field-wrapper, div#content, article"
        )

        if content_elem:
            full_text = self._clean_text(content_elem.get_text())
        else:
            # Fallback: get all text from page
            full_text = self._clean_text(soup.get_text())

        if len(full_text) < 100:
            logger.warning(f"Document too short: {url}")
            return None

        # Extract metadata
        year = self._extract_year(act_title, soup)
        act_number = self._extract_act_number(soup, act_title)

        return LegalDocument(
            source=self.source,
            url=url,
            case_title=act_title,  # Using case_title for act title
            court="Parliament of India",  # Central acts are from Parliament
            citation=act_number,
            date_decided=date(year, 1, 1) if year else None,
            full_text=full_text,
            subject_category="Legislation",
        )

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove common website artifacts
        text = re.sub(r"(Skip to|Back to|Home|Login|Register)\s*", "", text, flags=re.IGNORECASE)
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(line for line in lines if line)

    def _extract_year(self, title: str, soup: BeautifulSoup) -> int | None:
        """Extract year of enactment."""
        # Try to find year in title
        year_match = re.search(r"\b(1[89]\d{2}|20[0-2]\d)\b", title)
        if year_match:
            return int(year_match.group(1))

        # Try metadata
        year_elem = soup.select_one("span.date-issued, td.date")
        if year_elem:
            try:
                date_text = year_elem.get_text(strip=True)
                year_match = re.search(r"\b(1[89]\d{2}|20[0-2]\d)\b", date_text)
                if year_match:
                    return int(year_match.group(1))
            except Exception:
                pass

        return None

    def _extract_act_number(self, soup: BeautifulSoup, title: str) -> str | None:
        """Extract act number (e.g., 'Act No. 45 of 1860')."""
        # Look for act number pattern
        patterns = [
            r"Act\s+No\.?\s*\d+\s+of\s+\d{4}",
            r"No\.?\s*\d+\s+of\s+\d{4}",
        ]

        text = soup.get_text() + " " + title
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)

        return None
