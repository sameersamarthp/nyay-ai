"""
India Code scraper for central acts and statutes.

Source: https://www.indiacode.nic.in/
Official repository of all Central Acts.

Structure:
- Central Acts: /handle/123456789/1362 (main collection)
- Browse by title: /handle/123456789/1362/browse?type=shorttitle
- Individual acts: /handle/123456789/XXXXX?view_type=browse
- Acts are PDFs but pages have metadata
"""

import re
from datetime import date
from typing import Generator
from urllib.parse import urljoin, quote, unquote

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

    # Collection handles for different categories
    CENTRAL_ACTS_HANDLE = "123456789/1362"
    UNION_TERRITORIES_HANDLE = "123456789/14011"

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
        seen_urls = set()

        # Browse Central Acts by short title with pagination
        collections = [
            self.CENTRAL_ACTS_HANDLE,
            self.UNION_TERRITORIES_HANDLE,
        ]

        for handle in collections:
            offset = 0
            rpp = 100  # Results per page

            while True:
                browse_url = (
                    f"{self.base_url}/handle/{handle}/browse"
                    f"?type=shorttitle&rpp={rpp}&offset={offset}"
                )

                try:
                    response = self.fetch_page(browse_url)
                    soup = self.parse_html(response.text)

                    # Find act links in table rows
                    # Pattern: /handle/123456789/XXXXX?view_type=browse
                    act_links = soup.select('a[href*="view_type=browse"]')

                    if not act_links:
                        # Also try direct handle links in table cells
                        act_links = soup.select('td a[href*="/handle/123456789/"]')

                    found_new = False
                    for link in act_links:
                        href = link.get("href")
                        if href and "/handle/123456789/" in href:
                            # Skip browse and collection links
                            if "browse" in href and "view_type" not in href:
                                continue

                            full_url = urljoin(self.base_url, href)

                            # Ensure view_type=browse for proper page
                            if "view_type" not in full_url:
                                full_url = full_url.rstrip("/") + "?view_type=browse"

                            if full_url not in seen_urls:
                                seen_urls.add(full_url)
                                found_new = True
                                yield full_url

                    # Check if we found any new acts
                    if not found_new:
                        logger.info(f"No more acts found at offset {offset}")
                        break

                    # Move to next page
                    offset += rpp

                    # Safety limit
                    if offset > 2000:
                        logger.info(f"Reached offset limit for {handle}")
                        break

                except Exception as e:
                    logger.warning(f"Failed to browse {handle} at offset {offset}: {e}")
                    break

        # Also search for major acts by name
        major_acts = [
            "Indian Penal Code",
            "Code of Criminal Procedure",
            "Code of Civil Procedure",
            "Indian Evidence Act",
            "Indian Contract Act",
            "Constitution of India",
            "Companies Act",
            "Income Tax Act",
            "Goods and Services Tax",
            "Right to Information Act",
            "Motor Vehicles Act",
            "Negotiable Instruments Act",
            "Transfer of Property Act",
            "Specific Relief Act",
            "Limitation Act",
            "Arbitration and Conciliation Act",
            "Consumer Protection Act",
            "Information Technology Act",
            "Prevention of Corruption Act",
            "Narcotic Drugs",
        ]

        for act_name in major_acts:
            search_url = f"{self.base_url}/simple-search?query={quote(act_name)}&rpp=20"
            try:
                response = self.fetch_page(search_url)
                soup = self.parse_html(response.text)

                # Find act links in search results
                for link in soup.select('a[href*="/handle/123456789/"]'):
                    href = link.get("href")
                    if href and "browse?type" not in href:
                        full_url = urljoin(self.base_url, href)
                        if "view_type" not in full_url:
                            full_url = full_url.rstrip("/") + "?view_type=browse"

                        if full_url not in seen_urls:
                            seen_urls.add(full_url)
                            yield full_url

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
        # Get act title from <title> tag or meta
        title_elem = soup.select_one("title")
        if not title_elem:
            raise ParseError("Could not find act title")

        raw_title = title_elem.get_text(strip=True)

        # Clean up title - remove "India Code: " prefix
        act_title = re.sub(r"^India\s*Code:\s*", "", raw_title, flags=re.IGNORECASE)
        act_title = re.sub(r"\s+", " ", act_title).strip()

        if not act_title or act_title == "India Code":
            # Try alternate selectors
            alt_title = soup.select_one("#short_title, h1, h2.artifact-title")
            if alt_title:
                act_title = alt_title.get_text(strip=True)
            else:
                raise ParseError("Could not find valid act title")

        # Get PDF URL if available
        pdf_meta = soup.select_one('meta[name="citation_pdf_url"]')
        pdf_url = pdf_meta.get("content") if pdf_meta else None

        # Extract text from the page
        # Look for act details, preamble, sections
        text_parts = []

        # Add title
        text_parts.append(f"Title: {act_title}")

        # Look for act metadata table
        metadata_rows = soup.select("table.table tr")
        for row in metadata_rows:
            cells = row.select("td, th")
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True)
                value = cells[-1].get_text(strip=True)
                if key and value and len(value) < 500:
                    text_parts.append(f"{key}: {value}")

        # Look for any visible text content
        content_divs = soup.select("div.panel-body, div.tab-content, div.container")
        for div in content_divs:
            text = div.get_text(strip=True)
            if text and len(text) > 50 and len(text) < 5000:
                # Avoid duplicates and navigation text
                if "Browse" not in text and "Login" not in text:
                    text_parts.append(text)

        # If we have very little content, try to get everything
        if len("\n".join(text_parts)) < 200:
            main_content = soup.select_one("div#content, main, article")
            if main_content:
                full_text = self._clean_text(main_content.get_text())
                text_parts = [full_text]

        full_text = "\n\n".join(text_parts)
        full_text = self._clean_text(full_text)

        if len(full_text) < 50:
            logger.warning(f"Document too short: {url}")
            return None

        # Extract metadata
        year = self._extract_year(act_title, soup)
        act_number = self._extract_act_number(soup, act_title)
        enactment_date = self._extract_enactment_date(soup)
        ministry = self._extract_ministry(soup)

        # Add PDF URL reference to text if available
        if pdf_url:
            full_text = f"[PDF available at: {pdf_url}]\n\n{full_text}"

        return LegalDocument(
            source=self.source,
            url=url,
            case_title=act_title,
            court="Parliament of India",
            citation=act_number,
            date_decided=enactment_date or (date(year, 1, 1) if year else None),
            full_text=full_text,
            subject_category="Legislation",
            headnotes=ministry,  # Store ministry in headnotes field
        )

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove common website artifacts
        text = re.sub(
            r"(Skip to|Back to|Home|Login|Register|Sign in|Show All|Loading)\s*",
            "", text, flags=re.IGNORECASE
        )
        # Remove JavaScript artifacts
        text = re.sub(r"\$\([^)]+\)[^;]*;", "", text)
        text = re.sub(r"function\s*\([^)]*\)\s*\{[^}]*\}", "", text)

        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(line for line in lines if line and len(line) > 2)

    def _extract_year(self, title: str, soup: BeautifulSoup) -> int | None:
        """Extract year of enactment."""
        # Try to find year in title - match both formats: "2023" or "(1920 A.D.)"
        year_patterns = [
            r"\b(19\d{2}|20[0-2]\d)\b",
            r"\((\d{4})\s*A\.?D\.?\)",
        ]

        for pattern in year_patterns:
            match = re.search(pattern, title)
            if match:
                return int(match.group(1))

        # Try metadata
        for elem in soup.select("td, span, div"):
            text = elem.get_text()
            if "year" in text.lower() or "enacted" in text.lower():
                for pattern in year_patterns:
                    match = re.search(pattern, text)
                    if match:
                        return int(match.group(1))

        return None

    def _extract_act_number(self, soup: BeautifulSoup, title: str) -> str | None:
        """Extract act number (e.g., 'Act No. 45 of 1860')."""
        patterns = [
            r"Act\s+No\.?\s*\d+\s+of\s+\d{4}",
            r"No\.?\s*\d+\s+of\s+\d{4}",
            r"Act\s+\d+\s+of\s+\d{4}",
        ]

        text = soup.get_text() + " " + title
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)

        return None

    def _extract_enactment_date(self, soup: BeautifulSoup) -> date | None:
        """Extract enactment date."""
        # Look for date in metadata
        date_patterns = [
            r"(\d{1,2}[-/]\w+[-/]\d{4})",
            r"(\d{1,2}\s+\w+\s+\d{4})",
        ]

        for elem in soup.select("td"):
            text = elem.get_text(strip=True)
            for pattern in date_patterns:
                match = re.search(pattern, text)
                if match:
                    try:
                        return parse_date(match.group(1), fuzzy=True).date()
                    except Exception:
                        continue

        return None

    def _extract_ministry(self, soup: BeautifulSoup) -> str | None:
        """Extract ministry/department name."""
        for elem in soup.select("td, span"):
            text = elem.get_text(strip=True)
            if "Ministry" in text or "Department" in text:
                return text[:200]  # Limit length
        return None
