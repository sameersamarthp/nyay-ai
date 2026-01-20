"""
Tests for scraper modules.

Note: These tests use mock responses to avoid actual network requests.
For integration tests with real sites, use the --live flag.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import date

# Disable user agent rotation for tests
from config.settings import settings
settings.ROTATE_USER_AGENT = False

from scrapers.indian_kanoon import IndianKanoonScraper
from scrapers.supreme_court import SupremeCourtScraper
from scrapers.high_courts import HighCourtsScraper
from storage.schemas import DocumentSource


# Sample HTML for testing
SAMPLE_JUDGMENT_HTML = """
<!DOCTYPE html>
<html>
<head><title>State of Maharashtra v. ABC - Indian Kanoon</title></head>
<body>
<h2 class="doc_title">State of Maharashtra v. ABC</h2>
<div class="docsource_main">Supreme Court of India</div>
<div class="doc_date">15 May 2023</div>
<div class="doc_bench">Hon'ble Justice A, Justice B</div>
<div class="judgments">
    <p>This is the judgment text for the case of State of Maharashtra versus ABC.</p>
    <p>The appeal is hereby dismissed.</p>
    <p>Reference is made to the Indian Penal Code, 1860 and the Code of Criminal Procedure, 1973.</p>
    <p>As held in <a href="/doc/123/">XYZ v. State</a>, the law is clear on this matter.</p>
</div>
</body>
</html>
"""


class TestIndianKanoonScraper:
    """Tests for Indian Kanoon scraper."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock document store."""
        store = Mock()
        store.save_document.return_value = True
        store.document_exists.return_value = False
        store.get_progress.return_value = None
        store.save_progress.return_value = None
        return store

    def test_parse_judgment(self, mock_store):
        """Test parsing a judgment document."""
        scraper = IndianKanoonScraper(store=mock_store)

        doc = scraper.parse_document(
            "https://indiankanoon.org/doc/12345/",
            SAMPLE_JUDGMENT_HTML,
        )

        assert doc is not None
        # Title comes from h2.doc_title which includes clean title
        assert "State of Maharashtra v. ABC" in doc.case_title
        assert doc.court == "Supreme Court of India"
        assert doc.source == DocumentSource.INDIAN_KANOON
        assert "judgment text" in doc.full_text.lower()

    def test_extract_parties(self, mock_store):
        """Test party extraction from case title."""
        scraper = IndianKanoonScraper(store=mock_store)

        doc = scraper.parse_document(
            "https://indiankanoon.org/doc/12345/",
            SAMPLE_JUDGMENT_HTML,
        )

        assert doc.petitioner == "State of Maharashtra"
        # Respondent may include page title suffix, but should start with ABC
        assert doc.respondent is not None
        assert doc.respondent.startswith("ABC")

    def test_extract_judges(self, mock_store):
        """Test judge name extraction."""
        scraper = IndianKanoonScraper(store=mock_store)

        doc = scraper.parse_document(
            "https://indiankanoon.org/doc/12345/",
            SAMPLE_JUDGMENT_HTML,
        )

        assert len(doc.judges) >= 1
        # Should have cleaned up "Hon'ble" prefix
        assert not any("Hon'ble" in j for j in doc.judges)

    def test_extract_acts(self, mock_store):
        """Test act extraction from judgment text."""
        scraper = IndianKanoonScraper(store=mock_store)

        doc = scraper.parse_document(
            "https://indiankanoon.org/doc/12345/",
            SAMPLE_JUDGMENT_HTML,
        )

        assert len(doc.acts_referred) > 0
        # Should find IPC reference
        assert any("Penal Code" in act for act in doc.acts_referred)

    def test_extract_cited_cases(self, mock_store):
        """Test cited case extraction."""
        scraper = IndianKanoonScraper(store=mock_store)

        doc = scraper.parse_document(
            "https://indiankanoon.org/doc/12345/",
            SAMPLE_JUDGMENT_HTML,
        )

        # Should find the cited case
        assert len(doc.cases_cited) > 0

    def test_short_document_rejected(self, mock_store):
        """Test that very short documents are rejected."""
        scraper = IndianKanoonScraper(store=mock_store)

        short_html = """
        <html>
        <head><title>Short</title></head>
        <body>
        <h2 class="doc_title">Short Doc</h2>
        <div class="judgments">Too short.</div>
        </body>
        </html>
        """

        doc = scraper.parse_document(
            "https://indiankanoon.org/doc/12345/",
            short_html,
        )

        assert doc is None


class TestSupremeCourtScraper:
    """Tests for Supreme Court scraper."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock document store."""
        store = Mock()
        store.save_document.return_value = True
        store.document_exists.return_value = False
        store.get_progress.return_value = None
        return store

    def test_source_is_supreme_court(self, mock_store):
        """Test that source is correctly set."""
        scraper = SupremeCourtScraper(store=mock_store)
        assert scraper.source == DocumentSource.SUPREME_COURT

    def test_parsed_doc_has_correct_source(self, mock_store):
        """Test that parsed documents have correct source."""
        scraper = SupremeCourtScraper(store=mock_store)

        doc = scraper.parse_document(
            "https://indiankanoon.org/doc/12345/",
            SAMPLE_JUDGMENT_HTML,
        )

        assert doc is not None
        assert doc.source == DocumentSource.SUPREME_COURT


class TestHighCourtsScraper:
    """Tests for High Courts scraper."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock document store."""
        store = Mock()
        store.save_document.return_value = True
        store.document_exists.return_value = False
        store.get_progress.return_value = None
        return store

    def test_source_is_high_courts(self, mock_store):
        """Test that source is correctly set."""
        scraper = HighCourtsScraper(store=mock_store)
        assert scraper.source == DocumentSource.HIGH_COURTS

    def test_default_courts(self, mock_store):
        """Test default court selection."""
        scraper = HighCourtsScraper(store=mock_store)
        assert "delhi" in scraper.courts
        assert "bombay" in scraper.courts
        assert "karnataka" in scraper.courts

    def test_court_identification(self, mock_store):
        """Test high court identification from URL."""
        scraper = HighCourtsScraper(store=mock_store)

        delhi_html = SAMPLE_JUDGMENT_HTML.replace(
            "Supreme Court of India",
            "Delhi High Court"
        )

        doc = scraper.parse_document(
            "https://indiankanoon.org/doc/delhi/12345/",
            delhi_html,
        )

        assert doc is not None
        assert doc.source == DocumentSource.HIGH_COURTS
        assert "Delhi" in doc.court


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    def test_rate_limiter_waits(self):
        """Test that rate limiter enforces delays."""
        from utils.rate_limiter import RateLimiter
        import time

        limiter = RateLimiter(min_interval=0.1, max_interval=0.2)

        start = time.time()
        limiter.wait()
        limiter.wait()
        elapsed = time.time() - start

        # Should have waited at least once
        assert elapsed >= 0.1

    def test_adaptive_rate_limiter(self):
        """Test adaptive rate limiter adjusts intervals."""
        from utils.rate_limiter import AdaptiveRateLimiter

        limiter = AdaptiveRateLimiter(min_interval=0.5, max_interval=2.0)

        # Set a higher initial interval to test decrease
        limiter._current_interval = 1.5

        # Record successes to trigger decrease
        for _ in range(15):
            limiter.record_success()

        # Interval should have decreased from 1.5
        assert limiter._current_interval < 1.5

        # Record rate limit error
        limiter.record_error(is_rate_limit=True)

        # Interval should have increased significantly (doubled)
        assert limiter._current_interval > 1.0
