"""
Tests for storage module.
"""

import tempfile
from datetime import date, datetime
from pathlib import Path

import pytest

from storage.schemas import DocumentSource, LegalDocument, ScrapingProgress
from storage.document_store import DocumentStore


class TestLegalDocument:
    """Tests for LegalDocument schema."""

    def test_create_document(self):
        """Test creating a basic document."""
        doc = LegalDocument(
            source=DocumentSource.INDIAN_KANOON,
            url="https://indiankanoon.org/doc/123/",
            case_title="State v. ABC",
            court="Supreme Court of India",
            full_text="This is the judgment text. " * 100,
        )

        assert doc.doc_id  # Should be auto-generated
        assert doc.source == DocumentSource.INDIAN_KANOON
        assert doc.word_count > 0

    def test_doc_id_generation(self):
        """Test that doc_id is generated from case details."""
        doc1 = LegalDocument(
            source=DocumentSource.INDIAN_KANOON,
            url="https://indiankanoon.org/doc/123/",
            case_title="State v. ABC",
            court="Supreme Court of India",
            full_text="Test text",
        )

        doc2 = LegalDocument(
            source=DocumentSource.INDIAN_KANOON,
            url="https://indiankanoon.org/doc/456/",
            case_title="State v. ABC",
            court="Supreme Court of India",
            full_text="Test text",
        )

        # Same case details should generate same doc_id
        assert doc1.doc_id == doc2.doc_id

    def test_different_docs_different_ids(self):
        """Test that different cases get different IDs."""
        doc1 = LegalDocument(
            source=DocumentSource.INDIAN_KANOON,
            url="https://indiankanoon.org/doc/123/",
            case_title="State v. ABC",
            court="Supreme Court of India",
            full_text="Test text",
        )

        doc2 = LegalDocument(
            source=DocumentSource.INDIAN_KANOON,
            url="https://indiankanoon.org/doc/456/",
            case_title="State v. XYZ",
            court="Supreme Court of India",
            full_text="Test text",
        )

        assert doc1.doc_id != doc2.doc_id

    def test_to_dict_and_back(self):
        """Test serialization round-trip."""
        doc = LegalDocument(
            source=DocumentSource.SUPREME_COURT,
            url="https://example.com/doc/1",
            citation="2023 SCC 456",
            case_title="ABC v. DEF",
            court="Supreme Court of India",
            date_decided=date(2023, 5, 15),
            judges=["Justice A", "Justice B"],
            full_text="Full judgment text here.",
        )

        data = doc.to_dict()
        restored = LegalDocument.from_dict(data)

        assert restored.doc_id == doc.doc_id
        assert restored.citation == doc.citation
        assert restored.date_decided == doc.date_decided
        assert restored.judges == doc.judges


class TestDocumentStore:
    """Tests for DocumentStore."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary document store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "test.db"
            # Temporarily override settings
            from config.settings import settings
            original_db = settings.DB_PATH
            original_raw = settings.RAW_DATA_DIR
            original_project_root = settings.PROJECT_ROOT

            settings.DB_PATH = db_path
            settings.RAW_DATA_DIR = tmpdir_path / "raw"
            settings.PROJECT_ROOT = tmpdir_path

            store = DocumentStore(db_path=db_path)
            yield store

            # Restore settings
            settings.DB_PATH = original_db
            settings.RAW_DATA_DIR = original_raw
            settings.PROJECT_ROOT = original_project_root

    def test_save_and_retrieve(self, temp_store):
        """Test saving and retrieving a document."""
        doc = LegalDocument(
            source=DocumentSource.INDIAN_KANOON,
            url="https://indiankanoon.org/doc/123/",
            case_title="Test Case",
            court="Test Court",
            full_text="This is the test content.",
        )

        # Save document
        saved = temp_store.save_document(doc)
        assert saved is True

        # Retrieve document
        retrieved = temp_store.get_document(doc.doc_id)
        assert retrieved is not None
        assert retrieved.case_title == doc.case_title
        assert retrieved.full_text == doc.full_text

    def test_no_duplicates(self, temp_store):
        """Test that duplicates are not saved."""
        doc = LegalDocument(
            source=DocumentSource.INDIAN_KANOON,
            url="https://indiankanoon.org/doc/123/",
            case_title="Test Case",
            court="Test Court",
            full_text="This is the test content.",
        )

        # First save should succeed
        assert temp_store.save_document(doc) is True

        # Second save should return False (duplicate)
        assert temp_store.save_document(doc) is False

        # Count should be 1
        assert temp_store.count_documents() == 1

    def test_count_by_source(self, temp_store):
        """Test counting documents by source."""
        # Save documents from different sources
        for i in range(3):
            doc = LegalDocument(
                source=DocumentSource.INDIAN_KANOON,
                url=f"https://indiankanoon.org/doc/{i}/",
                case_title=f"IK Case {i}",
                court="Test Court",
                full_text=f"Content {i}",
            )
            temp_store.save_document(doc)

        for i in range(2):
            doc = LegalDocument(
                source=DocumentSource.SUPREME_COURT,
                url=f"https://sci.gov.in/doc/{i}/",
                case_title=f"SC Case {i}",
                court="Supreme Court",
                full_text=f"Content {i}",
            )
            temp_store.save_document(doc)

        assert temp_store.count_documents(DocumentSource.INDIAN_KANOON) == 3
        assert temp_store.count_documents(DocumentSource.SUPREME_COURT) == 2
        assert temp_store.count_documents() == 5

    def test_progress_tracking(self, temp_store):
        """Test scraping progress tracking."""
        progress = ScrapingProgress(
            source=DocumentSource.INDIAN_KANOON,
            total_target=5000,
            documents_collected=100,
            last_page=10,
        )

        temp_store.save_progress(progress)

        # Retrieve progress
        retrieved = temp_store.get_progress(DocumentSource.INDIAN_KANOON)
        assert retrieved is not None
        assert retrieved.documents_collected == 100
        assert retrieved.last_page == 10

    def test_stats(self, temp_store):
        """Test statistics generation."""
        # Add some documents
        doc = LegalDocument(
            source=DocumentSource.INDIAN_KANOON,
            url="https://indiankanoon.org/doc/1/",
            case_title="Test Case",
            court="Supreme Court of India",
            full_text="Test content",
        )
        temp_store.save_document(doc)

        stats = temp_store.get_stats()
        assert stats["total_documents"] == 1
        assert stats["by_source"]["indian_kanoon"] == 1
