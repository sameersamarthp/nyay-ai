"""
Document storage using SQLite + JSON files.

SQLite for metadata and fast queries, JSON for full document content.
Thread-safe for concurrent access.
"""

import json
import threading
from pathlib import Path
from typing import Any
from sqlite_utils import Database

from config.settings import settings
from storage.schemas import DocumentSource, LegalDocument, ScrapingProgress
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentStore:
    """Handles storage of legal documents using SQLite and JSON."""

    def __init__(self, db_path: Path | None = None):
        """Initialize document store.

        Args:
            db_path: Path to SQLite database. Defaults to settings.DB_PATH.
        """
        self.db_path = db_path or settings.DB_PATH
        self.raw_dir = settings.RAW_DATA_DIR

        # Ensure directories exist
        settings.ensure_directories()

        # Initialize database
        self.db = Database(self.db_path)
        self._init_tables()

        # Lock for thread-safe operations
        self._lock = threading.Lock()

    def _init_tables(self) -> None:
        """Create database tables if they don't exist."""
        # Documents metadata table
        if "documents" not in self.db.table_names():
            self.db["documents"].create(
                {
                    "doc_id": str,
                    "source": str,
                    "url": str,
                    "citation": str,
                    "case_number": str,
                    "case_title": str,
                    "court": str,
                    "petitioner": str,
                    "respondent": str,
                    "date_decided": str,
                    "subject_category": str,
                    "outcome": str,
                    "word_count": int,
                    "scraped_at": str,
                    "is_landmark": int,
                    "json_path": str,
                },
                pk="doc_id",
            )
            # Create indexes for common queries
            self.db["documents"].create_index(["source"], if_not_exists=True)
            self.db["documents"].create_index(["court"], if_not_exists=True)
            self.db["documents"].create_index(["date_decided"], if_not_exists=True)
            self.db["documents"].create_index(["subject_category"], if_not_exists=True)
            logger.info("Created documents table")

        # Scraping progress table
        if "scraping_progress" not in self.db.table_names():
            self.db["scraping_progress"].create(
                {
                    "source": str,
                    "total_target": int,
                    "documents_collected": int,
                    "last_page": int,
                    "last_url": str,
                    "last_updated": str,
                    "is_complete": int,
                },
                pk="source",
            )
            logger.info("Created scraping_progress table")

    def save_document(self, doc: LegalDocument) -> bool:
        """Save a legal document to storage (thread-safe).

        Args:
            doc: LegalDocument to save.

        Returns:
            True if saved successfully, False if duplicate.
        """
        with self._lock:
            # Check for existing document
            if self.document_exists(doc.doc_id):
                logger.debug(f"Document {doc.doc_id} already exists, skipping")
                return False

            # Save full document as JSON
            json_path = self._get_json_path(doc)
            json_path.parent.mkdir(parents=True, exist_ok=True)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)

            # Save metadata to SQLite
            metadata = {
                "doc_id": doc.doc_id,
                "source": doc.source.value,
                "url": doc.url,
                "citation": doc.citation,
                "case_number": doc.case_number,
                "case_title": doc.case_title,
                "court": doc.court,
                "petitioner": doc.petitioner,
                "respondent": doc.respondent,
                "date_decided": doc.date_decided.isoformat() if doc.date_decided else None,
                "subject_category": doc.subject_category,
                "outcome": doc.outcome,
                "word_count": doc.word_count,
                "scraped_at": doc.scraped_at.isoformat(),
                "is_landmark": int(doc.is_landmark),
                "json_path": str(json_path.relative_to(settings.PROJECT_ROOT)),
            }
            self.db["documents"].insert(metadata)
            logger.debug(f"Saved document: {doc.doc_id}")
            return True

    def _get_json_path(self, doc: LegalDocument) -> Path:
        """Get JSON file path for a document."""
        # Organize by source and first 2 chars of doc_id
        return self.raw_dir / doc.source.value / doc.doc_id[:2] / f"{doc.doc_id}.json"

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document already exists."""
        return self.db["documents"].count_where("doc_id = ?", [doc_id]) > 0

    def get_document(self, doc_id: str) -> LegalDocument | None:
        """Retrieve a document by ID.

        Args:
            doc_id: Document ID to retrieve.

        Returns:
            LegalDocument if found, None otherwise.
        """
        rows = list(self.db["documents"].rows_where("doc_id = ?", [doc_id]))
        if not rows:
            return None

        row = rows[0]
        json_path = settings.PROJECT_ROOT / row["json_path"]

        if not json_path.exists():
            logger.warning(f"JSON file missing for document {doc_id}")
            return None

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        return LegalDocument.from_dict(data)

    def count_documents(self, source: DocumentSource | None = None) -> int:
        """Count documents in storage.

        Args:
            source: Filter by source. None for all documents.

        Returns:
            Document count.
        """
        if source:
            return self.db["documents"].count_where("source = ?", [source.value])
        return self.db["documents"].count

    def get_documents_by_source(
        self, source: DocumentSource, limit: int | None = None
    ) -> list[LegalDocument]:
        """Get documents from a specific source.

        Args:
            source: Document source to filter by.
            limit: Maximum number to return.

        Returns:
            List of LegalDocuments.
        """
        query = "source = ?"
        params = [source.value]

        rows = self.db["documents"].rows_where(query, params, limit=limit)
        documents = []

        for row in rows:
            doc = self.get_document(row["doc_id"])
            if doc:
                documents.append(doc)

        return documents

    def get_existing_urls(self, source: DocumentSource) -> set[str]:
        """Get all existing document URLs for a source (for fast duplicate checking).

        Args:
            source: Document source to filter by.

        Returns:
            Set of document URLs.
        """
        rows = self.db["documents"].rows_where(
            "source = ?", [source.value], select="url"
        )
        return {row["url"] for row in rows}

    def save_progress(self, progress: ScrapingProgress) -> None:
        """Save or update scraping progress (thread-safe).

        Args:
            progress: ScrapingProgress to save.
        """
        with self._lock:
            self.db["scraping_progress"].upsert(
                progress.to_dict(),
                pk="source",
            )
            logger.debug(f"Saved progress for {progress.source.value}: {progress.documents_collected} docs")

    def get_progress(self, source: DocumentSource) -> ScrapingProgress | None:
        """Get scraping progress for a source.

        Args:
            source: Document source.

        Returns:
            ScrapingProgress if found, None otherwise.
        """
        rows = list(
            self.db["scraping_progress"].rows_where("source = ?", [source.value])
        )
        if not rows:
            return None
        return ScrapingProgress.from_dict(rows[0])

    def get_all_progress(self) -> dict[DocumentSource, ScrapingProgress]:
        """Get progress for all sources."""
        result = {}
        for row in self.db["scraping_progress"].rows:
            progress = ScrapingProgress.from_dict(row)
            result[progress.source] = progress
        return result

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "total_documents": self.db["documents"].count,
            "by_source": {},
            "by_court": {},
        }

        # Count by source
        for source in DocumentSource:
            count = self.count_documents(source)
            stats["by_source"][source.value] = count

        # Count by court
        for row in self.db.execute(
            "SELECT court, COUNT(*) as count FROM documents GROUP BY court"
        ).fetchall():
            stats["by_court"][row[0]] = row[1]

        return stats

