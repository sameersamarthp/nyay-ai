"""
AWS High Court document storage using SQLite.

Stores metadata and full text in SQLite for the AWS dataset.
Optimized for bulk inserts and PDF processing updates.
"""

import threading
from pathlib import Path
from typing import Any

from sqlite_utils import Database

from config.settings import settings
from storage.aws_schemas import AWSHighCourtDocument, AWSProcessingProgress
from utils.logger import get_logger

logger = get_logger(__name__)


class AWSDocumentStore:
    """Handles storage of AWS High Court documents using SQLite."""

    def __init__(self, db_path: Path | None = None):
        """Initialize AWS document store.

        Args:
            db_path: Path to SQLite database. Defaults to settings.DB_PATH.
        """
        self.db_path = db_path or settings.DB_PATH

        # Ensure directories exist
        settings.ensure_directories()

        # Initialize database
        self.db = Database(self.db_path)
        self._init_tables()

        # Lock for thread-safe operations
        self._lock = threading.Lock()

    def _init_tables(self) -> None:
        """Create database tables if they don't exist."""
        # AWS documents table
        if "aws_documents" not in self.db.table_names():
            self.db["aws_documents"].create(
                {
                    "cnr": str,  # Primary key
                    "doc_id": str,
                    "court_code": str,
                    "title": str,
                    "description": str,
                    "judge": str,
                    "pdf_link": str,
                    "date_of_registration": str,
                    "decision_date": str,
                    "disposal_nature": str,
                    "court": str,
                    "full_text": str,
                    "pdf_processed": int,
                    "word_count": int,
                    "year": int,
                    "bench": str,
                    "created_at": str,
                },
                pk="cnr",
            )
            # Create indexes for common queries
            self.db["aws_documents"].create_index(["court"], if_not_exists=True)
            self.db["aws_documents"].create_index(["decision_date"], if_not_exists=True)
            self.db["aws_documents"].create_index(["year"], if_not_exists=True)
            self.db["aws_documents"].create_index(["bench"], if_not_exists=True)
            self.db["aws_documents"].create_index(["pdf_processed"], if_not_exists=True)
            self.db["aws_documents"].create_index(["court_code", "bench"], if_not_exists=True)
            logger.info("Created aws_documents table")

        # Processing progress table
        if "aws_processing_progress" not in self.db.table_names():
            self.db["aws_processing_progress"].create(
                {
                    "id": str,  # court_code:bench:year
                    "court_code": str,
                    "bench": str,
                    "year": int,
                    "total_documents": int,
                    "documents_processed": int,
                    "last_cnr": str,
                    "last_updated": str,
                    "is_complete": int,
                },
                pk="id",
            )
            logger.info("Created aws_processing_progress table")

    def document_exists(self, cnr: str) -> bool:
        """Check if a document already exists by CNR."""
        return self.db["aws_documents"].count_where("cnr = ?", [cnr]) > 0

    def save_document(self, doc: AWSHighCourtDocument) -> bool:
        """Save a single AWS document to storage.

        Args:
            doc: AWSHighCourtDocument to save.

        Returns:
            True if saved successfully, False if duplicate.
        """
        with self._lock:
            if self.document_exists(doc.cnr):
                logger.debug(f"Document {doc.cnr} already exists, skipping")
                return False

            self.db["aws_documents"].insert(doc.to_dict())
            logger.debug(f"Saved document: {doc.cnr}")
            return True

    def bulk_insert_documents(self, docs: list[AWSHighCourtDocument], batch_size: int = 1000) -> int:
        """Bulk insert documents efficiently.

        Args:
            docs: List of AWSHighCourtDocument to insert.
            batch_size: Number of documents per batch.

        Returns:
            Number of documents inserted.
        """
        inserted = 0
        with self._lock:
            # Convert to dicts
            records = [doc.to_dict() for doc in docs]

            # Insert in batches, ignoring duplicates
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                self.db["aws_documents"].insert_all(batch, ignore=True)
                inserted += len(batch)
                logger.debug(f"Inserted batch {i // batch_size + 1}: {len(batch)} documents")

        return inserted

    def update_full_text(self, cnr: str, full_text: str) -> bool:
        """Update the full_text field for a document after PDF processing.

        Args:
            cnr: Document CNR (primary key).
            full_text: Extracted text from PDF.

        Returns:
            True if updated successfully.
        """
        word_count = len(full_text.split()) if full_text else 0
        with self._lock:
            self.db["aws_documents"].update(
                cnr,
                {"full_text": full_text, "pdf_processed": 1, "word_count": word_count},
            )
        return True

    def bulk_update_full_text(self, updates: list[tuple[str, str]]) -> int:
        """Bulk update full_text for multiple documents.

        Args:
            updates: List of (cnr, full_text) tuples.

        Returns:
            Number of documents updated.
        """
        updated = 0
        with self._lock:
            for cnr, full_text in updates:
                word_count = len(full_text.split()) if full_text else 0
                self.db["aws_documents"].update(
                    cnr,
                    {"full_text": full_text, "pdf_processed": 1, "word_count": word_count},
                )
                updated += 1
        return updated

    def get_unprocessed_documents(
        self, court_code: str | None = None, bench: str | None = None, limit: int | None = None
    ) -> list[dict]:
        """Get documents that haven't had PDF processing yet.

        Args:
            court_code: Filter by court code.
            bench: Filter by bench.
            limit: Maximum number to return.

        Returns:
            List of document dicts with cnr and pdf_link.
        """
        where_clauses = ["pdf_processed = 0"]
        params = []

        if court_code:
            where_clauses.append("court_code = ?")
            params.append(court_code)
        if bench:
            where_clauses.append("bench = ?")
            params.append(bench)

        where = " AND ".join(where_clauses)
        rows = self.db["aws_documents"].rows_where(
            where, params, select="cnr, pdf_link, court_code, bench", limit=limit
        )
        return list(rows)

    def count_documents(
        self, court_code: str | None = None, bench: str | None = None, processed_only: bool = False
    ) -> int:
        """Count documents with optional filters.

        Args:
            court_code: Filter by court code.
            bench: Filter by bench.
            processed_only: Only count PDF-processed documents.

        Returns:
            Document count.
        """
        where_clauses = []
        params = []

        if court_code:
            where_clauses.append("court_code = ?")
            params.append(court_code)
        if bench:
            where_clauses.append("bench = ?")
            params.append(bench)
        if processed_only:
            where_clauses.append("pdf_processed = 1")

        if where_clauses:
            return self.db["aws_documents"].count_where(" AND ".join(where_clauses), params)
        return self.db["aws_documents"].count

    def get_document(self, cnr: str) -> AWSHighCourtDocument | None:
        """Retrieve a document by CNR.

        Args:
            cnr: Document CNR.

        Returns:
            AWSHighCourtDocument if found, None otherwise.
        """
        rows = list(self.db["aws_documents"].rows_where("cnr = ?", [cnr]))
        if not rows:
            return None
        return AWSHighCourtDocument.from_dict(rows[0])

    def get_existing_cnrs(self, court_code: str | None = None, bench: str | None = None) -> set[str]:
        """Get all existing CNRs for fast duplicate checking.

        Args:
            court_code: Filter by court code.
            bench: Filter by bench.

        Returns:
            Set of CNRs.
        """
        where_clauses = []
        params = []

        if court_code:
            where_clauses.append("court_code = ?")
            params.append(court_code)
        if bench:
            where_clauses.append("bench = ?")
            params.append(bench)

        if where_clauses:
            rows = self.db["aws_documents"].rows_where(
                " AND ".join(where_clauses), params, select="cnr"
            )
        else:
            rows = self.db["aws_documents"].rows_where(select="cnr")

        return {row["cnr"] for row in rows}

    # Progress tracking methods
    def save_progress(self, progress: AWSProcessingProgress) -> None:
        """Save or update processing progress."""
        progress_id = f"{progress.court_code}:{progress.bench}:{progress.year}"
        data = progress.to_dict()
        data["id"] = progress_id
        with self._lock:
            self.db["aws_processing_progress"].upsert(data, pk="id")

    def get_progress(self, court_code: str, bench: str, year: int) -> AWSProcessingProgress | None:
        """Get processing progress for a specific court/bench/year."""
        progress_id = f"{court_code}:{bench}:{year}"
        rows = list(self.db["aws_processing_progress"].rows_where("id = ?", [progress_id]))
        if not rows:
            return None
        return AWSProcessingProgress.from_dict(rows[0])

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "total_documents": self.db["aws_documents"].count,
            "processed_documents": self.db["aws_documents"].count_where("pdf_processed = 1"),
            "unprocessed_documents": self.db["aws_documents"].count_where("pdf_processed = 0"),
            "by_court": {},
            "by_bench": {},
        }

        # Count by court
        for row in self.db.execute(
            "SELECT court, COUNT(*) as count FROM aws_documents GROUP BY court"
        ).fetchall():
            stats["by_court"][row[0]] = row[1]

        # Count by bench
        for row in self.db.execute(
            "SELECT bench, COUNT(*) as count FROM aws_documents GROUP BY bench"
        ).fetchall():
            stats["by_bench"][row[0]] = row[1]

        return stats

