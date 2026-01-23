"""
Storage for training data generation progress and examples.
"""

import json
import threading
from pathlib import Path

from sqlite_utils import Database

from config.settings import settings
from storage.training_schemas import (
    TrainingExample,
    TrainingProgress,
    TrainingRunMetadata,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class TrainingStore:
    """Handles storage of training data and progress."""

    def __init__(self, db_path: Path | None = None):
        """Initialize training store.

        Args:
            db_path: Path to SQLite database. Defaults to settings.DB_PATH.
        """
        self.db_path = db_path or settings.DB_PATH
        self.output_dir = settings.TRAINING_DATA_DIR

        # Ensure directories exist
        settings.ensure_directories()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db = Database(self.db_path)
        self._init_tables()

        # Lock for thread-safe operations
        self._lock = threading.Lock()

    def _init_tables(self) -> None:
        """Create database tables if they don't exist."""
        # Training progress table
        if "training_generation_progress" not in self.db.table_names():
            self.db["training_generation_progress"].create(
                {
                    "cnr": str,
                    "status": str,
                    "task_types_generated": str,
                    "examples_generated": int,
                    "error_message": str,
                    "retry_count": int,
                    "created_at": str,
                    "updated_at": str,
                    "input_tokens": int,
                    "output_tokens": int,
                },
                pk="cnr",
            )
            self.db["training_generation_progress"].create_index(
                ["status"], if_not_exists=True
            )
            logger.info("Created training_generation_progress table")

        # Training examples table
        if "training_examples" not in self.db.table_names():
            self.db["training_examples"].create(
                {
                    "id": str,
                    "cnr": str,
                    "task_type": str,
                    "instruction": str,
                    "input": str,
                    "output": str,
                    "created_at": str,
                    "split": str,
                    "input_tokens": int,
                    "output_tokens": int,
                },
                pk="id",
            )
            self.db["training_examples"].create_index(
                ["task_type"], if_not_exists=True
            )
            self.db["training_examples"].create_index(["split"], if_not_exists=True)
            self.db["training_examples"].create_index(["cnr"], if_not_exists=True)
            logger.info("Created training_examples table")

        # Run metadata table
        if "training_run_metadata" not in self.db.table_names():
            self.db["training_run_metadata"].create(
                {
                    "run_id": str,
                    "started_at": str,
                    "completed_at": str,
                    "status": str,
                    "total_documents": int,
                    "documents_processed": int,
                    "examples_generated": int,
                    "total_input_tokens": int,
                    "total_output_tokens": int,
                    "estimated_cost": float,
                    "config_snapshot": str,
                },
                pk="run_id",
            )
            logger.info("Created training_run_metadata table")

    # Progress tracking methods
    def save_progress(self, progress: TrainingProgress) -> None:
        """Save or update document progress."""
        with self._lock:
            self.db["training_generation_progress"].upsert(
                progress.to_db_dict(), pk="cnr"
            )

    def get_progress(self, cnr: str) -> TrainingProgress | None:
        """Get progress for a document."""
        rows = list(
            self.db["training_generation_progress"].rows_where("cnr = ?", [cnr])
        )
        if not rows:
            return None
        return TrainingProgress.from_db_dict(rows[0])

    def get_pending_cnrs(self, limit: int | None = None) -> list[str]:
        """Get CNRs that haven't been processed yet."""
        rows = self.db["training_generation_progress"].rows_where(
            "status = ?", ["pending"], select="cnr", limit=limit
        )
        return [row["cnr"] for row in rows]

    def get_failed_cnrs(self, max_retries: int = 3) -> list[str]:
        """Get CNRs that failed but can be retried."""
        rows = self.db["training_generation_progress"].rows_where(
            "status = ? AND retry_count < ?",
            ["failed", max_retries],
            select="cnr",
        )
        return [row["cnr"] for row in rows]

    def init_progress_for_documents(self, cnrs: list[str]) -> int:
        """Initialize progress tracking for a list of documents.

        Args:
            cnrs: List of document CNRs.

        Returns:
            Number of new progress records created.
        """
        existing = set(
            row["cnr"]
            for row in self.db["training_generation_progress"].rows_where(
                select="cnr"
            )
        )

        new_records = []
        for cnr in cnrs:
            if cnr not in existing:
                progress = TrainingProgress(cnr=cnr)
                new_records.append(progress.to_db_dict())

        if new_records:
            with self._lock:
                self.db["training_generation_progress"].insert_all(new_records)

        return len(new_records)

    # Example storage methods
    def save_example(self, example: TrainingExample) -> None:
        """Save a training example."""
        with self._lock:
            self.db["training_examples"].insert(example.to_db_dict())

    def save_examples(self, examples: list[TrainingExample]) -> int:
        """Save multiple training examples."""
        if not examples:
            return 0
        with self._lock:
            self.db["training_examples"].insert_all(
                [e.to_db_dict() for e in examples]
            )
        return len(examples)

    def get_examples_by_split(self, split: str) -> list[TrainingExample]:
        """Get all examples for a split (train/val)."""
        rows = self.db["training_examples"].rows_where("split = ?", [split])
        return [TrainingExample.from_db_dict(row) for row in rows]

    def count_examples(self, task_type: str | None = None) -> int:
        """Count total examples, optionally filtered by task type."""
        if task_type:
            return self.db["training_examples"].count_where(
                "task_type = ?", [task_type]
            )
        return self.db["training_examples"].count

    # Run metadata methods
    def start_run(self, total_documents: int, config: dict) -> TrainingRunMetadata:
        """Start a new training run."""
        run = TrainingRunMetadata(
            total_documents=total_documents,
            config_snapshot=config,
        )
        with self._lock:
            self.db["training_run_metadata"].insert(run.to_db_dict())
        return run

    def update_run(self, run: TrainingRunMetadata) -> None:
        """Update run metadata."""
        with self._lock:
            self.db["training_run_metadata"].update(run.run_id, run.to_db_dict())

    def get_latest_run(self) -> TrainingRunMetadata | None:
        """Get the most recent training run."""
        rows = list(
            self.db["training_run_metadata"].rows_where(
                order_by="-started_at", limit=1
            )
        )
        if not rows:
            return None
        return TrainingRunMetadata.from_db_dict(rows[0])

    # Export methods
    def export_to_jsonl(self) -> tuple[Path, Path]:
        """Export examples to JSONL files.

        Returns:
            Tuple of (train_path, val_path).
        """
        train_path = self.output_dir / "train.jsonl"
        val_path = self.output_dir / "val.jsonl"

        train_examples = self.get_examples_by_split("train")
        val_examples = self.get_examples_by_split("val")

        with open(train_path, "w", encoding="utf-8") as f:
            for example in train_examples:
                f.write(json.dumps(example.to_jsonl_dict()) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for example in val_examples:
                f.write(json.dumps(example.to_jsonl_dict()) + "\n")

        logger.info(
            f"Exported {len(train_examples)} train, {len(val_examples)} val examples"
        )
        return train_path, val_path

    # Statistics
    def get_stats(self) -> dict:
        """Get training generation statistics."""
        progress_stats = {}
        for status in ["pending", "in_progress", "completed", "failed", "skipped"]:
            progress_stats[status] = self.db[
                "training_generation_progress"
            ].count_where("status = ?", [status])

        task_stats = {}
        for row in self.db.execute(
            "SELECT task_type, COUNT(*) FROM training_examples GROUP BY task_type"
        ).fetchall():
            task_stats[row[0]] = row[1]

        split_stats = {}
        for row in self.db.execute(
            "SELECT split, COUNT(*) FROM training_examples GROUP BY split"
        ).fetchall():
            split_stats[row[0]] = row[1]

        return {
            "progress": progress_stats,
            "by_task_type": task_stats,
            "by_split": split_stats,
            "total_examples": self.db["training_examples"].count,
        }
