"""
Pydantic schemas for AWS High Court dataset.
"""

from datetime import date, datetime
from hashlib import sha256
from typing import Self

from pydantic import BaseModel, Field, computed_field, model_validator


class AWSHighCourtDocument(BaseModel):
    """Schema for AWS High Court judgment document."""

    # Primary Identifier
    cnr: str  # Primary key (e.g., HCBM030079862025)
    doc_id: str = ""  # Generated hash for compatibility

    # From Parquet (Direct mapping)
    court_code: str  # e.g., "27~1"
    title: str  # Full title with case number + parties
    description: str | None = None  # Truncated summary (~300 chars)
    judge: str | None = None  # Raw judge string
    pdf_link: str | None = None  # Relative path to PDF
    date_of_registration: date | None = None
    decision_date: date | None = None
    disposal_nature: str | None = None  # e.g., "DISPOSED OFF"
    court: str  # e.g., "Bombay High Court"

    # PDF Extracted Content
    full_text: str | None = None  # Extracted from PDF (nullable until processed)
    pdf_processed: bool = False  # Track processing status

    # Metadata
    year: int  # Partition key (e.g., 2025)
    bench: str  # For tracking source (e.g., "hcaurdb")
    created_at: datetime = Field(default_factory=datetime.now)

    @computed_field
    @property
    def word_count(self) -> int | None:
        """Calculate word count of full text."""
        if self.full_text:
            return len(self.full_text.split())
        return None

    @model_validator(mode="after")
    def generate_doc_id(self) -> Self:
        """Generate doc_id from CNR if not provided."""
        if not self.doc_id:
            self.doc_id = sha256(self.cnr.encode()).hexdigest()[:16]
        return self

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        data = self.model_dump()
        # Convert date/datetime to string for storage
        if data["date_of_registration"]:
            data["date_of_registration"] = data["date_of_registration"].isoformat()
        if data["decision_date"]:
            data["decision_date"] = data["decision_date"].isoformat()
        data["created_at"] = data["created_at"].isoformat()
        data["pdf_processed"] = int(data["pdf_processed"])
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create instance from dictionary."""
        # Convert string dates back to date objects
        if data.get("date_of_registration") and isinstance(data["date_of_registration"], str):
            data["date_of_registration"] = date.fromisoformat(data["date_of_registration"])
        if data.get("decision_date") and isinstance(data["decision_date"], str):
            data["decision_date"] = date.fromisoformat(data["decision_date"])
        if data.get("created_at") and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "pdf_processed" in data:
            data["pdf_processed"] = bool(data["pdf_processed"])
        return cls(**data)

    @classmethod
    def from_parquet_row(cls, row: dict, year: int, bench: str) -> Self:
        """Create instance from a parquet row."""
        # Parse date_of_registration from string format "DD-MM-YYYY"
        date_of_reg = None
        if row.get("date_of_registration"):
            try:
                parts = row["date_of_registration"].split("-")
                if len(parts) == 3:
                    date_of_reg = date(int(parts[2]), int(parts[1]), int(parts[0]))
            except (ValueError, IndexError):
                pass

        # decision_date is already datetime64, convert to date
        decision_dt = None
        if row.get("decision_date") and not pd.isna(row["decision_date"]):
            decision_dt = row["decision_date"].date() if hasattr(row["decision_date"], "date") else None

        return cls(
            cnr=row["cnr"],
            court_code=row.get("court_code", ""),
            title=row["title"],
            description=row.get("description"),
            judge=row.get("judge"),
            pdf_link=row.get("pdf_link"),
            date_of_registration=date_of_reg,
            decision_date=decision_dt,
            disposal_nature=row.get("disposal_nature"),
            court=row["court"],
            year=year,
            bench=bench,
        )


# Import pandas here to avoid circular imports and make it optional
try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore


class AWSProcessingProgress(BaseModel):
    """Track PDF processing progress for resume functionality."""

    court_code: str
    bench: str
    year: int
    total_documents: int
    documents_processed: int = 0
    last_cnr: str | None = None
    last_updated: datetime = Field(default_factory=datetime.now)
    is_complete: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        data = self.model_dump()
        data["last_updated"] = data["last_updated"].isoformat()
        data["is_complete"] = int(data["is_complete"])
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create instance from dictionary."""
        if data.get("last_updated") and isinstance(data["last_updated"], str):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        if "is_complete" in data:
            data["is_complete"] = bool(data["is_complete"])
        return cls(**data)
