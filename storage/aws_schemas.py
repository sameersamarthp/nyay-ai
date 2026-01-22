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
