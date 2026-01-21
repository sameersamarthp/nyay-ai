"""
Pydantic schemas for legal documents and scraping metadata.
"""

from datetime import date, datetime
from enum import Enum
from hashlib import sha256
from typing import Self

from pydantic import BaseModel, Field, computed_field, model_validator


class DocumentSource(str, Enum):
    """Source of legal documents."""

    INDIAN_KANOON = "indian_kanoon"
    SUPREME_COURT = "supreme_court"
    HIGH_COURTS = "high_courts"
    INDIA_CODE = "india_code"


class LegalDocument(BaseModel):
    """Schema for a legal document (judgment or statute)."""

    # Identifiers
    doc_id: str = ""  # Generated from citation+court+date if not provided
    source: DocumentSource
    url: str
    cnr: str | None = None  # eCourts unique Case Number Record (e.g., DLHC010012342023)

    # Case Information
    citation: str | None = None  # e.g., "2023 SCC 456", "AIR 2022 SC 1234"
    case_number: str | None = None  # e.g., "Criminal Appeal No. 123/2023"
    case_title: str  # e.g., "State of Maharashtra v. ABC"
    court: str  # e.g., "Supreme Court of India"

    # Parties
    petitioner: str | None = None
    respondent: str | None = None

    # Bench
    judges: list[str] = Field(default_factory=list)
    author_judge: str | None = None  # Judge who authored the judgment (distinct from full bench)

    # Dates
    date_decided: date | None = None

    # Classification
    subject_category: str | None = None  # Criminal, Civil, Constitutional, etc.
    acts_referred: list[str] = Field(default_factory=list)
    sections_referred: list[str] = Field(default_factory=list)
    cases_cited: list[str] = Field(default_factory=list)

    # Outcome
    outcome: str | None = None  # Allowed, Dismissed, Remanded, etc.

    # Content
    headnotes: str | None = None  # Summary if available
    full_text: str  # Complete judgment text

    # Metadata
    scraped_at: datetime = Field(default_factory=datetime.now)
    is_landmark: bool = False
    available_languages: list[str] = Field(default_factory=list)  # e.g., ["en", "hi", "kn"]

    @computed_field
    @property
    def word_count(self) -> int:
        """Calculate word count of full text."""
        return len(self.full_text.split())

    @model_validator(mode="after")
    def generate_doc_id(self) -> Self:
        """Generate doc_id if not provided."""
        if not self.doc_id:
            # Prefer CNR as it's the official eCourts unique identifier
            if self.cnr:
                id_string = self.cnr
            else:
                # Fallback to hash of key fields
                id_string = f"{self.citation or ''}{self.court}{self.date_decided or ''}{self.case_title}"
            self.doc_id = sha256(id_string.encode()).hexdigest()[:16]
        return self

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        data = self.model_dump()
        # Convert date/datetime to string for JSON storage
        if data["date_decided"]:
            data["date_decided"] = data["date_decided"].isoformat()
        data["scraped_at"] = data["scraped_at"].isoformat()
        data["source"] = data["source"].value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create instance from dictionary."""
        # Convert string dates back to date objects
        if data.get("date_decided") and isinstance(data["date_decided"], str):
            data["date_decided"] = date.fromisoformat(data["date_decided"])
        if data.get("scraped_at") and isinstance(data["scraped_at"], str):
            data["scraped_at"] = datetime.fromisoformat(data["scraped_at"])
        return cls(**data)


class ScrapingProgress(BaseModel):
    """Track scraping progress for resume functionality."""

    source: DocumentSource
    total_target: int
    documents_collected: int = 0
    last_page: int = 0
    last_url: str | None = None
    last_updated: datetime = Field(default_factory=datetime.now)
    is_complete: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        data = self.model_dump()
        data["last_updated"] = data["last_updated"].isoformat()
        data["source"] = data["source"].value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create instance from dictionary."""
        if data.get("last_updated") and isinstance(data["last_updated"], str):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)


class StatuteDocument(BaseModel):
    """Schema specifically for statutes/acts from India Code."""

    doc_id: str = ""
    source: DocumentSource = DocumentSource.INDIA_CODE
    url: str

    # Act Information
    act_title: str  # e.g., "Indian Penal Code, 1860"
    act_number: str | None = None  # e.g., "Act No. 45 of 1860"
    year_enacted: int | None = None

    # Content
    preamble: str | None = None
    sections: list[dict] = Field(default_factory=list)  # List of {number, title, content}
    schedules: list[dict] = Field(default_factory=list)
    full_text: str

    # Metadata
    ministry: str | None = None
    last_amended: date | None = None
    scraped_at: datetime = Field(default_factory=datetime.now)

    @computed_field
    @property
    def word_count(self) -> int:
        """Calculate word count of full text."""
        return len(self.full_text.split())

    @model_validator(mode="after")
    def generate_doc_id(self) -> Self:
        """Generate doc_id if not provided."""
        if not self.doc_id:
            id_string = f"{self.act_title}{self.year_enacted or ''}"
            self.doc_id = sha256(id_string.encode()).hexdigest()[:16]
        return self

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        data = self.model_dump()
        if data["last_amended"]:
            data["last_amended"] = data["last_amended"].isoformat()
        data["scraped_at"] = data["scraped_at"].isoformat()
        data["source"] = data["source"].value
        return data
