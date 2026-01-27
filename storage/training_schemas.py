"""
Pydantic schemas for training data generation.
"""

import json
from datetime import datetime
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, Field


class TrainingExample(BaseModel):
    """A single training example."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    cnr: str
    task_type: str  # Store as string for JSON serialization
    instruction: str
    input: str
    output: str
    created_at: datetime = Field(default_factory=datetime.now)
    split: str = "train"  # train or val
    input_tokens: int = 0
    output_tokens: int = 0

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Convert to JSONL format for training."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "metadata": {
                "cnr": self.cnr,
                "task_type": self.task_type,
            },
        }

    def to_db_dict(self) -> dict[str, Any]:
        """Convert to dict for database storage."""
        data = self.model_dump()
        data["created_at"] = data["created_at"].isoformat()
        return data

    @classmethod
    def from_db_dict(cls, data: dict[str, Any]) -> Self:
        """Create from database dict."""
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class TrainingProgress(BaseModel):
    """Progress tracking for a single document."""

    cnr: str
    status: str = "pending"  # pending, in_progress, completed, failed, skipped
    task_types_generated: list[str] = Field(default_factory=list)
    examples_generated: int = 0
    error_message: str | None = None
    retry_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    input_tokens: int = 0
    output_tokens: int = 0

    def to_db_dict(self) -> dict[str, Any]:
        """Convert to dict for database storage."""
        data = self.model_dump()
        data["created_at"] = data["created_at"].isoformat()
        data["updated_at"] = data["updated_at"].isoformat()
        data["task_types_generated"] = json.dumps(data["task_types_generated"])
        return data

    @classmethod
    def from_db_dict(cls, data: dict[str, Any]) -> Self:
        """Create from database dict."""
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if isinstance(data.get("task_types_generated"), str):
            data["task_types_generated"] = json.loads(data["task_types_generated"])
        return cls(**data)


class TrainingRunMetadata(BaseModel):
    """Metadata for a training generation run."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    status: str = "running"  # running, completed, failed, interrupted
    total_documents: int = 0
    documents_processed: int = 0
    examples_generated: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost: float = 0.0
    config_snapshot: dict[str, Any] = Field(default_factory=dict)

    def to_db_dict(self) -> dict[str, Any]:
        """Convert to dict for database storage."""
        data = self.model_dump()
        data["started_at"] = data["started_at"].isoformat()
        if data["completed_at"]:
            data["completed_at"] = data["completed_at"].isoformat()
        data["config_snapshot"] = json.dumps(data["config_snapshot"])
        return data

    @classmethod
    def from_db_dict(cls, data: dict[str, Any]) -> Self:
        """Create from database dict."""
        if isinstance(data.get("started_at"), str):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at") and isinstance(data["completed_at"], str):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        if isinstance(data.get("config_snapshot"), str):
            data["config_snapshot"] = json.loads(data["config_snapshot"])
        return cls(**data)
