"""
Configuration settings for Nyay AI India.

All configuration is centralized here. Use environment variables for secrets.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    DB_PATH: Path = DATA_DIR / "metadata.db"

    # Scraping targets
    TARGET_TOTAL: int = 10_000
    TARGET_INDIAN_KANOON: int = 5_000
    TARGET_SUPREME_COURT: int = 2_000
    TARGET_HIGH_COURTS: int = 2_000
    TARGET_INDIA_CODE: int = 1_000

    # Date range for recent cases
    DATE_START: str = "2019-01-01"
    DATE_END: str = "2024-12-31"

    # Rate limiting (be respectful!)
    MIN_REQUEST_INTERVAL: float = 2.0  # Minimum seconds between requests
    MAX_REQUEST_INTERVAL: float = 4.0  # Maximum (with jitter)
    RATE_LIMIT_BACKOFF: int = 60  # Seconds to wait on 429 response
    MAX_RETRIES: int = 3  # Max retry attempts per request
    REQUEST_TIMEOUT: int = 30  # Request timeout in seconds

    # Indian Kanoon settings
    INDIAN_KANOON_BASE_URL: str = "https://indiankanoon.org"
    INDIAN_KANOON_API_URL: str = "https://api.indiankanoon.org"
    INDIAN_KANOON_API_TOKEN: str = ""  # Set via environment variable

    # Supreme Court settings (via Indian Kanoon)
    SUPREME_COURT_SEARCH_FILTER: str = "doctypes: supremecourt"

    # High Courts settings (via Indian Kanoon)
    HIGH_COURTS: list[str] = [
        "Delhi High Court",
        "Bombay High Court",
        "Karnataka High Court",
    ]
    HIGH_COURT_SEARCH_FILTER: str = "doctypes: highcourts"

    # India Code settings
    INDIA_CODE_BASE_URL: str = "https://www.indiacode.nic.in"

    # Subject categories to collect
    SUBJECT_CATEGORIES: list[str] = [
        "Criminal",
        "Civil",
        "Constitutional",
        "Tax",
        "Labour",
        "Family",
        "Property",
        "Corporate",
        "Environmental",
        "Intellectual Property",
    ]

    # Checkpointing
    CHECKPOINT_INTERVAL: int = 100  # Save progress every N documents
    ENABLE_RESUME: bool = True  # Allow resuming interrupted collection

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # User agent rotation
    ROTATE_USER_AGENT: bool = True

    # LLM Settings (Phase 2: Training Data Generation)
    ANTHROPIC_API_KEY: str = ""  # Set via environment variable
    LLM_MODEL: str = "claude-3-haiku-20240307"  # Cost-effective for generation
    LLM_MAX_TOKENS: int = 2048  # Max output tokens per generation
    LLM_TEMPERATURE: float = 0.7  # Balance variety and consistency
    LLM_REQUESTS_PER_MINUTE: int = 50  # Conservative rate limit
    LLM_MIN_REQUEST_INTERVAL: float = 1.2  # Ensures ~50 RPM
    LLM_MAX_RETRIES: int = 3  # Retry failed API calls
    LLM_RETRY_DELAY: float = 2.0  # Base delay for exponential backoff

    # Training Data Settings
    TRAINING_DATA_DIR: Path = DATA_DIR / "training"
    TRAINING_TARGET_TOTAL: int = 8000  # Total training examples
    TRAINING_TARGET_PER_TYPE: int = 2000  # 4 types x 2000 each
    TRAINING_DOCUMENTS_NEEDED: int = 4000  # 2 examples per document
    TRAINING_VAL_SPLIT: float = 0.1  # 10% validation set
    TRAINING_MIN_WORD_COUNT: int = 500  # Minimum words for document quality
    TRAINING_MAX_WORD_COUNT: int = 15000  # Maximum words (truncate longer)
    TRAINING_MAX_INPUT_CHARS: int = 12000  # Truncate input to fit context
    TRAINING_CHECKPOINT_INTERVAL: int = 50  # Save progress every N documents

    # Cost Tracking (Haiku pricing per 1K tokens)
    HAIKU_INPUT_COST_PER_1K: float = 0.00025  # $0.25 per 1M input tokens
    HAIKU_OUTPUT_COST_PER_1K: float = 0.00125  # $1.25 per 1M output tokens
    TRAINING_COST_LIMIT: float = 15.0  # Stop if cost exceeds this amount

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
