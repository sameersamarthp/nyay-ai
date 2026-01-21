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
    TARGET_INDIA_CODE: int = 5_000

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

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
