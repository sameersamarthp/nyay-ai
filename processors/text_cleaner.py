"""
Light text preprocessing for legal documents.

Handles common OCR artifacts and formatting issues from PDF extraction.
"""

import re

from utils.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """Clean and preprocess legal document text."""

    # Common OCR errors in Indian legal docs
    OCR_REPLACEMENTS = {
        r"\bvs\.?\b": "versus",  # vs/vs. -> versus
        r"\bu/s\.?\b": "under section",
        r"\bs\.?\s*(\d+)": r"Section \1",  # s.302 -> Section 302
        r"(?<!\w)&(?!\w)": "and",  # & -> and (when standalone)
    }

    # Patterns to remove
    REMOVE_PATTERNS = [
        r"Page\s+\d+\s+of\s+\d+",  # Page numbers
        r"^\s*\d+\s*$",  # Standalone numbers (line numbers)
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",  # Control characters
    ]

    def __init__(self, max_chars: int | None = None):
        """Initialize text cleaner.

        Args:
            max_chars: Maximum characters to return (truncates end).
        """
        self.max_chars = max_chars

    def clean(self, text: str) -> str | None:
        """Clean document text.

        Args:
            text: Raw document text.

        Returns:
            Cleaned text or None if text is empty/invalid.
        """
        if not text or not text.strip():
            return None

        cleaned = text

        # Remove unwanted patterns
        for pattern in self.REMOVE_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE)

        # Apply OCR replacements (case-insensitive)
        for pattern, replacement in self.OCR_REPLACEMENTS.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # Normalize whitespace
        cleaned = self._normalize_whitespace(cleaned)

        # Truncate if needed
        if self.max_chars and len(cleaned) > self.max_chars:
            cleaned = self._smart_truncate(cleaned, self.max_chars)
            logger.debug(f"Truncated text from {len(text)} to {len(cleaned)} chars")

        return cleaned if cleaned.strip() else None

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph structure."""
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Replace multiple spaces with single space
        text = re.sub(r"[ \t]+", " ", text)
        # Strip leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    def _smart_truncate(self, text: str, max_chars: int) -> str:
        """Truncate text at a sentence boundary if possible."""
        if len(text) <= max_chars:
            return text

        # Find last sentence ending before max_chars
        truncated = text[:max_chars]

        # Look for sentence endings
        last_period = truncated.rfind(".")
        last_newline = truncated.rfind("\n\n")

        # Use the later of period or paragraph break
        cut_point = max(last_period, last_newline)

        if cut_point > max_chars * 0.7:  # Only if we keep 70%+
            return text[: cut_point + 1].strip()

        # Fall back to hard cut
        return truncated.strip() + "..."


def clean_text(text: str, max_chars: int | None = None) -> str | None:
    """Convenience function for text cleaning.

    Args:
        text: Raw document text.
        max_chars: Maximum characters to return.

    Returns:
        Cleaned text or None.
    """
    cleaner = TextCleaner(max_chars=max_chars)
    return cleaner.clean(text)
