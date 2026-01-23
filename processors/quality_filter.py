"""
Filter and sample documents for training data generation.

Applies quality filters and balanced sampling across courts.
"""

import random
from collections import Counter
from dataclasses import dataclass

from config.settings import settings
from storage.aws_document_store import AWSDocumentStore
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FilterStats:
    """Statistics from filtering operation."""

    total_documents: int
    after_word_filter: int
    after_quality_filter: int
    selected_for_training: int
    by_court: dict[str, int]


class QualityFilter:
    """Filter documents based on quality criteria."""

    def __init__(
        self,
        min_word_count: int = settings.TRAINING_MIN_WORD_COUNT,
        max_word_count: int = settings.TRAINING_MAX_WORD_COUNT,
    ):
        """Initialize quality filter.

        Args:
            min_word_count: Minimum words required.
            max_word_count: Maximum words allowed.
        """
        self.min_word_count = min_word_count
        self.max_word_count = max_word_count

    def passes_quality_check(self, doc: dict) -> bool:
        """Check if a document passes quality filters.

        Args:
            doc: Document dict with word_count, full_text, etc.

        Returns:
            True if document passes all quality checks.
        """
        # Must have full text
        if not doc.get("full_text"):
            return False

        # Word count filter
        word_count = doc.get("word_count", 0)
        if not (self.min_word_count <= word_count <= self.max_word_count):
            return False

        # Basic content quality checks
        text = doc["full_text"]

        # Check for excessive repetition (OCR artifacts)
        if self._has_excessive_repetition(text):
            return False

        # Check for minimum unique words (not just repeated content)
        unique_words = len(set(text.lower().split()))
        if unique_words < 100:  # Very low vocabulary suggests bad OCR
            return False

        return True

    def _has_excessive_repetition(self, text: str, threshold: float = 0.3) -> bool:
        """Check if text has too much repeated content."""
        words = text.lower().split()
        if len(words) < 50:
            return False

        # Check if any single word appears more than threshold of total
        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1]

        return (most_common_count / len(words)) > threshold


class DocumentSampler:
    """Sample documents with balanced distribution."""

    def __init__(
        self,
        store: AWSDocumentStore,
        quality_filter: QualityFilter | None = None,
        seed: int = 42,
    ):
        """Initialize document sampler.

        Args:
            store: Database store instance.
            quality_filter: Quality filter to apply.
            seed: Random seed for reproducibility.
        """
        self.store = store
        self.quality_filter = quality_filter or QualityFilter()
        self.rng = random.Random(seed)

    def get_eligible_documents(self) -> dict[str, list[dict]]:
        """Get all documents passing quality filters, grouped by court.

        Returns:
            Dict mapping court name to list of eligible document dicts.
        """
        eligible_by_court: dict[str, list[dict]] = {}

        # Query documents in word count range
        query = """
            SELECT cnr, court, word_count, full_text, title
            FROM aws_documents
            WHERE word_count BETWEEN ? AND ?
            AND full_text IS NOT NULL
        """

        cursor = self.store.db.execute(
            query,
            [self.quality_filter.min_word_count, self.quality_filter.max_word_count],
        )

        for row in cursor.fetchall():
            doc = dict(
                zip(["cnr", "court", "word_count", "full_text", "title"], row)
            )

            if self.quality_filter.passes_quality_check(doc):
                court = doc["court"]
                if court not in eligible_by_court:
                    eligible_by_court[court] = []
                eligible_by_court[court].append(doc)

        return eligible_by_court

    def sample_documents(
        self,
        total_needed: int = settings.TRAINING_DOCUMENTS_NEEDED,
        balance_by_court: bool = True,
    ) -> tuple[list[dict], FilterStats]:
        """Sample documents for training data generation.

        Args:
            total_needed: Total number of documents to sample.
            balance_by_court: If True, sample equally from each court.

        Returns:
            Tuple of (sampled documents list, filter statistics).
        """
        eligible_by_court = self.get_eligible_documents()

        total_eligible = sum(len(docs) for docs in eligible_by_court.values())
        courts = list(eligible_by_court.keys())

        logger.info(
            f"Found {total_eligible} eligible documents across {len(courts)} courts"
        )

        if balance_by_court and len(courts) > 1:
            # Sample equally from each court
            per_court = total_needed // len(courts)
            remainder = total_needed % len(courts)

            sampled = []
            stats_by_court = {}

            for i, court in enumerate(sorted(courts)):
                court_docs = eligible_by_court[court]
                n_to_sample = per_court + (1 if i < remainder else 0)
                n_to_sample = min(n_to_sample, len(court_docs))

                court_sample = self.rng.sample(court_docs, n_to_sample)
                sampled.extend(court_sample)
                stats_by_court[court] = len(court_sample)

                logger.info(f"Sampled {len(court_sample)} from {court}")
        else:
            # Sample from all documents
            all_docs = [doc for docs in eligible_by_court.values() for doc in docs]
            n_to_sample = min(total_needed, len(all_docs))
            sampled = self.rng.sample(all_docs, n_to_sample)
            stats_by_court = {
                court: len([d for d in sampled if d["court"] == court])
                for court in courts
            }

        # Shuffle final sample
        self.rng.shuffle(sampled)

        stats = FilterStats(
            total_documents=self.store.count_documents(processed_only=True),
            after_word_filter=total_eligible,
            after_quality_filter=total_eligible,  # Same for now
            selected_for_training=len(sampled),
            by_court=stats_by_court,
        )

        return sampled, stats


def sample_for_training(
    store: AWSDocumentStore,
    total_needed: int = settings.TRAINING_DOCUMENTS_NEEDED,
    seed: int = 42,
) -> tuple[list[dict], FilterStats]:
    """Convenience function to sample documents for training.

    Args:
        store: Database store instance.
        total_needed: Number of documents to sample.
        seed: Random seed.

    Returns:
        Tuple of (sampled documents, statistics).
    """
    sampler = DocumentSampler(store, seed=seed)
    return sampler.sample_documents(total_needed=total_needed)
