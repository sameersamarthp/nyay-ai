#!/usr/bin/env python3
"""
Main entry point for data collection.

Usage:
    # Dry run (10 documents from each source)
    python scripts/run_collection.py --dry-run

    # Run single source
    python scripts/run_collection.py --source indian_kanoon --target 5000

    # Run all sources
    python scripts/run_collection.py --source all

    # Resume interrupted collection
    python scripts/run_collection.py --resume
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from storage.document_store import DocumentStore
from storage.schemas import DocumentSource
from scrapers import (
    IndianKanoonScraper,
    SupremeCourtScraper,
    HighCourtsScraper,
    IndiaCodeScraper,
)
from utils.logger import setup_root_logger, get_logger

logger = get_logger(__name__)


def get_scraper(source: str, store: DocumentStore, target: int | None = None):
    """Get scraper instance for a source.

    Args:
        source: Source name.
        store: Document store instance.
        target: Optional target count override.

    Returns:
        Scraper instance.
    """
    scrapers = {
        "indian_kanoon": IndianKanoonScraper,
        "supreme_court": SupremeCourtScraper,
        "high_courts": HighCourtsScraper,
        "india_code": IndiaCodeScraper,
    }

    if source not in scrapers:
        raise ValueError(f"Unknown source: {source}. Available: {list(scrapers.keys())}")

    return scrapers[source](store=store, target_count=target)


def run_dry_run(sources: list[str], count: int = 10) -> None:
    """Run a dry run to test scrapers.

    Args:
        sources: List of sources to test.
        count: Number of documents to fetch per source.
    """
    logger.info(f"Starting dry run: {count} documents per source")
    store = DocumentStore()

    for source in sources:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {source}")
        logger.info(f"{'='*50}")

        try:
            scraper = get_scraper(source, store)
            docs = scraper.dry_run(count=count)

            logger.info(f"\nResults for {source}:")
            logger.info(f"  Documents fetched: {len(docs)}")

            if docs:
                sample = docs[0]
                logger.info(f"  Sample document:")
                logger.info(f"    Title: {sample.case_title[:60]}...")
                logger.info(f"    Court: {sample.court}")
                logger.info(f"    Date: {sample.date_decided}")
                logger.info(f"    Word count: {sample.word_count}")

        except Exception as e:
            logger.error(f"Error testing {source}: {e}")


def run_collection(
    sources: list[str],
    targets: dict[str, int] | None = None,
    resume: bool = True,
) -> dict[str, int]:
    """Run data collection.

    Args:
        sources: List of sources to collect from.
        targets: Optional dict of source -> target count.
        resume: Whether to resume from previous progress.

    Returns:
        Dict of source -> documents collected.
    """
    logger.info("Starting data collection")
    settings.ensure_directories()

    store = DocumentStore()
    results = {}

    for source in sources:
        logger.info(f"\n{'='*50}")
        logger.info(f"Collecting from {source}")
        logger.info(f"{'='*50}")

        try:
            target = targets.get(source) if targets else None
            scraper = get_scraper(source, store, target)
            count = scraper.scrape(resume=resume)
            results[source] = count

        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            break

        except Exception as e:
            logger.error(f"Error collecting from {source}: {e}")
            results[source] = 0

    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("Collection Summary")
    logger.info(f"{'='*50}")

    total = 0
    for source, count in results.items():
        logger.info(f"  {source}: {count} documents")
        total += count

    logger.info(f"  Total: {total} documents")

    # Print stats
    stats = store.get_stats()
    logger.info(f"\nDatabase Stats:")
    logger.info(f"  Total in database: {stats['total_documents']}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect legal documents from various sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run                      # Test with 10 docs each
  %(prog)s --source indian_kanoon         # Collect from Indian Kanoon
  %(prog)s --source all --target 10000    # Collect 10k docs total
  %(prog)s --resume                       # Resume previous collection
        """,
    )

    parser.add_argument(
        "--source",
        choices=["indian_kanoon", "supreme_court", "high_courts", "india_code", "all"],
        default="all",
        help="Source to collect from (default: all)",
    )

    parser.add_argument(
        "--target",
        type=int,
        help="Target number of documents (overrides default)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test run with 10 documents per source",
    )

    parser.add_argument(
        "--dry-run-count",
        type=int,
        default=10,
        help="Number of documents for dry run (default: 10)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from previous progress (default: True)",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore previous progress",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection statistics and exit",
    )

    args = parser.parse_args()

    # Setup logging
    setup_root_logger()

    # Determine sources
    if args.source == "all":
        sources = ["indian_kanoon", "supreme_court", "high_courts", "india_code"]
    else:
        sources = [args.source]

    # Stats only
    if args.stats:
        store = DocumentStore()
        stats = store.get_stats()
        print("\nCollection Statistics:")
        print(f"  Total documents: {stats['total_documents']}")
        print("\n  By source:")
        for source, count in stats["by_source"].items():
            print(f"    {source}: {count}")
        print("\n  By court:")
        for court, count in sorted(stats["by_court"].items(), key=lambda x: -x[1])[:10]:
            print(f"    {court}: {count}")
        return

    # Dry run
    if args.dry_run:
        run_dry_run(sources, args.dry_run_count)
        return

    # Full collection
    targets = None
    if args.target:
        # Distribute target across sources
        per_source = args.target // len(sources)
        targets = {s: per_source for s in sources}

    resume = not args.no_resume
    run_collection(sources, targets, resume)


if __name__ == "__main__":
    main()
