#!/usr/bin/env python3
"""
Load AWS High Court metadata from parquet files into SQLite database.

This script reads parquet files from the AWS dataset and inserts
the metadata into the aws_documents table. Full text extraction
from PDFs is handled separately by process_aws_pdfs.py.

Usage:
    python scripts/load_aws_metadata.py --data-dir ./data/aws_data/data
    python scripts/load_aws_metadata.py --court 7_26 --bench dhcdb
    python scripts/load_aws_metadata.py --year 2025
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.aws_document_store import AWSDocumentStore
from storage.aws_schemas import AWSHighCourtDocument
from utils.logger import get_logger, get_script_logger

logger = get_logger(__name__)
script_output = get_script_logger(__name__)


def parse_date_of_registration(date_str: str | None) -> date | None:
    """Parse date_of_registration from DD-MM-YYYY format."""
    if not date_str or pd.isna(date_str):
        return None
    try:
        parts = date_str.split("-")
        if len(parts) == 3:
            return date(int(parts[2]), int(parts[1]), int(parts[0]))
    except (ValueError, IndexError):
        pass
    return None


def parse_decision_date(dt) -> date | None:
    """Parse decision_date from pandas Timestamp."""
    if dt is None or pd.isna(dt):
        return None
    try:
        if hasattr(dt, "date"):
            return dt.date()
        return None
    except Exception:
        return None


def parquet_row_to_document(row: pd.Series, year: int, bench: str) -> AWSHighCourtDocument | None:
    """Convert a parquet row to AWSHighCourtDocument."""
    try:
        # Skip rows without CNR
        cnr = row.get("cnr")
        if not cnr or pd.isna(cnr):
            return None

        return AWSHighCourtDocument(
            cnr=str(cnr),
            court_code=str(row.get("court_code", "")),
            title=str(row.get("title", "")),
            description=str(row.get("description")) if row.get("description") and not pd.isna(row.get("description")) else None,
            judge=str(row.get("judge")) if row.get("judge") and not pd.isna(row.get("judge")) else None,
            pdf_link=str(row.get("pdf_link")) if row.get("pdf_link") and not pd.isna(row.get("pdf_link")) else None,
            date_of_registration=parse_date_of_registration(row.get("date_of_registration")),
            decision_date=parse_decision_date(row.get("decision_date")),
            disposal_nature=str(row.get("disposal_nature")) if row.get("disposal_nature") and not pd.isna(row.get("disposal_nature")) else None,
            court=str(row.get("court", "")),
            year=year,
            bench=bench,
        )
    except Exception as e:
        logger.warning(f"Failed to convert row to document: {e}")
        return None


def find_parquet_files(
    base_path: Path,
    year: int | None = None,
    court: str | None = None,
    bench: str | None = None,
) -> list[tuple[Path, int, str]]:
    """Find parquet files matching the given filters.

    Returns:
        List of tuples: (parquet_path, year, bench)
    """
    results = []

    # Build search pattern
    if year:
        year_pattern = f"year={year}"
    else:
        year_pattern = "year=*"

    if court:
        court_pattern = f"court={court}"
    else:
        court_pattern = "court=*"

    if bench:
        bench_pattern = f"bench={bench}"
    else:
        bench_pattern = "bench=*"

    pattern = f"{year_pattern}/{court_pattern}/{bench_pattern}/metadata.parquet"

    for pq_path in base_path.glob(pattern):
        # Extract year and bench from path
        parts = pq_path.parts
        pq_year = None
        pq_bench = None

        for part in parts:
            if part.startswith("year="):
                pq_year = int(part.split("=")[1])
            elif part.startswith("bench="):
                pq_bench = part.split("=")[1]

        if pq_year and pq_bench:
            results.append((pq_path, pq_year, pq_bench))

    return sorted(results)


def load_parquet_to_db(
    parquet_path: Path,
    year: int,
    bench: str,
    store: AWSDocumentStore,
    batch_size: int = 1000,
) -> tuple[int, int]:
    """Load a single parquet file into the database.

    Returns:
        Tuple of (total_rows, inserted_rows)
    """
    logger.info(f"Loading {parquet_path}")

    # Read parquet file
    df = pd.read_parquet(parquet_path)
    total_rows = len(df)

    if total_rows == 0:
        return 0, 0

    # Get existing CNRs for this bench to skip duplicates
    existing_cnrs = store.get_existing_cnrs(bench=bench)
    logger.info(f"Found {len(existing_cnrs)} existing documents for bench {bench}")

    # Convert rows to documents in batches
    documents = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=total_rows, desc=f"Processing {bench}", leave=False):
        cnr = row.get("cnr")
        if cnr and cnr in existing_cnrs:
            skipped += 1
            continue

        doc = parquet_row_to_document(row, year, bench)
        if doc:
            documents.append(doc)

        # Insert in batches
        if len(documents) >= batch_size:
            store.bulk_insert_documents(documents)
            documents = []

    # Insert remaining documents
    if documents:
        store.bulk_insert_documents(documents)

    inserted = total_rows - skipped
    logger.info(f"Loaded {inserted} documents from {bench} (skipped {skipped} duplicates)")
    return total_rows, inserted


def main():
    parser = argparse.ArgumentParser(description="Load AWS High Court metadata into database")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/aws_data/data"),
        help="Base directory containing parquet files",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Filter by year (e.g., 2025)",
    )
    parser.add_argument(
        "--court",
        type=str,
        default=None,
        help="Filter by court code (e.g., 7_26)",
    )
    parser.add_argument(
        "--bench",
        type=str,
        default=None,
        help="Filter by bench (e.g., dhcdb)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for database inserts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be loaded without actually loading",
    )

    args = parser.parse_args()

    # Find parquet files
    parquet_files = find_parquet_files(
        args.data_dir, year=args.year, court=args.court, bench=args.bench
    )

    if not parquet_files:
        logger.error(f"No parquet files found in {args.data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(parquet_files)} parquet files to process")

    if args.dry_run:
        script_output.info("\nDry run - would load the following files:")
        for pq_path, year, bench in parquet_files:
            df = pd.read_parquet(pq_path)
            script_output.info(f"  {pq_path}: {len(df)} rows (year={year}, bench={bench})")
        return

    # Initialize store
    store = AWSDocumentStore()

    # Load each parquet file
    total_loaded = 0
    total_inserted = 0

    for pq_path, year, bench in tqdm(parquet_files, desc="Loading parquet files"):
        loaded, inserted = load_parquet_to_db(
            pq_path, year, bench, store, args.batch_size
        )
        total_loaded += loaded
        total_inserted += inserted

    # Print summary
    script_output.info("\n" + "=" * 50)
    script_output.info("LOADING COMPLETE")
    script_output.info("=" * 50)
    script_output.info(f"Total rows processed: {total_loaded:,}")
    script_output.info(f"Total documents inserted: {total_inserted:,}")

    # Show stats
    stats = store.get_stats()
    script_output.info(f"\nDatabase statistics:")
    script_output.info(f"  Total documents: {stats['total_documents']:,}")
    script_output.info(f"  Processed (with PDF text): {stats['processed_documents']:,}")
    script_output.info(f"  Unprocessed (no PDF text): {stats['unprocessed_documents']:,}")

    if stats["by_court"]:
        script_output.info(f"\nBy court:")
        for court, count in sorted(stats["by_court"].items(), key=lambda x: -x[1])[:10]:
            script_output.info(f"  {court}: {count:,}")


if __name__ == "__main__":
    main()
