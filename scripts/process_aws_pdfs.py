#!/usr/bin/env python3
"""
Process PDFs from AWS High Court dataset and extract text into database.

This script reads tar files containing PDFs, extracts text using PyPDF2,
and updates the aws_documents table with the full_text content.

Usage:
    python scripts/process_aws_pdfs.py --tar-dir ./data/aws_data/tar
    python scripts/process_aws_pdfs.py --court 7_26 --bench dhcdb
    python scripts/process_aws_pdfs.py --limit 1000  # Process first 1000 only
    python scripts/process_aws_pdfs.py --resume  # Resume from last position
"""

import argparse
import signal
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.pdf_extractor import TarPDFExtractor, find_tar_files, get_tar_pdf_count
from storage.aws_document_store import AWSDocumentStore
from storage.aws_schemas import AWSProcessingProgress
from utils.logger import get_logger

logger = get_logger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    if shutdown_requested:
        print("\nForce quitting...")
        sys.exit(1)
    print("\nShutdown requested. Finishing current batch...")
    shutdown_requested = True


def extract_court_bench_from_path(tar_path: Path) -> tuple[str, str, int]:
    """Extract court code, bench, and year from tar file path.

    Expected path format: .../year=2025/court=7_26/bench=dhcdb/data.tar

    Note: Tar paths use underscore (7_26) but database uses tilde (7~26).
    This function returns the database format (with tilde).
    """
    court_code = ""
    bench = ""
    year = 0

    for part in tar_path.parts:
        if part.startswith("year="):
            year = int(part.split("=")[1])
        elif part.startswith("court="):
            # Convert underscore to tilde to match database format
            court_code = part.split("=")[1].replace("_", "~")
        elif part.startswith("bench="):
            bench = part.split("=")[1]

    return court_code, bench, year


def process_tar_file(
    tar_path: Path,
    store: AWSDocumentStore,
    limit: int | None = None,
    resume_from_cnr: str | None = None,
    batch_size: int = 100,
) -> tuple[int, int, str | None]:
    """Process a single tar file and update database with extracted text.

    Args:
        tar_path: Path to tar file.
        store: Database store instance.
        limit: Maximum number of PDFs to process.
        resume_from_cnr: CNR to resume from (skip PDFs until this one).
        batch_size: Number of updates to batch together.

    Returns:
        Tuple of (processed_count, success_count, last_cnr)
    """
    global shutdown_requested

    court_code, bench, year = extract_court_bench_from_path(tar_path)
    logger.info(f"Processing tar file: {tar_path.name} (court={court_code}, bench={bench})")

    # Get existing CNRs in database for this bench
    existing_cnrs = store.get_existing_cnrs(court_code=court_code, bench=bench)
    logger.info(f"Found {len(existing_cnrs)} documents in database for this bench")

    processed = 0
    success = 0
    last_cnr = None
    updates_batch = []
    skipping = resume_from_cnr is not None

    with TarPDFExtractor(tar_path) as extractor:
        pdf_list = extractor.list_pdfs()
        total_pdfs = len(pdf_list)
        logger.info(f"Found {total_pdfs} PDFs in tar file")

        # Determine actual number to process
        if limit:
            total_to_process = min(limit, total_pdfs)
        else:
            total_to_process = total_pdfs

        with tqdm(total=total_to_process, desc=f"Processing {bench}") as pbar:
            for cnr, text in extractor.iter_pdfs():
                if shutdown_requested:
                    logger.info("Shutdown requested, saving progress...")
                    break

                # Handle resume logic
                if skipping:
                    if cnr == resume_from_cnr:
                        skipping = False
                        logger.info(f"Resuming from CNR: {cnr}")
                    continue

                # Check if this CNR exists in database
                if cnr not in existing_cnrs:
                    logger.debug(f"CNR {cnr} not in database, skipping")
                    continue

                processed += 1
                last_cnr = cnr

                if text:
                    updates_batch.append((cnr, text))
                    success += 1

                # Batch update
                if len(updates_batch) >= batch_size:
                    store.bulk_update_full_text(updates_batch)
                    updates_batch = []

                pbar.update(1)

                # Check limit
                if limit and processed >= limit:
                    logger.info(f"Reached limit of {limit} documents")
                    break

        # Flush remaining updates
        if updates_batch:
            store.bulk_update_full_text(updates_batch)

    return processed, success, last_cnr


def main():
    parser = argparse.ArgumentParser(description="Process AWS High Court PDFs and extract text")
    parser.add_argument(
        "--tar-dir",
        type=Path,
        default=Path("data/aws_data/tar"),
        help="Base directory containing tar files",
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
        "--limit",
        type=int,
        default=None,
        help="Maximum number of PDFs to process per tar file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for database updates",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last saved progress",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be processed",
    )

    args = parser.parse_args()

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Find tar files
    tar_files = find_tar_files(args.tar_dir, court_code=args.court, bench=args.bench)

    if not tar_files:
        logger.error(f"No tar files found in {args.tar_dir}")
        print("\nMake sure to download the tar files first:")
        print("  aws s3 sync s3://indian-high-court-judgments/data/tar/year=2025/court=7_26/bench=dhcdb ./data/aws_data/tar/year=2025/court=7_26/bench=dhcdb --no-sign-request")
        sys.exit(1)

    logger.info(f"Found {len(tar_files)} tar files to process")

    if args.dry_run:
        print("\nDry run - would process the following tar files:")
        for tar_path in tar_files:
            pdf_count = get_tar_pdf_count(tar_path)
            court_code, bench, year = extract_court_bench_from_path(tar_path)
            print(f"  {tar_path.name}: {pdf_count} PDFs (court={court_code}, bench={bench}, year={year})")
        return

    # Initialize store
    store = AWSDocumentStore()

    # Process each tar file
    total_processed = 0
    total_success = 0

    for tar_path in tar_files:
        if shutdown_requested:
            break

        court_code, bench, year = extract_court_bench_from_path(tar_path)

        # Check for resume
        resume_from_cnr = None
        if args.resume:
            progress = store.get_progress(court_code, bench, year)
            if progress and not progress.is_complete:
                resume_from_cnr = progress.last_cnr
                logger.info(f"Resuming from CNR: {resume_from_cnr}")
            elif progress and progress.is_complete:
                logger.info(f"Skipping {bench} - already complete")
                continue

        # Process tar file
        processed, success, last_cnr = process_tar_file(
            tar_path,
            store,
            limit=args.limit,
            resume_from_cnr=resume_from_cnr,
            batch_size=args.batch_size,
        )

        total_processed += processed
        total_success += success

        # Save progress
        if last_cnr:
            progress = AWSProcessingProgress(
                court_code=court_code,
                bench=bench,
                year=year,
                total_documents=processed,
                documents_processed=success,
                last_cnr=last_cnr,
                is_complete=not shutdown_requested and (args.limit is None or processed < args.limit),
            )
            store.save_progress(progress)

    # Print summary
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE" if not shutdown_requested else "PROCESSING INTERRUPTED")
    print("=" * 50)
    print(f"Total PDFs processed: {total_processed:,}")
    print(f"Successful extractions: {total_success:,}")
    print(f"Failed extractions: {total_processed - total_success:,}")

    # Show stats
    stats = store.get_stats()
    print(f"\nDatabase statistics:")
    print(f"  Total documents: {stats['total_documents']:,}")
    print(f"  With full text: {stats['processed_documents']:,}")
    print(f"  Without full text: {stats['unprocessed_documents']:,}")

    if shutdown_requested:
        print("\nProgress saved. Run with --resume to continue.")


if __name__ == "__main__":
    main()
