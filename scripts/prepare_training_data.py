#!/usr/bin/env python3
"""
Generate training data from legal documents using Claude API.

This script:
1. Samples documents from the database
2. Generates training examples using Claude Haiku
3. Tracks progress for resume capability
4. Exports to JSONL format

Usage:
    python scripts/prepare_training_data.py
    python scripts/prepare_training_data.py --resume
    python scripts/prepare_training_data.py --dry-run
    python scripts/prepare_training_data.py --limit 100
"""

import argparse
import random
import signal
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from processors.quality_filter import sample_for_training
from processors.llm_generator import LLMGenerator, TaskAssigner
from storage.aws_document_store import AWSDocumentStore
from storage.training_store import TrainingStore
from storage.training_schemas import TrainingExample, TrainingProgress
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
    print("\nShutdown requested. Finishing current document...")
    shutdown_requested = True


def assign_splits(
    n_examples: int, val_ratio: float = 0.1, seed: int = 42
) -> list[str]:
    """Assign train/val splits to examples.

    Args:
        n_examples: Number of examples.
        val_ratio: Ratio for validation set.
        seed: Random seed.

    Returns:
        List of 'train' or 'val' assignments.
    """
    rng = random.Random(seed)
    n_val = int(n_examples * val_ratio)
    splits = ["train"] * (n_examples - n_val) + ["val"] * n_val
    rng.shuffle(splits)
    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from legal documents"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last saved progress",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making API calls",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=settings.TRAINING_COST_LIMIT,
        help="Stop if estimated cost exceeds this amount",
    )

    args = parser.parse_args()

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize stores
    doc_store = AWSDocumentStore()
    training_store = TrainingStore()

    # Check for resume
    if args.resume:
        pending_cnrs = training_store.get_pending_cnrs()
        failed_cnrs = training_store.get_failed_cnrs()

        if pending_cnrs or failed_cnrs:
            print(
                f"Resuming: {len(pending_cnrs)} pending, {len(failed_cnrs)} to retry"
            )
            cnrs_to_process = pending_cnrs + failed_cnrs
            # Get document data for these CNRs
            documents = []
            for cnr in cnrs_to_process:
                doc = doc_store.get_document(cnr)
                if doc:
                    documents.append(
                        {
                            "cnr": doc.cnr,
                            "full_text": doc.full_text,
                            "court": doc.court,
                            "title": doc.title,
                        }
                    )
        else:
            print("No pending documents to resume. Starting fresh.")
            args.resume = False

    if not args.resume:
        # Sample documents
        print("Sampling documents...")
        documents, stats = sample_for_training(
            doc_store,
            total_needed=args.limit or settings.TRAINING_DOCUMENTS_NEEDED,
            seed=args.seed,
        )

        print(f"\nSampling Statistics:")
        print(f"  Total in database: {stats.total_documents:,}")
        print(f"  After word filter: {stats.after_word_filter:,}")
        print(f"  Selected: {stats.selected_for_training:,}")
        print(f"  By court:")
        for court, count in stats.by_court.items():
            print(f"    {court}: {count:,}")

        # Initialize progress tracking
        cnrs = [doc["cnr"] for doc in documents]
        new_count = training_store.init_progress_for_documents(cnrs)
        print(f"Initialized {new_count} new progress records")

    if args.dry_run:
        print("\nDry run - would process:")
        print(f"  Documents: {len(documents)}")
        print(f"  Examples to generate: {len(documents) * 2}")
        print(
            f"  Estimated cost: ${len(documents) * 2 * 0.002:.2f}"
        )  # Rough estimate
        return

    # Start run tracking
    config = {
        "model": settings.LLM_MODEL,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "documents": len(documents),
        "seed": args.seed,
    }
    run = training_store.start_run(len(documents), config)

    # Initialize generator and task assigner
    generator = LLMGenerator()
    task_assigner = TaskAssigner(seed=args.seed)

    # Pre-assign splits
    total_examples = len(documents) * 2
    splits = assign_splits(total_examples, settings.TRAINING_VAL_SPLIT, args.seed)
    split_index = 0

    # Process documents
    processed = 0
    examples_generated = 0

    print(f"\nProcessing {len(documents)} documents...")

    with tqdm(total=len(documents), desc="Generating examples") as pbar:
        for doc in documents:
            if shutdown_requested:
                logger.info("Shutdown requested, saving progress...")
                break

            cnr = doc["cnr"]
            full_text = doc["full_text"]

            if not full_text:
                logger.warning(f"No text for {cnr}, skipping")
                training_store.save_progress(
                    TrainingProgress(
                        cnr=cnr, status="skipped", error_message="No full_text"
                    )
                )
                pbar.update(1)
                continue

            # Mark as in progress
            progress = TrainingProgress(cnr=cnr, status="in_progress")
            training_store.save_progress(progress)

            # Get task types for this document
            task_types = task_assigner.assign_tasks(cnr)

            # Generate examples
            result = generator.generate_for_document(cnr, full_text, task_types)

            if result.success and result.examples:
                # Save examples
                for example in result.examples:
                    split = (
                        splits[split_index] if split_index < len(splits) else "train"
                    )
                    split_index += 1

                    training_example = TrainingExample(
                        cnr=example.cnr,
                        task_type=example.task_type.value,
                        instruction=example.instruction,
                        input=example.input_text,
                        output=example.output_text,
                        split=split,
                        input_tokens=example.input_tokens,
                        output_tokens=example.output_tokens,
                    )
                    training_store.save_example(training_example)
                    examples_generated += 1

                # Update progress
                progress.status = "completed"
                progress.task_types_generated = [t.value for t in task_types]
                progress.examples_generated = len(result.examples)
                progress.input_tokens = result.total_input_tokens
                progress.output_tokens = result.total_output_tokens
            else:
                progress.status = "failed"
                progress.error_message = result.error_message
                progress.retry_count += 1

            progress.updated_at = datetime.now()
            training_store.save_progress(progress)

            processed += 1
            pbar.update(1)

            # Check cost limit
            current_cost = generator.get_estimated_cost()
            if current_cost > args.cost_limit:
                logger.warning(f"Cost limit reached: ${current_cost:.2f}")
                print(f"\nCost limit of ${args.cost_limit:.2f} reached!")
                break

            # Periodic checkpoint
            if processed % settings.TRAINING_CHECKPOINT_INTERVAL == 0:
                run.documents_processed = processed
                run.examples_generated = examples_generated
                run.total_input_tokens = generator.total_input_tokens
                run.total_output_tokens = generator.total_output_tokens
                run.estimated_cost = generator.get_estimated_cost()
                training_store.update_run(run)
                logger.info(
                    f"Checkpoint: {processed} docs, ${run.estimated_cost:.2f} cost"
                )

    # Finalize run
    run.documents_processed = processed
    run.examples_generated = examples_generated
    run.total_input_tokens = generator.total_input_tokens
    run.total_output_tokens = generator.total_output_tokens
    run.estimated_cost = generator.get_estimated_cost()
    run.completed_at = datetime.now()
    run.status = "interrupted" if shutdown_requested else "completed"
    training_store.update_run(run)

    # Export to JSONL
    if not shutdown_requested:
        print("\nExporting to JSONL...")
        train_path, val_path = training_store.export_to_jsonl()
        print(f"  Train: {train_path}")
        print(f"  Val: {val_path}")

    # Print summary
    stats = training_store.get_stats()
    print("\n" + "=" * 50)
    print("GENERATION COMPLETE" if not shutdown_requested else "GENERATION INTERRUPTED")
    print("=" * 50)
    print(f"Documents processed: {processed:,}")
    print(f"Examples generated: {examples_generated:,}")
    print(f"Estimated cost: ${run.estimated_cost:.2f}")
    print(f"\nBy task type:")
    for task, count in stats.get("by_task_type", {}).items():
        print(f"  {task}: {count:,}")
    print(f"\nBy split:")
    for split, count in stats.get("by_split", {}).items():
        print(f"  {split}: {count:,}")

    if shutdown_requested:
        print("\nProgress saved. Run with --resume to continue.")


if __name__ == "__main__":
    main()
