#!/usr/bin/env python3
"""
Helper script for manual review of generated training data.

Usage:
    python scripts/manual_review_helper.py --sample 50
    python scripts/manual_review_helper.py --cnr DLHC010011762025
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.aws_document_store import AWSDocumentStore
from utils.logger import get_script_logger

script_output = get_script_logger(__name__)


def display_example_for_review(example: dict, source_doc: str):
    """Display an example alongside source document for review."""
    script_output.info("\n" + "=" * 80)
    script_output.info(f"CNR: {example['metadata']['cnr']}")
    script_output.info(f"Task Type: {example['metadata']['task_type']}")
    script_output.info("=" * 80)

    script_output.info("\n📄 SOURCE DOCUMENT (First 3000 chars):")
    script_output.info("-" * 80)
    script_output.info(source_doc[:3000])
    if len(source_doc) > 3000:
        script_output.info(f"\n... [truncated, {len(source_doc) - 3000} more chars]")

    script_output.info("\n" + "=" * 80)
    script_output.info("📝 INSTRUCTION:")
    script_output.info("-" * 80)
    script_output.info(example['instruction'])

    script_output.info("\n" + "=" * 80)
    script_output.info("🤖 GENERATED OUTPUT:")
    script_output.info("-" * 80)
    script_output.info(example['output'])

    script_output.info("\n" + "=" * 80)
    script_output.info("🔍 REVIEW CHECKLIST:")
    script_output.info("-" * 80)
    script_output.info("1. [ ] Factually accurate (check dates, parties, outcomes)")
    script_output.info("2. [ ] Legally sound (correct interpretation of law)")
    script_output.info("3. [ ] No hallucination (all facts from source document)")
    script_output.info("4. [ ] Clear and well-structured")
    script_output.info("5. [ ] Appropriate length and detail")
    script_output.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Manual review helper")
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of random examples to review",
    )
    parser.add_argument(
        "--cnr",
        type=str,
        help="Review specific CNR",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "both"],
        default="both",
        help="Which split to review",
    )

    args = parser.parse_args()

    # Load examples
    examples = []
    if args.split in ["train", "both"]:
        with open("data/training/train.jsonl", "r") as f:
            examples.extend([json.loads(line) for line in f])
    if args.split in ["val", "both"]:
        with open("data/training/val.jsonl", "r") as f:
            examples.extend([json.loads(line) for line in f])

    script_output.info(f"Loaded {len(examples)} examples")

    # Filter by CNR if specified
    if args.cnr:
        examples = [ex for ex in examples if ex['metadata']['cnr'] == args.cnr]
        if not examples:
            script_output.info(f"No examples found for CNR: {args.cnr}")
            return
    else:
        # Random sample
        examples = random.sample(examples, min(args.sample, len(examples)))

    # Load source documents
    store = AWSDocumentStore()

    script_output.info(f"\nReviewing {len(examples)} examples...")

    for i, example in enumerate(examples, 1):
        cnr = example['metadata']['cnr']
        doc = store.get_document(cnr)

        if not doc or not doc.full_text:
            script_output.info(f"\nSkipping {cnr} - no source document found")
            continue

        script_output.info(f"\n\n{'='*80}")
        script_output.info(f"EXAMPLE {i} of {len(examples)}")
        display_example_for_review(example, doc.full_text)

        if i < len(examples):
            response = input("\n\nPress Enter for next, 'q' to quit: ")
            if response.lower() == 'q':
                break


if __name__ == "__main__":
    main()
