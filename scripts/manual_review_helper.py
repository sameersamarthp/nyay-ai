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


def display_example_for_review(example: dict, source_doc: str):
    """Display an example alongside source document for review."""
    print("\n" + "=" * 80)
    print(f"CNR: {example['metadata']['cnr']}")
    print(f"Task Type: {example['metadata']['task_type']}")
    print("=" * 80)

    print("\n📄 SOURCE DOCUMENT (First 3000 chars):")
    print("-" * 80)
    print(source_doc[:3000])
    if len(source_doc) > 3000:
        print(f"\n... [truncated, {len(source_doc) - 3000} more chars]")

    print("\n" + "=" * 80)
    print("📝 INSTRUCTION:")
    print("-" * 80)
    print(example['instruction'])

    print("\n" + "=" * 80)
    print("🤖 GENERATED OUTPUT:")
    print("-" * 80)
    print(example['output'])

    print("\n" + "=" * 80)
    print("🔍 REVIEW CHECKLIST:")
    print("-" * 80)
    print("1. [ ] Factually accurate (check dates, parties, outcomes)")
    print("2. [ ] Legally sound (correct interpretation of law)")
    print("3. [ ] No hallucination (all facts from source document)")
    print("4. [ ] Clear and well-structured")
    print("5. [ ] Appropriate length and detail")
    print("=" * 80)


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

    print(f"Loaded {len(examples)} examples")

    # Filter by CNR if specified
    if args.cnr:
        examples = [ex for ex in examples if ex['metadata']['cnr'] == args.cnr]
        if not examples:
            print(f"No examples found for CNR: {args.cnr}")
            return
    else:
        # Random sample
        examples = random.sample(examples, min(args.sample, len(examples)))

    # Load source documents
    store = AWSDocumentStore()

    print(f"\nReviewing {len(examples)} examples...")

    for i, example in enumerate(examples, 1):
        cnr = example['metadata']['cnr']
        doc = store.get_document(cnr)

        if not doc or not doc.full_text:
            print(f"\nSkipping {cnr} - no source document found")
            continue

        print(f"\n\n{'='*80}")
        print(f"EXAMPLE {i} of {len(examples)}")
        display_example_for_review(example, doc.full_text)

        if i < len(examples):
            response = input("\n\nPress Enter for next, 'q' to quit: ")
            if response.lower() == 'q':
                break


if __name__ == "__main__":
    main()
