#!/usr/bin/env python3
"""
Interactive manual review of generated training data.

Collects your assessment for each example and tracks bad CNRs.

Usage:
    python scripts/interactive_review.py --sample 50
    python scripts/interactive_review.py --cnr DLHC010011762025
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.aws_document_store import AWSDocumentStore


class ReviewSession:
    """Track review session results."""

    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.results = []
        self.bad_cnrs = []

    def add_result(self, cnr: str, task_type: str, checks: dict, notes: str = ""):
        """Add review result."""
        result = {
            "cnr": cnr,
            "task_type": task_type,
            "checks": checks,
            "notes": notes,
            "passed": all(checks.values()),
        }
        self.results.append(result)

        if not result["passed"]:
            self.bad_cnrs.append(cnr)

    def save(self):
        """Save review results."""
        # Save detailed results as JSON
        with open(self.output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save bad CNRs to separate file
        bad_cnrs_file = self.output_file.parent / "bad_cnrs.txt"
        if self.bad_cnrs:
            with open(bad_cnrs_file, "w") as f:
                for cnr in self.bad_cnrs:
                    f.write(f"{cnr}\n")

        # Print summary
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed

        print(f"\n{'='*70}")
        print("REVIEW SESSION SUMMARY")
        print(f"{'='*70}")
        print(f"Total reviewed: {total}")
        print(f"Passed all checks: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed one or more: {failed} ({failed/total*100:.1f}%)")
        print(f"\nResults saved to: {self.output_file}")
        if self.bad_cnrs:
            print(f"Bad CNRs saved to: {bad_cnrs_file}")
            print(f"\nTo filter them out:")
            print(f"  python scripts/filter_bad_examples.py \\")
            print(f"    --input data/training/train.jsonl \\")
            print(f"    --remove {bad_cnrs_file} \\")
            print(f"    --output data/training/train_clean.jsonl")


def ask_yes_no(prompt: str, default: str = "y") -> bool:
    """Ask a yes/no question.

    Args:
        prompt: Question to ask
        default: Default answer ('y' or 'n')

    Returns:
        True for yes, False for no
    """
    default_hint = "[Y/n]" if default == "y" else "[y/N]"
    while True:
        response = input(f"{prompt} {default_hint}: ").strip().lower()

        if not response:  # User pressed Enter
            return default == "y"

        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Please enter 'y' or 'n' (or just press Enter for default)")


def review_example(cnr: str, example: dict, source_doc: str) -> dict | None:
    """Review a single example interactively.

    Returns:
        Dict with check results, or None if user wants to skip
    """
    print("\n" + "=" * 80)
    print(f"CNR: {cnr}")
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
    print(example["instruction"])

    print("\n" + "=" * 80)
    print("🤖 GENERATED OUTPUT:")
    print("-" * 80)
    print(example["output"])

    print("\n" + "=" * 80)
    print("🔍 INTERACTIVE REVIEW:")
    print("-" * 80)

    # Ask each checklist question
    checks = {}

    print("\nCompare the generated output with the source document above.\n")

    checks["factually_accurate"] = ask_yes_no(
        "1. Factually accurate (dates, parties, outcomes match source)?", default="y"
    )

    checks["legally_sound"] = ask_yes_no(
        "2. Legally sound (correct interpretation of law)?", default="y"
    )

    checks["no_hallucination"] = ask_yes_no(
        "3. No hallucination (all facts are from source document)?", default="y"
    )

    checks["clear_structured"] = ask_yes_no(
        "4. Clear and well-structured?", default="y"
    )

    checks["appropriate_length"] = ask_yes_no(
        "5. Appropriate length and detail?", default="y"
    )

    # Ask for notes if any check failed
    notes = ""
    if not all(checks.values()):
        print("\nOptional: Add notes about the issues (press Enter to skip):")
        notes = input("> ").strip()

    return {"checks": checks, "notes": notes}


def main():
    parser = argparse.ArgumentParser(description="Interactive manual review")
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training/review_results.json"),
        help="Output file for review results",
    )
    parser.add_argument(
        "--continue-session",
        type=Path,
        help="Continue from previous review session (JSON file)",
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
        examples = [ex for ex in examples if ex["metadata"]["cnr"] == args.cnr]
        if not examples:
            print(f"No examples found for CNR: {args.cnr}")
            return
    else:
        # Random sample
        examples = random.sample(examples, min(args.sample, len(examples)))

    # Initialize or continue session
    session = ReviewSession(args.output)

    # Load previous session if continuing
    already_reviewed = set()
    if args.continue_session:
        with open(args.continue_session, "r") as f:
            previous_results = json.load(f)
            session.results = previous_results
            already_reviewed = {r["cnr"] for r in previous_results}
            print(f"Continuing from previous session: {len(already_reviewed)} already reviewed")

    # Filter out already reviewed
    examples = [ex for ex in examples if ex["metadata"]["cnr"] not in already_reviewed]

    if not examples:
        print("No new examples to review!")
        return

    # Load source documents
    store = AWSDocumentStore()

    print(f"\nReviewing {len(examples)} examples...")
    print("For each question, press Enter for 'yes' or type 'n' for 'no'\n")

    reviewed = 0
    for i, example in enumerate(examples, 1):
        cnr = example["metadata"]["cnr"]
        task_type = example["metadata"]["task_type"]
        doc = store.get_document(cnr)

        if not doc or not doc.full_text:
            print(f"\nSkipping {cnr} - no source document found")
            continue

        print(f"\n\n{'='*80}")
        print(f"EXAMPLE {i} of {len(examples)}")

        result = review_example(cnr, example, doc.full_text)

        if result:
            session.add_result(cnr, task_type, result["checks"], result["notes"])
            reviewed += 1

            # Show progress
            passed = all(result["checks"].values())
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"\n{status} - Example marked as {'good' if passed else 'bad'}")

            # Save after each review (in case of interruption)
            session.save()

        # Ask to continue
        if i < len(examples):
            print("\n" + "-" * 80)
            continue_review = input(
                "Press Enter to continue, 's' to save and stop: "
            ).strip().lower()
            if continue_review == "s":
                print("\nSaving and stopping...")
                break

    print(f"\n✅ Reviewed {reviewed} examples")
    session.save()


if __name__ == "__main__":
    main()
