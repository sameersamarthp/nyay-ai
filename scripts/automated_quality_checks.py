#!/usr/bin/env python3
"""
Automated quality checks for generated training data.

Flags examples that may need manual review.

Usage:
    python scripts/automated_quality_checks.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.aws_document_store import AWSDocumentStore


def check_output_quality(example: dict, source_doc: str) -> list[str]:
    """Run automated quality checks on a generated example.

    Returns:
        List of warning messages (empty if all checks pass).
    """
    warnings = []
    output = example['output']
    task_type = example['metadata']['task_type']

    # Check 1: Output length
    if len(output) < 100:
        warnings.append("⚠️  Output too short (< 100 chars)")
    if len(output) > 5000:
        warnings.append("⚠️  Output very long (> 5000 chars)")

    # Check 2: Hallucination detection - check for common patterns
    hallucination_patterns = [
        r"I don't have",
        r"I cannot",
        r"As an AI",
        r"I apologize",
        r"I'm unable",
        r"not available in the",
        r"cannot find",
        r"based on the information provided, I cannot",
    ]
    for pattern in hallucination_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            warnings.append(f"⚠️  Possible refusal/uncertainty: '{pattern}'")

    # Check 3: Empty sections (for structured outputs)
    if "Not mentioned" in output and output.count("Not mentioned") > 5:
        warnings.append("⚠️  Too many 'Not mentioned' fields")

    # Check 4: Repetition detection
    words = output.lower().split()
    if len(words) > 50:
        word_counts = Counter(words)
        most_common = word_counts.most_common(1)[0]
        if most_common[1] / len(words) > 0.1:  # Single word > 10% of output
            warnings.append(
                f"⚠️  Excessive repetition: '{most_common[0]}' appears {most_common[1]} times"
            )

    # Check 5: Task-specific checks
    if task_type == "summarization":
        if len(output) < 200:
            warnings.append("⚠️  Summary too brief for legal document")
        if "summary" not in output.lower() and "case" not in output.lower():
            warnings.append("⚠️  Missing key legal summary terms")

    if task_type == "research_qa":
        if "QUESTION:" not in output or "ANSWER:" not in output:
            warnings.append("⚠️  Missing Q&A format markers")

    if task_type == "outcome_analysis":
        outcome_keywords = ["outcome", "decision", "allowed", "dismissed", "disposed"]
        if not any(kw in output.lower() for kw in outcome_keywords):
            warnings.append("⚠️  Missing outcome indicators")

    if task_type == "info_extraction":
        if output.count(":") < 3:  # Structured extraction should have multiple fields
            warnings.append("⚠️  Insufficient structure for extraction task")

    # Check 6: Factual consistency (basic)
    # Extract dates from source and check if they appear in output
    source_dates = re.findall(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", source_doc)
    if source_dates:
        # Check if at least one date from source appears in output
        dates_in_output = any(date in output for date in source_dates[:5])
        if not dates_in_output and task_type in ["summarization", "info_extraction"]:
            warnings.append("⚠️  No dates from source document in output")

    return warnings


def main():
    print("Running automated quality checks...")
    print("=" * 80)

    # Load examples
    examples = []
    with open("data/training/train.jsonl", "r") as f:
        examples.extend([("train", json.loads(line)) for line in f])
    with open("data/training/val.jsonl", "r") as f:
        examples.extend([("val", json.loads(line)) for line in f])

    print(f"Loaded {len(examples)} examples\n")

    # Load source documents
    store = AWSDocumentStore()

    flagged = []
    clean = 0

    for split, example in examples:
        cnr = example['metadata']['cnr']
        doc = store.get_document(cnr)

        if not doc or not doc.full_text:
            flagged.append((split, cnr, ["❌ Source document not found"]))
            continue

        warnings = check_output_quality(example, doc.full_text)

        if warnings:
            flagged.append((split, cnr, warnings))
        else:
            clean += 1

    # Report results
    print("\n" + "=" * 80)
    print("QUALITY CHECK RESULTS")
    print("=" * 80)
    print(f"Total examples: {len(examples)}")
    print(f"Clean examples: {clean} ({clean/len(examples)*100:.1f}%)")
    print(f"Flagged for review: {len(flagged)} ({len(flagged)/len(examples)*100:.1f}%)")

    if flagged:
        print("\n" + "=" * 80)
        print("FLAGGED EXAMPLES (need manual review):")
        print("=" * 80)

        for split, cnr, warnings in flagged:
            print(f"\n[{split.upper()}] CNR: {cnr}")
            for warning in warnings:
                print(f"  {warning}")

        # Summary by warning type
        print("\n" + "=" * 80)
        print("WARNING SUMMARY:")
        print("=" * 80)
        warning_counts = Counter()
        for _, _, warnings in flagged:
            for warning in warnings:
                # Extract warning type (first part before colon)
                warning_type = warning.split(":")[0].strip()
                warning_counts[warning_type] += 1

        for warning_type, count in warning_counts.most_common():
            print(f"{warning_type}: {count}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    if len(flagged) == 0:
        print("✅ All examples passed automated checks!")
        print("   Still recommend manual review of 5-10% sample.")
    elif len(flagged) / len(examples) < 0.1:
        print(f"✅ Only {len(flagged)/len(examples)*100:.1f}% flagged - good quality!")
        print("   Manually review flagged examples above.")
    elif len(flagged) / len(examples) < 0.3:
        print(f"⚠️  {len(flagged)/len(examples)*100:.1f}% flagged - moderate issues.")
        print("   Review flagged examples and random 10% sample.")
    else:
        print(f"❌ {len(flagged)/len(examples)*100:.1f}% flagged - significant issues!")
        print("   Consider regenerating with better prompts or different model.")


if __name__ == "__main__":
    main()
