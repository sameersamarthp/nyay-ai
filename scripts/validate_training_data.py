#!/usr/bin/env python3
"""
Validate generated training data.

Checks:
- File format (valid JSONL)
- Required fields present
- Distribution across task types
- Output quality metrics

Usage:
    python scripts/validate_training_data.py
    python scripts/validate_training_data.py --input-dir ./data/training
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from utils.logger import get_logger, get_script_logger

logger = get_logger(__name__)
script_output = get_script_logger(__name__)


def validate_jsonl_file(file_path: Path) -> tuple[bool, dict]:
    """Validate a JSONL file.

    Args:
        file_path: Path to JSONL file.

    Returns:
        Tuple of (is_valid, stats_dict).
    """
    stats = {
        "total_lines": 0,
        "valid_lines": 0,
        "invalid_lines": 0,
        "missing_fields": Counter(),
        "task_types": Counter(),
        "avg_instruction_len": 0,
        "avg_input_len": 0,
        "avg_output_len": 0,
        "errors": [],
    }

    required_fields = ["instruction", "input", "output"]
    instruction_lens = []
    input_lens = []
    output_lens = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                stats["total_lines"] += 1
                line = line.strip()

                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Check required fields
                    missing = [f for f in required_fields if f not in data]
                    if missing:
                        stats["invalid_lines"] += 1
                        for field in missing:
                            stats["missing_fields"][field] += 1
                        stats["errors"].append(
                            f"Line {line_num}: Missing fields {missing}"
                        )
                        continue

                    # Check for empty values
                    empty = [f for f in required_fields if not data[f].strip()]
                    if empty:
                        stats["invalid_lines"] += 1
                        stats["errors"].append(f"Line {line_num}: Empty fields {empty}")
                        continue

                    stats["valid_lines"] += 1

                    # Collect metrics
                    instruction_lens.append(len(data["instruction"]))
                    input_lens.append(len(data["input"]))
                    output_lens.append(len(data["output"]))

                    # Track task type from metadata
                    if "metadata" in data and "task_type" in data["metadata"]:
                        stats["task_types"][data["metadata"]["task_type"]] += 1

                except json.JSONDecodeError as e:
                    stats["invalid_lines"] += 1
                    stats["errors"].append(f"Line {line_num}: Invalid JSON - {e}")

    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return False, stats

    # Calculate averages
    if instruction_lens:
        stats["avg_instruction_len"] = sum(instruction_lens) / len(instruction_lens)
    if input_lens:
        stats["avg_input_len"] = sum(input_lens) / len(input_lens)
    if output_lens:
        stats["avg_output_len"] = sum(output_lens) / len(output_lens)

    is_valid = stats["invalid_lines"] == 0
    return is_valid, stats


def check_distribution(train_stats: dict, val_stats: dict) -> list[str]:
    """Check if distribution meets requirements.

    Returns:
        List of warning messages.
    """
    warnings = []

    # Check train/val split ratio
    total = train_stats["valid_lines"] + val_stats["valid_lines"]
    if total > 0:
        val_ratio = val_stats["valid_lines"] / total
        if val_ratio < 0.08 or val_ratio > 0.12:
            warnings.append(
                f"Val split ratio {val_ratio:.2%} outside expected 10% +/- 2%"
            )

    # Check task type distribution
    all_tasks = Counter()
    all_tasks.update(train_stats["task_types"])
    all_tasks.update(val_stats["task_types"])

    if all_tasks:
        expected_per_type = total / len(all_tasks)
        for task, count in all_tasks.items():
            deviation = abs(count - expected_per_type) / expected_per_type
            if deviation > 0.2:  # More than 20% deviation
                warnings.append(
                    f"Task '{task}' has {count} examples, "
                    f"expected ~{expected_per_type:.0f} (deviation: {deviation:.1%})"
                )

    return warnings


def main():
    parser = argparse.ArgumentParser(description="Validate training data")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=settings.TRAINING_DATA_DIR,
        help="Directory containing train.jsonl and val.jsonl",
    )

    args = parser.parse_args()

    train_path = args.input_dir / "train.jsonl"
    val_path = args.input_dir / "val.jsonl"

    # Check files exist
    if not train_path.exists():
        script_output.error(f"Error: {train_path} not found")
        sys.exit(1)

    if not val_path.exists():
        script_output.error(f"Error: {val_path} not found")
        sys.exit(1)

    script_output.info("Validating training data...")
    script_output.info("=" * 50)

    # Validate train.jsonl
    script_output.info(f"\nValidating {train_path.name}...")
    train_valid, train_stats = validate_jsonl_file(train_path)

    script_output.info(f"  Total lines: {train_stats['total_lines']:,}")
    script_output.info(f"  Valid examples: {train_stats['valid_lines']:,}")
    script_output.info(f"  Invalid examples: {train_stats['invalid_lines']:,}")
    script_output.info(
        f"  Avg instruction length: {train_stats['avg_instruction_len']:.0f} chars"
    )
    script_output.info(f"  Avg input length: {train_stats['avg_input_len']:.0f} chars")
    script_output.info(f"  Avg output length: {train_stats['avg_output_len']:.0f} chars")

    if train_stats["task_types"]:
        script_output.info(f"  Task type distribution:")
        for task, count in sorted(train_stats["task_types"].items()):
            script_output.info(f"    {task}: {count:,}")

    if train_stats["errors"][:5]:
        script_output.info(f"  First errors:")
        for error in train_stats["errors"][:5]:
            script_output.info(f"    {error}")

    # Validate val.jsonl
    script_output.info(f"\nValidating {val_path.name}...")
    val_valid, val_stats = validate_jsonl_file(val_path)

    script_output.info(f"  Total lines: {val_stats['total_lines']:,}")
    script_output.info(f"  Valid examples: {val_stats['valid_lines']:,}")
    script_output.info(f"  Invalid examples: {val_stats['invalid_lines']:,}")
    script_output.info(f"  Avg instruction length: {val_stats['avg_instruction_len']:.0f} chars")
    script_output.info(f"  Avg input length: {val_stats['avg_input_len']:.0f} chars")
    script_output.info(f"  Avg output length: {val_stats['avg_output_len']:.0f} chars")

    if val_stats["task_types"]:
        script_output.info(f"  Task type distribution:")
        for task, count in sorted(val_stats["task_types"].items()):
            script_output.info(f"    {task}: {count:,}")

    # Check distribution
    script_output.info("\nDistribution Analysis...")
    warnings = check_distribution(train_stats, val_stats)

    if warnings:
        script_output.info("  Warnings:")
        for warning in warnings:
            script_output.info(f"    - {warning}")
    else:
        script_output.info("  Distribution looks good!")

    # Summary
    script_output.info("\n" + "=" * 50)
    total_examples = train_stats["valid_lines"] + val_stats["valid_lines"]
    script_output.info(f"Total valid examples: {total_examples:,}")
    script_output.info(f"Train: {train_stats['valid_lines']:,}")
    script_output.info(f"Val: {val_stats['valid_lines']:,}")

    all_valid = train_valid and val_valid and not warnings
    if all_valid:
        script_output.info("\nValidation PASSED")
        sys.exit(0)
    else:
        script_output.info("\nValidation FAILED - see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
