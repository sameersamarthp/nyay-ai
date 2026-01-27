#!/usr/bin/env python3
"""
Filter out bad examples from training data.

Usage:
    # Create a list of bad CNRs (one per line)
    echo "DLHC010011762025" > bad_cnrs.txt
    echo "HCBM030212662022" >> bad_cnrs.txt

    # Filter them out
    python scripts/filter_bad_examples.py \
      --input data/training/train.jsonl \
      --remove bad_cnrs.txt \
      --output data/training/train_clean.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_script_logger

script_output = get_script_logger(__name__)


def filter_examples(input_file: Path, remove_cnrs: set[str], output_file: Path) -> None:
    """Filter out examples with specified CNRs.

    Args:
        input_file: Input JSONL file
        remove_cnrs: Set of CNRs to remove
        output_file: Output JSONL file
    """
    kept = 0
    removed = 0

    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            example = json.loads(line)
            cnr = example["metadata"]["cnr"]

            if cnr in remove_cnrs:
                removed += 1
                script_output.info(f"Removing: {cnr} ({example['metadata']['task_type']})")
            else:
                fout.write(line)
                kept += 1

    script_output.info(f"\n{'='*60}")
    script_output.info(f"Filtering complete:")
    script_output.info(f"  Kept: {kept}")
    script_output.info(f"  Removed: {removed}")
    script_output.info(f"  Output: {output_file}")
    script_output.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Filter bad examples from training data")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file (e.g., train.jsonl)",
    )
    parser.add_argument(
        "--remove",
        type=Path,
        required=True,
        help="Text file with CNRs to remove (one per line)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file (e.g., train_clean.jsonl)",
    )

    args = parser.parse_args()

    # Load CNRs to remove
    with open(args.remove, "r") as f:
        remove_cnrs = set(line.strip() for line in f if line.strip())

    script_output.info(f"Loaded {len(remove_cnrs)} CNRs to remove")

    # Filter
    filter_examples(args.input, remove_cnrs, args.output)


if __name__ == "__main__":
    main()
