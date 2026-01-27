#!/usr/bin/env python3
"""
Convert training data from JSONL to MLX chat format.

MLX-LM expects data in the format:
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}

Usage:
    python scripts/convert_to_mlx_format.py
    python scripts/convert_to_mlx_format.py --input data/training/train.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger, get_script_logger

logger = get_logger(__name__)
script_output = get_script_logger(__name__)


def load_config(config_path: Path) -> dict:
    """Load training configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_system_prompt(task_type: str, config: dict) -> str:
    """Get system prompt for a task type."""
    system_prompts = config.get("system_prompts", {})

    # Map task types to system prompts
    prompt_map = {
        "summarization": system_prompts.get("summarization"),
        "research_qa": system_prompts.get("research_qa"),
        "outcome_analysis": system_prompts.get("outcome_analysis"),
        "info_extraction": system_prompts.get("info_extraction"),
    }

    return prompt_map.get(task_type, system_prompts.get("default", ""))


def convert_example_to_chat(example: dict, config: dict) -> dict:
    """Convert a training example to chat format.

    Args:
        example: JSONL example with instruction, input, output, metadata
        config: Training configuration

    Returns:
        Example in chat format for MLX
    """
    task_type = example["metadata"]["task_type"]
    system_prompt = get_system_prompt(task_type, config)

    # Combine instruction and input as user message
    user_message = f"{example['instruction']}\n\n{example['input']}"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": example["output"]}
        ],
        "metadata": example.get("metadata", {})
    }


def convert_jsonl_to_chat(
    input_path: Path,
    output_path: Path,
    config: dict
) -> None:
    """Convert JSONL file to chat format.

    Args:
        input_path: Input JSONL file
        output_path: Output JSONL file (chat format)
        config: Training configuration
    """
    converted = 0
    errors = 0

    with open(input_path) as infile, open(output_path, "w") as outfile:
        for line_no, line in enumerate(infile, 1):
            try:
                example = json.loads(line)
                chat_example = convert_example_to_chat(example, config)
                outfile.write(json.dumps(chat_example) + "\n")
                converted += 1
            except Exception as e:
                logger.error(f"Error converting line {line_no}: {e}")
                errors += 1

    return converted, errors


def main():
    parser = argparse.ArgumentParser(
        description="Convert training data to MLX chat format"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/training_config.yaml"),
        help="Training config file"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input JSONL file (default: from config)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL file (default: {input}_mlx.jsonl)"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "both"],
        default="both",
        help="Which split to convert"
    )

    args = parser.parse_args()

    # Load config
    script_output.info("Loading configuration...")
    config = load_config(args.config)

    # Determine input/output paths
    splits_to_process = []
    if args.split in ["train", "both"]:
        splits_to_process.append(("train", Path(config["training"]["train_data"])))
    if args.split in ["val", "both"]:
        splits_to_process.append(("val", Path(config["training"]["val_data"])))

    # Convert each split
    for split_name, input_path in splits_to_process:
        if not input_path.exists():
            script_output.error(f"Input file not found: {input_path}")
            continue

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = input_path.parent / f"{input_path.stem}_mlx.jsonl"

        script_output.info(f"\nConverting {split_name} split...")
        script_output.info(f"  Input:  {input_path}")
        script_output.info(f"  Output: {output_path}")

        converted, errors = convert_jsonl_to_chat(input_path, output_path, config)

        script_output.info(f"  Converted: {converted:,} examples")
        if errors > 0:
            script_output.info(f"  Errors: {errors}")

    script_output.info("\n✅ Conversion complete!")
    script_output.info("\nNext steps:")
    script_output.info("  1. Review converted files")
    script_output.info("  2. Run training: python scripts/train_model.py")


if __name__ == "__main__":
    main()
