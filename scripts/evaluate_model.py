#!/usr/bin/env python3
"""
Evaluate fine-tuned Nyay AI model.

This script:
1. Loads the fine-tuned model
2. Runs evaluation on validation set
3. Generates sample outputs for manual review
4. Computes metrics (loss, perplexity, accuracy)

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --checkpoint models/nyay-ai-checkpoints/checkpoint-3000
    python scripts/evaluate_model.py --samples 10  # Generate 10 sample outputs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import mlx.core as mx
import numpy as np
import yaml
from mlx_lm import generate, load
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger, get_script_logger

logger = get_logger(__name__)
script_output = get_script_logger(__name__)


def load_config(config_path: Path) -> dict:
    """Load training configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset(data_path: Path, max_samples: int = None) -> List[dict]:
    """Load dataset from JSONL file."""
    examples = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            examples.append(json.loads(line))
    return examples


def evaluate_perplexity(model, tokenizer, dataset: List[dict]) -> Dict[str, float]:
    """Compute perplexity on validation set.

    Args:
        model: The model
        tokenizer: Tokenizer
        dataset: Validation dataset

    Returns:
        Dictionary of metrics
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for example in tqdm(dataset, desc="Computing perplexity"):
        messages = example["messages"]

        # Tokenize
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        tokens = tokenizer.encode(text, add_special_tokens=True)

        if len(tokens) < 2:
            continue

        # Prepare inputs
        input_ids = mx.array([tokens[:-1]])
        labels = mx.array([tokens[1:]])

        # Forward pass
        logits = model(input_ids)

        # Compute loss
        from mlx import nn
        loss = nn.losses.cross_entropy(
            logits[0],
            labels[0],
            reduction='mean'
        )

        total_loss += loss.item() * (len(tokens) - 1)
        total_tokens += len(tokens) - 1

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
        "eval_samples": len(dataset),
        "eval_tokens": total_tokens
    }


def generate_samples(
    model,
    tokenizer,
    dataset: List[dict],
    num_samples: int = 5,
    max_tokens: int = 512
) -> List[Dict]:
    """Generate sample outputs for manual review.

    Args:
        model: The model
        tokenizer: Tokenizer
        dataset: Dataset to sample from
        num_samples: Number of samples to generate
        max_tokens: Max tokens to generate

    Returns:
        List of examples with generated outputs
    """
    model.eval()

    samples = []
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in tqdm(indices, desc="Generating samples"):
        example = dataset[idx]
        messages = example["messages"]

        # Prepare input (system + user only)
        input_messages = messages[:2]  # system + user

        # Format prompt
        prompt = tokenizer.apply_chat_template(
            input_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            verbose=False
        )

        samples.append({
            "task_type": example["metadata"]["task_type"],
            "input": input_messages[1]["content"],
            "reference_output": messages[2]["content"],
            "generated_output": response,
            "metadata": example["metadata"]
        })

    return samples


def compute_accuracy_metrics(samples: List[Dict]) -> Dict[str, float]:
    """Compute accuracy metrics on generated samples.

    Args:
        samples: List of samples with reference and generated outputs

    Returns:
        Dictionary of accuracy metrics
    """
    # Simple metrics based on length and content similarity
    metrics = {
        "avg_reference_length": np.mean([len(s["reference_output"]) for s in samples]),
        "avg_generated_length": np.mean([len(s["generated_output"]) for s in samples]),
        "num_samples": len(samples)
    }

    # Check for hallucination indicators
    hallucination_keywords = [
        "not mentioned", "not specified", "not provided",
        "unclear", "cannot determine"
    ]

    hallucinations = 0
    for sample in samples:
        gen = sample["generated_output"].lower()
        if any(keyword in gen for keyword in hallucination_keywords):
            hallucinations += 1

    metrics["hallucination_rate"] = hallucinations / len(samples) if samples else 0

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Nyay AI model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/training_config.yaml"),
        help="Training config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to checkpoint (default: use base model)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples to generate for review"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results.json"),
        help="Output file for results"
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        help="Max samples for perplexity evaluation (default: all)"
    )

    args = parser.parse_args()

    # Load configuration
    script_output.info("Loading configuration...")
    config = load_config(args.config)

    # Load model
    script_output.info(f"Loading model...")
    model_path = args.checkpoint if args.checkpoint else config["model"]["path"]

    try:
        model, tokenizer = load(str(model_path))
        script_output.info(f"✓ Model loaded from {model_path}")
    except Exception as e:
        script_output.error(f"Failed to load model: {e}")
        return

    # Load validation set
    script_output.info("\nLoading validation set...")
    val_path = Path(config["training"]["val_data"]).parent / "val_mlx.jsonl"

    if not val_path.exists():
        script_output.error(f"Validation data not found: {val_path}")
        script_output.info("Run: python scripts/convert_to_mlx_format.py")
        return

    val_dataset = load_dataset(val_path, args.max_eval_samples)
    script_output.info(f"  Loaded: {len(val_dataset):,} examples")

    # Evaluate perplexity
    script_output.info("\n" + "=" * 70)
    script_output.info("EVALUATING PERPLEXITY")
    script_output.info("=" * 70)

    perplexity_metrics = evaluate_perplexity(model, tokenizer, val_dataset)

    script_output.info(f"\nResults:")
    script_output.info(f"  Loss: {perplexity_metrics['eval_loss']:.4f}")
    script_output.info(f"  Perplexity: {perplexity_metrics['eval_perplexity']:.2f}")
    script_output.info(f"  Samples: {perplexity_metrics['eval_samples']:,}")
    script_output.info(f"  Tokens: {perplexity_metrics['eval_tokens']:,}")

    # Generate samples
    script_output.info("\n" + "=" * 70)
    script_output.info(f"GENERATING {args.samples} SAMPLE OUTPUTS")
    script_output.info("=" * 70)

    samples = generate_samples(
        model,
        tokenizer,
        val_dataset,
        num_samples=args.samples,
        max_tokens=config["training"].get("max_seq_length", 512)
    )

    # Compute accuracy metrics
    accuracy_metrics = compute_accuracy_metrics(samples)

    script_output.info(f"\nGeneration Metrics:")
    script_output.info(f"  Avg reference length: {accuracy_metrics['avg_reference_length']:.0f} chars")
    script_output.info(f"  Avg generated length: {accuracy_metrics['avg_generated_length']:.0f} chars")
    script_output.info(f"  Hallucination rate: {accuracy_metrics['hallucination_rate']:.1%}")

    # Display samples
    script_output.info("\n" + "=" * 70)
    script_output.info("SAMPLE OUTPUTS (first 3)")
    script_output.info("=" * 70)

    for i, sample in enumerate(samples[:3], 1):
        script_output.info(f"\n--- Sample {i} ---")
        script_output.info(f"Task: {sample['task_type']}")
        script_output.info(f"\nInput: {sample['input'][:200]}...")
        script_output.info(f"\nReference Output: {sample['reference_output'][:200]}...")
        script_output.info(f"\nGenerated Output: {sample['generated_output'][:200]}...")

    # Save results
    results = {
        "perplexity_metrics": perplexity_metrics,
        "accuracy_metrics": accuracy_metrics,
        "samples": samples,
        "config": {
            "model_path": str(model_path),
            "val_samples": len(val_dataset),
            "generated_samples": len(samples)
        }
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    script_output.info(f"\n✅ Evaluation complete!")
    script_output.info(f"Results saved to: {args.output}")

    # Summary
    script_output.info("\n" + "=" * 70)
    script_output.info("EVALUATION SUMMARY")
    script_output.info("=" * 70)
    script_output.info(f"Perplexity: {perplexity_metrics['eval_perplexity']:.2f}")
    script_output.info(f"Loss: {perplexity_metrics['eval_loss']:.4f}")
    script_output.info(f"Hallucination rate: {accuracy_metrics['hallucination_rate']:.1%}")

    # Quality assessment
    if perplexity_metrics['eval_perplexity'] < 3.0:
        script_output.info("\n✅ Model quality: EXCELLENT (perplexity < 3.0)")
    elif perplexity_metrics['eval_perplexity'] < 5.0:
        script_output.info("\n✅ Model quality: GOOD (perplexity < 5.0)")
    elif perplexity_metrics['eval_perplexity'] < 10.0:
        script_output.info("\n⚠️  Model quality: FAIR (perplexity < 10.0)")
    else:
        script_output.info("\n❌ Model quality: POOR (perplexity >= 10.0)")
        script_output.info("   Consider training longer or adjusting hyperparameters")


if __name__ == "__main__":
    main()
