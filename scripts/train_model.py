#!/usr/bin/env python3
"""
Fine-tune Llama 3.2 3B with QLoRA 8-bit using MLX.

This script:
1. Loads the base model and applies LoRA adapters
2. Trains on legal judgment data
3. Evaluates on validation set
4. Saves checkpoints periodically

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --config config/training_config.yaml
    python scripts/train_model.py --resume models/nyay-ai-checkpoints/checkpoint-1000
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx_lm import load
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import build_schedule
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
    """Load dataset from JSONL file.

    Args:
        data_path: Path to JSONL file
        max_samples: Maximum number of samples to load (for testing)

    Returns:
        List of examples
    """
    examples = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            examples.append(json.loads(line))
    return examples


def convert_to_training_format(examples: List[dict], tokenizer) -> Tuple[mx.array, mx.array]:
    """Convert examples to MLX training format.

    Args:
        examples: List of chat-formatted examples
        tokenizer: Tokenizer instance

    Returns:
        Tuple of (input_ids, labels)
    """
    all_input_ids = []
    all_labels = []

    for example in examples:
        messages = example["messages"]

        # Tokenize the full conversation
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        tokens = tokenizer.encode(text, add_special_tokens=True)

        all_input_ids.append(tokens)
        all_labels.append(tokens[1:] + [tokenizer.eos_token_id])

    return all_input_ids, all_labels


def compute_loss(model, batch_input_ids, batch_labels, lengths):
    """Compute cross-entropy loss for a batch.

    Args:
        model: The model
        batch_input_ids: Input token IDs
        batch_labels: Target token IDs
        lengths: Sequence lengths

    Returns:
        Loss value
    """
    logits = model(batch_input_ids)

    # Compute cross-entropy loss
    logits = logits.astype(mx.float32)

    # Create loss mask for padding
    loss_mask = mx.arange(logits.shape[1])[None, :] < lengths[:, None]

    # Compute loss only on non-padded tokens
    ce_loss = nn.losses.cross_entropy(
        logits,
        batch_labels,
        reduction='none'
    )

    masked_loss = ce_loss * loss_mask
    return masked_loss.sum() / loss_mask.sum()


def evaluate(
    model,
    val_dataset: List[dict],
    tokenizer,
    max_samples: int = None
) -> Dict[str, float]:
    """Evaluate model on validation set.

    Args:
        model: The model
        val_dataset: Validation dataset
        tokenizer: Tokenizer
        max_samples: Maximum samples to evaluate

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    if max_samples:
        val_dataset = val_dataset[:max_samples]

    total_loss = 0.0
    total_tokens = 0

    for example in tqdm(val_dataset, desc="Evaluating", leave=False):
        messages = example["messages"]

        # Tokenize
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        tokens = tokenizer.encode(text, add_special_tokens=True)

        # Prepare inputs
        input_ids = mx.array([tokens[:-1]])
        labels = mx.array([tokens[1:]])
        lengths = mx.array([len(tokens) - 1])

        # Compute loss
        loss = compute_loss(model, input_ids, labels, lengths)

        total_loss += loss.item() * (len(tokens) - 1)
        total_tokens += len(tokens) - 1

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    model.train()

    return {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
        "eval_samples": len(val_dataset)
    }


def save_checkpoint(
    model,
    optimizer,
    step: int,
    output_dir: Path,
    config: dict,
    metrics: dict = None
):
    """Save training checkpoint.

    Args:
        model: The model
        optimizer: Optimizer state
        step: Current training step
        output_dir: Directory to save checkpoint
        config: Training configuration
        metrics: Optional training metrics
    """
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapters (only trainable parameters)
    adapter_path = checkpoint_dir / "adapters.safetensors"

    # Extract LoRA parameters using MLX tree_flatten
    from mlx.utils import tree_flatten
    params = tree_flatten(model.parameters())
    lora_params = {name: param for name, param in params if "lora" in name.lower()}

    # Save using MLX
    mx.save_safetensors(str(adapter_path), lora_params)

    # Save training state
    state = {
        "step": step,
        "config": config,
        "metrics": metrics or {}
    }

    with open(checkpoint_dir / "training_state.json", "w") as f:
        json.dump(state, f, indent=2)

    logger.info(f"Saved checkpoint to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 3B with QLoRA")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/training_config.yaml"),
        help="Training config file"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode with limited samples"
    )

    args = parser.parse_args()

    # Load configuration
    script_output.info("Loading configuration...")
    config = load_config(args.config)

    # Test mode settings
    if args.test:
        script_output.info("⚠️  TEST MODE: Using limited dataset")
        config["training"]["max_steps"] = 20
        test_samples = 50
    else:
        test_samples = None

    # Set seed for reproducibility
    np.random.seed(config["seed"])
    mx.random.seed(config["seed"])

    # Load model and tokenizer
    script_output.info(f"Loading model: {config['model']['name']}...")
    model_path = config["model"]["path"]

    try:
        model, tokenizer = load(model_path)
        script_output.info(f"✓ Model loaded from {model_path}")
    except Exception as e:
        script_output.error(f"Failed to load model: {e}")
        script_output.info("\nTrying to download model...")
        # Download if not found
        model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
        script_output.info("✓ Model downloaded and loaded")

    # Apply LoRA
    script_output.info("Applying LoRA adapters...")
    lora_config = config["lora"]

    # Convert linear layers to LoRA
    def apply_lora_to_linear(module, name=""):
        if isinstance(module, nn.Linear):
            # Check if this layer should have LoRA
            layer_names = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

            if any(ln in name for ln in layer_names):
                return LoRALinear.from_linear(
                    module,
                    r=lora_config["rank"],
                    alpha=lora_config["alpha"],
                    dropout=lora_config["dropout"],
                    scale=lora_config["alpha"] / lora_config["rank"]
                )

        # Recursively apply to child modules
        if hasattr(module, "children"):
            for child_name, child in module.children().items():
                setattr(module, child_name, apply_lora_to_linear(child, f"{name}.{child_name}"))

        return module

    model = apply_lora_to_linear(model)

    # Count parameters (using MLX tree_flatten)
    from mlx.utils import tree_flatten
    params = tree_flatten(model.parameters())
    total_params = sum(p.size for _, p in params)

    # Count trainable params (LoRA parameters)
    trainable_params = sum(
        p.size for k, p in params if "lora" in k.lower()
    )

    script_output.info(f"✓ LoRA applied")
    script_output.info(f"  Total parameters: {total_params:,}")
    script_output.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # Load datasets
    script_output.info("\nLoading datasets...")
    train_path = Path(config["training"]["train_data"]).parent / "train_mlx.jsonl"
    val_path = Path(config["training"]["val_data"]).parent / "val_mlx.jsonl"

    if not train_path.exists():
        script_output.error(f"Training data not found: {train_path}")
        script_output.info("Run: python scripts/convert_to_mlx_format.py")
        return

    train_dataset = load_dataset(train_path, test_samples)
    val_dataset = load_dataset(val_path, test_samples // 10 if test_samples else None)

    script_output.info(f"  Train: {len(train_dataset):,} examples")
    script_output.info(f"  Val: {len(val_dataset):,} examples")

    # Setup optimizer
    script_output.info("\nSetting up optimizer...")
    learning_rate = config["training"]["learning_rate"]
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Training setup
    max_steps = config["training"]["max_steps"]
    eval_steps = config["evaluation"]["eval_steps"]
    save_steps = config["evaluation"]["save_steps"]
    logging_steps = config["evaluation"]["logging_steps"]

    output_dir = Path(config["checkpointing"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    script_output.info("\n" + "=" * 70)
    script_output.info("STARTING TRAINING")
    script_output.info("=" * 70)
    script_output.info(f"Max steps: {max_steps}")
    script_output.info(f"Batch size: {config['training']['batch_size']}")
    script_output.info(f"Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    script_output.info(f"Learning rate: {learning_rate}")
    script_output.info("=" * 70 + "\n")

    model.train()
    best_eval_loss = float('inf')
    global_step = 0
    start_time = time.time()

    # Use MLX-LM's built-in training function
    from mlx_lm.tuner.trainer import train as mlx_train

    # Create training args
    training_args = TrainingArgs(
        model=model_path,
        data=str(train_path),
        train=True,
        num_iterations=max_steps,
        val_batches=len(val_dataset) // config["evaluation"]["eval_batch_size"],
        learning_rate=learning_rate,
        batch_size=config["training"]["batch_size"],
        iters_to_accumulate=config["training"]["gradient_accumulation_steps"],
        steps_per_eval=eval_steps,
        steps_per_report=logging_steps,
        save_every=save_steps,
        adapter_path=str(output_dir),
        test=False,
        seed=config["seed"]
    )

    script_output.info("Starting MLX training loop...")
    script_output.info(f"Progress will be saved to: {output_dir}")

    # Run training
    try:
        mlx_train(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            optimizer=optimizer
        )

        script_output.info("\n" + "=" * 70)
        script_output.info("TRAINING COMPLETE")
        script_output.info("=" * 70)

        elapsed = time.time() - start_time
        script_output.info(f"Total time: {elapsed/3600:.2f} hours")
        script_output.info(f"Final checkpoint: {output_dir}")

    except KeyboardInterrupt:
        script_output.info("\n⚠️  Training interrupted by user")
        script_output.info(f"Progress saved to: {output_dir}")

    script_output.info("\nNext steps:")
    script_output.info("  1. Evaluate: python scripts/evaluate_model.py")
    script_output.info("  2. Export: python scripts/export_model.py")


if __name__ == "__main__":
    main()
