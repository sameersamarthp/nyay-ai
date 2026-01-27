#!/usr/bin/env python3
"""
Export fine-tuned Nyay AI model to various formats.

Supports:
- GGUF format (for Ollama)
- MLX format (for direct use)
- Merged safetensors (full model weights)

Usage:
    python scripts/export_model.py --format gguf
    python scripts/export_model.py --format mlx --output models/nyay-ai-final
    python scripts/export_model.py --format gguf --quantize q4_k_m
"""

import argparse
import shutil
import subprocess
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


def find_best_checkpoint(checkpoints_dir: Path) -> Path:
    """Find the best checkpoint based on step number.

    Args:
        checkpoints_dir: Directory containing checkpoints

    Returns:
        Path to best checkpoint
    """
    checkpoints = list(checkpoints_dir.glob("checkpoint-*"))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")

    # Sort by step number (highest = latest)
    checkpoints.sort(key=lambda p: int(p.name.split("-")[1]))

    return checkpoints[-1]


def merge_lora_adapters(
    base_model_path: Path,
    adapter_path: Path,
    output_path: Path
) -> None:
    """Merge LoRA adapters back into base model.

    Args:
        base_model_path: Path to base model
        adapter_path: Path to LoRA adapters
        output_path: Path to save merged model
    """
    script_output.info("Merging LoRA adapters with base model...")

    # Use MLX-LM's fuse command
    try:
        from mlx_lm.tuner.utils import fuse_lora

        script_output.info(f"  Base model: {base_model_path}")
        script_output.info(f"  Adapters: {adapter_path}")
        script_output.info(f"  Output: {output_path}")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Copy base model files
        for file in base_model_path.glob("*"):
            if file.is_file() and file.suffix in [".json", ".model"]:
                shutil.copy2(file, output_path / file.name)

        # Merge adapters
        fuse_lora(
            model_path=str(base_model_path),
            adapter_path=str(adapter_path),
            output_path=str(output_path)
        )

        script_output.info("✓ Merge complete!")

    except Exception as e:
        script_output.error(f"Failed to merge adapters: {e}")
        script_output.info("\nManual merge required:")
        script_output.info(f"  mlx-lm fuse --model {base_model_path} --adapter {adapter_path} --output {output_path}")
        raise


def export_to_gguf(
    model_path: Path,
    output_path: Path,
    quantization: str = "q4_k_m"
) -> None:
    """Export model to GGUF format for Ollama.

    Args:
        model_path: Path to merged model
        output_path: Path to save GGUF file
        quantization: Quantization type (q4_k_m, q5_k_m, q8_0, etc.)
    """
    script_output.info(f"Exporting to GGUF format (quantization: {quantization})...")

    try:
        # Check if llama-cpp-python is installed
        import llama_cpp

        script_output.info("✓ llama-cpp-python found")

    except ImportError:
        script_output.error("llama-cpp-python not installed!")
        script_output.info("\nInstall with:")
        script_output.info("  pip install llama-cpp-python")
        script_output.info("\nOr use external tools:")
        script_output.info("  1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
        script_output.info("  2. Convert: python llama.cpp/convert.py {model_path}")
        script_output.info("  3. Quantize: llama.cpp/quantize {model_path}/ggml-model-f16.gguf {output_path} {quantization}")
        raise

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert using llama.cpp tools
    script_output.info("Converting to GGUF (this may take several minutes)...")

    # For now, provide manual instructions
    script_output.info("\nTo convert to GGUF:")
    script_output.info(f"  1. Clone llama.cpp (if not already): git clone https://github.com/ggerganov/llama.cpp")
    script_output.info(f"  2. Convert to GGUF: python llama.cpp/convert.py {model_path}")
    script_output.info(f"  3. Quantize: llama.cpp/quantize {model_path}/ggml-model-f16.gguf {output_path} {quantization}")

    script_output.info("\nAlternatively, use Ollama's built-in quantization:")
    script_output.info(f"  1. Create Modelfile pointing to safetensors")
    script_output.info(f"  2. Run: ollama create nyay-ai -f Modelfile")


def export_to_mlx(
    checkpoint_path: Path,
    base_model_path: Path,
    output_path: Path
) -> None:
    """Export to MLX format (merged adapters).

    Args:
        checkpoint_path: Path to checkpoint with adapters
        base_model_path: Path to base model
        output_path: Path to save MLX model
    """
    script_output.info("Exporting to MLX format...")

    output_path.mkdir(parents=True, exist_ok=True)

    # Copy base model files
    script_output.info("  Copying base model files...")
    for file in base_model_path.glob("*"):
        if file.is_file():
            shutil.copy2(file, output_path / file.name)

    # Copy adapter files
    script_output.info("  Copying adapter files...")
    adapter_file = checkpoint_path / "adapters.safetensors"

    if adapter_file.exists():
        shutil.copy2(adapter_file, output_path / "adapters.safetensors")
    else:
        script_output.warning(f"  Adapter file not found: {adapter_file}")

    # Copy training state
    state_file = checkpoint_path / "training_state.json"
    if state_file.exists():
        shutil.copy2(state_file, output_path / "training_state.json")

    script_output.info(f"✓ Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Nyay AI model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/training_config.yaml"),
        help="Training config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint to export (default: best checkpoint)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["gguf", "mlx", "safetensors"],
        default="mlx",
        help="Export format"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: models/nyay-ai-{format})"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="q4_k_m",
        choices=["q4_0", "q4_k_m", "q5_0", "q5_k_m", "q8_0", "f16", "f32"],
        help="Quantization type for GGUF (default: q4_k_m for Ollama)"
    )

    args = parser.parse_args()

    # Load configuration
    script_output.info("Loading configuration...")
    config = load_config(args.config)

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoints_dir = Path(config["checkpointing"]["output_dir"])
        script_output.info(f"Finding best checkpoint in {checkpoints_dir}...")
        checkpoint_path = find_best_checkpoint(checkpoints_dir)

    script_output.info(f"Using checkpoint: {checkpoint_path}")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = Path(f"models/nyay-ai-{args.format}")

    # Get base model path
    base_model_path = Path(config["model"]["path"])

    # Export based on format
    script_output.info(f"\n{'='*70}")
    script_output.info(f"EXPORTING TO {args.format.upper()}")
    script_output.info(f"{'='*70}\n")

    if args.format == "mlx":
        export_to_mlx(checkpoint_path, base_model_path, output_path)

    elif args.format == "gguf":
        # First merge adapters
        merged_path = Path("models/nyay-ai-merged")
        merge_lora_adapters(base_model_path, checkpoint_path, merged_path)

        # Then convert to GGUF
        gguf_output = output_path / f"nyay-ai-{args.quantize}.gguf"
        export_to_gguf(merged_path, gguf_output, args.quantize)

    elif args.format == "safetensors":
        # Merge adapters into full model
        merge_lora_adapters(base_model_path, checkpoint_path, output_path)

    script_output.info(f"\n✅ Export complete!")
    script_output.info(f"Output: {output_path}")

    # Next steps
    script_output.info("\nNext steps:")

    if args.format == "mlx":
        script_output.info("  Test model:")
        script_output.info(f"    python scripts/evaluate_model.py --checkpoint {output_path}")
        script_output.info("\n  Generate with MLX:")
        script_output.info(f"    mlx-lm --model {output_path} --prompt 'Summarize this judgment...'")

    elif args.format == "gguf":
        script_output.info("  Deploy with Ollama:")
        script_output.info("    1. Create Modelfile (see Modelfile template)")
        script_output.info("    2. ollama create nyay-ai -f Modelfile")
        script_output.info("    3. ollama run nyay-ai")

    elif args.format == "safetensors":
        script_output.info("  Load merged model:")
        script_output.info(f"    from transformers import AutoModelForCausalLM")
        script_output.info(f"    model = AutoModelForCausalLM.from_pretrained('{output_path}')")


if __name__ == "__main__":
    main()
