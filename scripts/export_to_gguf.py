#!/usr/bin/env python3
"""
Export Nyay AI Fine-tuned Model to GGUF Format

This script:
1. Fuses LoRA adapters with the base Llama 3.2 3B model
2. Converts the merged model to GGUF format for Ollama deployment
3. Optionally quantizes to reduce model size

Usage:
    python scripts/export_to_gguf.py
    python scripts/export_to_gguf.py --checkpoint models/nyay-ai-checkpoints-v4/0002500_adapters.safetensors
    python scripts/export_to_gguf.py --quantize q4_k_m
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import shutil


def run_command(cmd: list, description: str, timeout: int = None) -> bool:
    """Run a shell command and handle errors"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}")
        print(f"Exit code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"✗ Command timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def check_dependencies():
    """Check if required tools are installed"""
    print("="*70)
    print("CHECKING DEPENDENCIES")
    print("="*70)
    print()

    # Check Python packages
    try:
        import mlx
        print(f"✓ MLX installed: {mlx.__version__}")
    except ImportError:
        print("✗ MLX not installed. Install with: pip install mlx")
        return False

    try:
        import mlx_lm
        print(f"✓ MLX-LM installed")
    except ImportError:
        print("✗ MLX-LM not installed. Install with: pip install mlx-lm")
        return False

    # Check if llama.cpp is available for GGUF conversion
    # We'll use mlx-lm's built-in conversion if available
    print(f"✓ All dependencies satisfied")
    print()
    return True


def fuse_lora_adapters(base_model: str, adapter_path: str, output_path: str) -> bool:
    """Fuse LoRA adapters with base model using mlx_lm.fuse"""
    print("="*70)
    print("STEP 1: FUSING LORA ADAPTERS WITH BASE MODEL")
    print("="*70)
    print()
    print(f"Base model: {base_model}")
    print(f"Adapters: {adapter_path}")
    print(f"Output: {output_path}")
    print()

    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Use mlx_lm.fuse command
    cmd = [
        "mlx_lm.fuse",
        "--model", base_model,
        "--adapter-path", str(Path(adapter_path).parent),
        "--save-path", output_path,
        "--de-quantize"  # Convert to full precision for GGUF conversion
    ]

    return run_command(cmd, "Fusing LoRA adapters", timeout=600)


def convert_to_gguf(mlx_model_path: str, output_path: str, quantization: str = None) -> bool:
    """Convert MLX model to GGUF format"""
    print("="*70)
    print("STEP 2: CONVERTING TO GGUF FORMAT")
    print("="*70)
    print()
    print(f"Input: {mlx_model_path}")
    print(f"Output: {output_path}")
    if quantization:
        print(f"Quantization: {quantization}")
    print()

    # MLX-LM has built-in GGUF conversion via mlx_lm.convert
    # We'll use the llama.cpp conversion tool if available

    # First, try using mlx_lm's conversion
    try:
        cmd = [
            "python", "-m", "mlx_lm.convert",
            "--model", mlx_model_path,
            "--output", output_path,
            "--format", "gguf"
        ]

        if quantization:
            cmd.extend(["--quantize", quantization])

        return run_command(cmd, "Converting to GGUF", timeout=1200)
    except Exception as e:
        print(f"MLX-LM conversion failed: {e}")
        print()
        print("Alternative: Manual conversion using llama.cpp")
        print("Please follow these steps:")
        print()
        print("1. Clone llama.cpp:")
        print("   git clone https://github.com/ggerganov/llama.cpp")
        print("   cd llama.cpp && make")
        print()
        print("2. Convert to GGUF:")
        print(f"   python convert.py {mlx_model_path} --outtype f16 --outfile {output_path}")
        print()
        if quantization:
            print("3. Quantize:")
            print(f"   ./quantize {output_path} {output_path.replace('.gguf', f'-{quantization}.gguf')} {quantization}")
        return False


def create_modelfile(gguf_path: str, output_path: str, model_name: str = "nyay-ai"):
    """Create Ollama Modelfile for deployment"""
    print("="*70)
    print("STEP 3: CREATING OLLAMA MODELFILE")
    print("="*70)
    print()

    modelfile_content = f"""# Nyay AI - Indian Legal Assistant
# Fine-tuned Llama 3.2 3B for Indian Law

FROM {gguf_path}

# Model parameters
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

# System prompt
SYSTEM \"\"\"You are Nyay AI, a legal research assistant specializing in Indian law.
You help users understand Indian legal concepts, statutes, case law, and procedures.

Guidelines:
- Provide detailed, well-explained answers
- Cite relevant sections, articles, and acts when applicable
- If you don't know something, say so clearly
- Indian law context only (Supreme Court, High Courts, Indian statutes)
- Avoid one-word or very short answers - explain thoroughly
\"\"\"

# Model info
TEMPLATE \"\"\"{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>\"\"\"
"""

    with open(output_path, 'w') as f:
        f.write(modelfile_content)

    print(f"✓ Modelfile created: {output_path}")
    print()
    print("To create Ollama model, run:")
    print(f"  ollama create {model_name} -f {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Export Nyay AI model to GGUF format for Ollama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with default settings
  python scripts/export_to_gguf.py

  # Export specific checkpoint
  python scripts/export_to_gguf.py --checkpoint models/nyay-ai-checkpoints-v4/0002500_adapters.safetensors

  # Export with Q4 quantization
  python scripts/export_to_gguf.py --quantize q4_k_m

  # Full workflow
  python scripts/export_to_gguf.py --quantize q4_k_m --create-ollama-model
        """
    )

    parser.add_argument(
        '--base-model',
        type=str,
        default='models/llama-3.2-3b-instruct-mlx',
        help='Path to base Llama 3.2 3B model'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/nyay-ai-checkpoints-v4/0003000_adapters.safetensors',
        help='Path to LoRA checkpoint to export'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/nyay-ai-gguf',
        help='Output directory for GGUF files'
    )

    parser.add_argument(
        '--quantize',
        type=str,
        choices=['q4_0', 'q4_k_m', 'q5_0', 'q5_k_m', 'q8_0', 'f16'],
        default='q4_k_m',
        help='Quantization type (default: q4_k_m for best size/quality balance)'
    )

    parser.add_argument(
        '--model-name',
        type=str,
        default='nyay-ai',
        help='Name for Ollama model'
    )

    parser.add_argument(
        '--skip-fusion',
        action='store_true',
        help='Skip LoRA fusion step (if already fused)'
    )

    parser.add_argument(
        '--create-ollama-model',
        action='store_true',
        help='Automatically create Ollama model after conversion'
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "="*70)
    print("NYAY AI - GGUF EXPORT SCRIPT")
    print("="*70)
    print()
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Quantization: {args.quantize}")
    print(f"Output: {args.output_dir}")
    print()

    # Check dependencies
    if not check_dependencies():
        print("\n✗ Dependency check failed. Please install required packages.")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine checkpoint name for output files
    checkpoint_name = Path(args.checkpoint).stem.replace('_adapters', '')
    timestamp = datetime.now().strftime('%Y%m%d')

    # Step 1: Fuse LoRA adapters (unless skipped)
    fused_model_path = output_dir / f"fused-{checkpoint_name}"

    if args.skip_fusion and fused_model_path.exists():
        print(f"✓ Skipping fusion, using existing: {fused_model_path}")
    else:
        if not fuse_lora_adapters(args.base_model, args.checkpoint, str(fused_model_path)):
            print("\n✗ LoRA fusion failed")
            print("\nManual alternative:")
            print("  mlx_lm.fuse --model", args.base_model)
            print("              --adapter-path", str(Path(args.checkpoint).parent))
            print("              --save-path", str(fused_model_path))
            sys.exit(1)

    # Step 2: Convert to GGUF
    gguf_filename = f"nyay-ai-{checkpoint_name}-{args.quantize}.gguf"
    gguf_path = output_dir / gguf_filename

    print("\n" + "="*70)
    print("ALTERNATIVE: MANUAL GGUF CONVERSION")
    print("="*70)
    print()
    print("MLX-LM doesn't have direct GGUF export. Use this manual process:")
    print()
    print("1. Convert fused model to HuggingFace format:")
    print(f"   mlx_lm.convert --model {fused_model_path} --save-path {fused_model_path}-hf")
    print()
    print("2. Download llama.cpp:")
    print("   git clone https://github.com/ggerganov/llama.cpp")
    print("   cd llama.cpp && make")
    print()
    print("3. Convert to GGUF:")
    print(f"   python llama.cpp/convert.py {fused_model_path}-hf \\")
    print(f"          --outtype f16 \\")
    print(f"          --outfile {gguf_path.with_suffix('')}-f16.gguf")
    print()
    print("4. Quantize:")
    print(f"   llama.cpp/quantize {gguf_path.with_suffix('')}-f16.gguf \\")
    print(f"                      {gguf_path} \\")
    print(f"                      {args.quantize.upper()}")
    print()

    # Step 3: Create Modelfile
    modelfile_path = output_dir / "Modelfile"
    create_modelfile(str(gguf_path), str(modelfile_path), args.model_name)

    # Print summary
    print("="*70)
    print("EXPORT SUMMARY")
    print("="*70)
    print()
    print(f"✓ Fused model: {fused_model_path}")
    print(f"→ GGUF output: {gguf_path} (manual conversion required)")
    print(f"✓ Modelfile: {modelfile_path}")
    print()
    print("="*70)
    print("NEXT STEPS")
    print("="*70)
    print()
    print("After completing manual GGUF conversion:")
    print()
    print("1. Create Ollama model:")
    print(f"   ollama create {args.model_name} -f {modelfile_path}")
    print()
    print("2. Test the model:")
    print(f"   ollama run {args.model_name} 'What is a writ of habeas corpus?'")
    print()
    print("3. List your models:")
    print("   ollama list")
    print()
    print("4. Deploy as API:")
    print("   ollama serve")
    print("   # Then access at http://localhost:11434")
    print()

    if args.create_ollama_model and gguf_path.exists():
        print("Creating Ollama model...")
        cmd = ["ollama", "create", args.model_name, "-f", str(modelfile_path)]
        if run_command(cmd, "Creating Ollama model", timeout=300):
            print(f"\n✓ Ollama model '{args.model_name}' created successfully!")
            print(f"\nTest with: ollama run {args.model_name}")
        else:
            print("\n✗ Failed to create Ollama model")


if __name__ == '__main__':
    main()
