# Phase 3: Model Training - Detailed Specification

## Overview

Fine-tune Llama 3.2 3B on 8,000 Indian legal examples using QLoRA 8-bit on MacBook Pro M2.

| Aspect | Value |
|--------|-------|
| **Base Model** | meta-llama/Llama-3.2-3B-Instruct |
| **Parameters** | 3 billion |
| **Method** | QLoRA (Quantized Low-Rank Adaptation) |
| **Quantization** | 8-bit (best quality/memory trade-off) |
| **Framework** | MLX (Apple Silicon optimized) |
| **Hardware** | MacBook Pro M2, 32GB RAM |
| **Training Data** | 8,000 examples (7,200 train + 800 val) |
| **Estimated Time** | 4-8 hours |

---

## Why QLoRA 8-bit?

### The Memory Problem

Full fine-tuning of a 3B model requires:

| Component | Memory |
|-----------|--------|
| Model weights (FP16) | 6 GB |
| Gradients | 6 GB |
| Optimizer states (Adam) | 12 GB |
| Activations | 4-8 GB |
| **Total** | **28-32 GB** |

This exceeds comfortable memory on a 32GB M2 (shared with OS).

### QLoRA Solution

| Component | Memory |
|-----------|--------|
| Model weights (8-bit quantized) | 3 GB |
| LoRA adapters (FP16) | 16 MB |
| Gradients (LoRA only) | 16 MB |
| Optimizer states (LoRA only) | 32 MB |
| Activations | 4 GB |
| **Total** | **~8 GB** |

### Why 8-bit over 4-bit?

| Aspect | 4-bit | 8-bit (Chosen) |
|--------|-------|----------------|
| Model size | 1.5 GB | 3 GB |
| Total memory | ~6 GB | ~8 GB |
| Quality loss | 2-3% | 0.5-1% |
| Legal accuracy | Good | **Better** |
| M2 compatible | ✅ Yes | ✅ Yes |

**Decision:** 8-bit gives better quality with acceptable memory usage for legal AI where accuracy matters.

---

## Training Method: LoRA Explained

### What is LoRA?

Instead of updating all 3B parameters, LoRA adds small trainable matrices:

```
Original Weight Matrix W (frozen):
┌─────────────────────────────────────┐
│                                     │
│     3B parameters (NOT trained)     │
│                                     │
└─────────────────────────────────────┘

LoRA Adapter (trained):
┌───────┐     ┌───────┐
│   A   │  ×  │   B   │  = Low-rank update
│ d × r │     │ r × d │
└───────┘     └───────┘

Final output = W × input + (A × B) × input
```

### LoRA Parameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `lora_rank` (r) | 16 | Adapter capacity (higher = more learning) |
| `lora_alpha` | 32 | Scaling factor (typically 2× rank) |
| `lora_dropout` | 0.05 | Regularization |
| `target_modules` | q_proj, k_proj, v_proj, o_proj | Which layers to adapt |

### Trainable Parameters

| Model | Total Params | LoRA Params | Percentage |
|-------|--------------|-------------|------------|
| Llama 3.2 3B | 3,000,000,000 | ~8,000,000 | **0.27%** |

We train only 0.27% of parameters while achieving 97-99% of full fine-tune quality.

---

## Framework: MLX

### Why MLX?

| Framework | M2 Optimization | Speed | Ease of Use |
|-----------|-----------------|-------|-------------|
| **MLX** | ✅ Native | ⚡ Fastest | ✅ Simple |
| Unsloth | ❌ CUDA-first | 🔄 Good | ✅ Simple |
| HF + PEFT | ❌ General | 🐢 Slower | ⚠️ Complex |

MLX is Apple's machine learning framework, optimized for Apple Silicon.

### MLX Benefits

- Native Metal GPU acceleration
- Unified memory (no CPU↔GPU transfers)
- Lazy evaluation (memory efficient)
- Simple Python API

---

## Data Format

### Input Format (from Phase 2)

```json
{
  "instruction": "Analyze the outcome of this judgment...",
  "input": "[Full judgment text...]",
  "output": "[Generated analysis...]",
  "metadata": {
    "cnr": "HCBM030212662025",
    "task_type": "outcome_analysis"
  }
}
```

### MLX Chat Format (converted)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Nyay AI, a legal assistant specializing in Indian law. Provide accurate, well-structured legal analysis."
    },
    {
      "role": "user",
      "content": "Analyze the outcome of this judgment:\n\n[Full judgment text...]"
    },
    {
      "role": "assistant",
      "content": "[Generated analysis...]"
    }
  ]
}
```

### Conversion Script

```python
# scripts/convert_to_mlx_format.py

import json
from pathlib import Path

SYSTEM_PROMPT = """You are Nyay AI, a legal assistant specializing in Indian law.
You provide accurate, well-structured legal analysis based on Indian court judgments.
Always cite specific sections, acts, and precedents when relevant."""

def convert_to_mlx_format(input_path: Path, output_path: Path):
    """Convert Phase 2 JSONL to MLX chat format."""

    with open(input_path) as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)

            # Combine instruction and input
            user_content = data['instruction']
            if data.get('input'):
                user_content += f"\n\n{data['input']}"

            # Create chat format
            mlx_example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": data['output']}
                ]
            }

            f_out.write(json.dumps(mlx_example) + '\n')

if __name__ == "__main__":
    convert_to_mlx_format(
        Path("data/training/train.jsonl"),
        Path("data/training/train_mlx.jsonl")
    )
    convert_to_mlx_format(
        Path("data/training/val.jsonl"),
        Path("data/training/val_mlx.jsonl")
    )
    print("Conversion complete!")
```

---

## Training Configuration

### config/training_config.yaml

```yaml
# Model Configuration
model:
  name: "meta-llama/Llama-3.2-3B-Instruct"
  quantization: 8  # 8-bit quantization

# LoRA Configuration
lora:
  rank: 16              # Adapter rank (capacity)
  alpha: 32             # Scaling factor
  dropout: 0.05         # Regularization
  target_modules:       # Layers to adapt
    - q_proj            # Query projection
    - k_proj            # Key projection
    - v_proj            # Value projection
    - o_proj            # Output projection

# Training Configuration
training:
  batch_size: 4                  # Samples per batch
  gradient_accumulation: 4       # Effective batch = 16
  learning_rate: 2.0e-4          # Initial LR
  epochs: 3                      # Training epochs
  warmup_ratio: 0.1              # Warmup steps ratio
  max_seq_length: 2048           # Max sequence length

# Optimizer
optimizer:
  type: "adamw"
  weight_decay: 0.01

# Learning Rate Schedule
scheduler:
  type: "cosine"
  min_lr_ratio: 0.1              # Final LR = 10% of initial

# Checkpointing
checkpointing:
  save_every: 500                # Save every N steps
  eval_every: 250                # Evaluate every N steps
  output_dir: "models/nyay-ai-adapter"

# Logging
logging:
  log_every: 10                  # Log every N steps
  wandb: false                   # Weights & Biases integration
```

### Hyperparameter Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **batch_size** | 4 | Fits in memory with gradient accumulation |
| **gradient_accumulation** | 4 | Effective batch = 16 (good for stability) |
| **learning_rate** | 2e-4 | Standard for LoRA fine-tuning |
| **epochs** | 3 | Enough for 8K examples without overfitting |
| **lora_rank** | 16 | Good balance of capacity and efficiency |
| **max_seq_length** | 2048 | Covers most legal documents |

---

## Training Script

### scripts/train_model.py

```python
#!/usr/bin/env python3
"""
Fine-tune Llama 3.2 3B on Indian legal data using MLX with QLoRA 8-bit.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --config config/training_config.yaml
    python scripts/train_model.py --resume models/nyay-ai-adapter/checkpoint-500
"""

import argparse
import yaml
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.tuner import TrainingArgs, train

def load_config(config_path: str) -> dict:
    """Load training configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train Nyay AI model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training config"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    print("=" * 60)
    print("NYAY AI - Model Training")
    print("=" * 60)
    print(f"Model: {config['model']['name']}")
    print(f"Method: QLoRA {config['model']['quantization']}-bit")
    print(f"LoRA Rank: {config['lora']['rank']}")
    print(f"Epochs: {config['training']['epochs']}")
    print("=" * 60)

    # Load model with quantization
    print("\nLoading model...")
    model, tokenizer = load(
        config['model']['name'],
        tokenizer_config={"trust_remote_code": True}
    )

    # Setup training arguments
    training_args = TrainingArgs(
        batch_size=config['training']['batch_size'],
        iters=config['training']['epochs'] * 7200 // config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        lora_rank=config['lora']['rank'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        adapter_path=config['checkpointing']['output_dir'],
        save_every=config['checkpointing']['save_every'],
        val_batches=100,
    )

    # Start training
    print("\nStarting training...")
    print("This will take 4-8 hours. Progress will be logged below.\n")

    train(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=Path("data/training/train_mlx.jsonl"),
        val_dataset=Path("data/training/val_mlx.jsonl"),
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Adapter saved to: {config['checkpointing']['output_dir']}")
    print("\nNext steps:")
    print("1. Evaluate: python scripts/evaluate_model.py")
    print("2. Export: python scripts/export_model.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## Evaluation

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Training Loss** | How well model fits training data | < 1.0 |
| **Validation Loss** | Generalization to unseen data | < 1.2 |
| **ROUGE-L** | Summary overlap with reference | > 0.4 |
| **Legal Accuracy** | Correct legal interpretations | > 75% |
| **Hallucination Rate** | Made-up facts/citations | < 5% |
| **Human Approval** | Manual quality assessment | > 70% |

### Evaluation Script

```python
# scripts/evaluate_model.py

"""
Evaluate fine-tuned Nyay AI model on test cases.
"""

import json
from pathlib import Path
from mlx_lm import load, generate

def evaluate_model(adapter_path: str, test_file: str):
    """Run evaluation on test cases."""

    # Load model with adapter
    model, tokenizer = load(
        "meta-llama/Llama-3.2-3B-Instruct",
        adapter_path=adapter_path
    )

    # Load test cases
    with open(test_file) as f:
        test_cases = [json.loads(line) for line in f]

    results = {
        "total": len(test_cases),
        "by_task": {},
        "samples": []
    }

    for i, test in enumerate(test_cases[:50]):  # Evaluate 50 samples
        # Generate response
        prompt = format_prompt(test)
        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=500,
            temp=0.7
        )

        # Store sample
        results["samples"].append({
            "input": test["messages"][1]["content"][:200],
            "expected": test["messages"][2]["content"][:200],
            "generated": response[:200]
        })

        print(f"Evaluated {i+1}/{min(50, len(test_cases))}")

    # Save results
    with open("evaluation/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to evaluation/results.json")
    print("Run manual review to assess quality.")

if __name__ == "__main__":
    evaluate_model(
        adapter_path="models/nyay-ai-adapter",
        test_file="data/training/val_mlx.jsonl"
    )
```

### Manual Evaluation Checklist

For each generated output, check:

| Check | Question | Pass Criteria |
|-------|----------|---------------|
| **Accuracy** | Are facts correct? | All facts match source |
| **Completeness** | Are key points covered? | No major omissions |
| **Legal Soundness** | Is legal reasoning correct? | Correct interpretation |
| **No Hallucination** | Are all citations real? | No made-up references |
| **Clarity** | Is output well-structured? | Easy to understand |

---

## Export & Deployment

### Export to GGUF

```python
# scripts/export_model.py

"""
Export fine-tuned model to GGUF format for deployment.
"""

def export_to_gguf(adapter_path: str, output_path: str, quantize: str = "q4_k_m"):
    """
    Export model to GGUF format.

    Quantization options:
    - q4_k_m: 4-bit, good balance (recommended)
    - q5_k_m: 5-bit, better quality
    - q8_0: 8-bit, best quality
    """

    # Step 1: Merge adapter with base model
    print("Merging adapter with base model...")
    # mlx_lm.fuse --model meta-llama/Llama-3.2-3B-Instruct --adapter-path {adapter_path}

    # Step 2: Convert to GGUF
    print(f"Converting to GGUF with {quantize} quantization...")
    # python convert.py --outtype {quantize}

    print(f"Model exported to: {output_path}")
```

### Deployment Options

#### Option 1: Ollama (Recommended)

```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./models/nyay-ai.gguf

SYSTEM """You are Nyay AI, a legal assistant specializing in Indian law.
You provide accurate, well-structured legal analysis based on Indian court judgments.
Always cite specific sections, acts, and precedents when relevant."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
EOF

# Create and run
ollama create nyay-ai -f Modelfile
ollama run nyay-ai
```

#### Option 2: Python API

```python
from mlx_lm import load, generate

# Load model
model, tokenizer = load("models/nyay-ai-merged")

# Generate
response = generate(
    model, tokenizer,
    prompt="Summarize this judgment: ...",
    max_tokens=500
)
print(response)
```

#### Option 3: REST API (llama.cpp)

```bash
# Start server
./llama-server -m models/nyay-ai.gguf -c 4096 --port 8080

# Query
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Summarize this judgment: ..."}'
```

---

## Dependencies

### requirements-training.txt

```txt
# MLX (Apple Silicon native)
mlx>=0.10.0
mlx-lm>=0.10.0

# Hugging Face (for model download)
transformers>=4.36.0
huggingface-hub>=0.20.0

# Data processing
datasets>=2.16.0
pyyaml>=6.0

# Evaluation
rouge-score>=0.1.2
nltk>=3.8.0

# Utilities
tqdm>=4.66.0
rich>=13.0.0

# Export (optional)
llama-cpp-python>=0.2.0
```

---

## Directory Structure (Phase 3)

```
nyay-ai-india/
├── config/
│   └── training_config.yaml       # Training hyperparameters
├── scripts/
│   ├── convert_to_mlx_format.py   # Data conversion
│   ├── train_model.py             # Training script
│   ├── evaluate_model.py          # Evaluation
│   └── export_model.py            # GGUF export
├── models/                        # (gitignored)
│   ├── nyay-ai-adapter/           # LoRA weights
│   │   ├── adapter_config.json
│   │   └── adapters.safetensors
│   ├── nyay-ai-merged/            # Merged model
│   └── nyay-ai.gguf               # Quantized for deployment
├── evaluation/
│   ├── test_cases.json            # Manual test cases
│   └── results/                   # Evaluation results
├── requirements-training.txt      # Training dependencies
└── Modelfile                      # Ollama config
```

---

## Timeline

| Step | Task | Duration |
|------|------|----------|
| 1 | Setup environment | 30 min |
| 2 | Convert data format | 15 min |
| 3 | Download base model | 30-60 min |
| 4 | Training (3 epochs) | 4-8 hours |
| 5 | Evaluation | 1 hour |
| 6 | Export & quantize | 30 min |
| 7 | Ollama deployment | 15 min |
| **Total** | | **7-11 hours** |

---

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training loss | < 1.0 | Training logs |
| Validation loss | < 1.2 | Training logs |
| Summarization quality | > 70% approval | Human review |
| Q&A accuracy | > 75% correct | Manual check |
| Hallucination rate | < 5% | Manual check |
| Inference speed | > 20 tok/s | Benchmark |
| Model size (GGUF) | < 3 GB | File size |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch_size to 2, increase gradient_accumulation |
| Slow training | Check MLX is using GPU: `mx.metal.is_available()` |
| High loss | Reduce learning_rate, increase epochs |
| Overfitting | Reduce epochs, increase lora_dropout |
| Model download fails | Use `huggingface-cli login` first |

### Monitoring Training

```bash
# Watch GPU usage
sudo powermetrics --samplers gpu_power -i 1000

# Check memory
top -l 1 | grep PhysMem
```

---

## Next Steps After Phase 3

1. **Deploy locally** with Ollama
2. **Create web interface** (Gradio/Streamlit)
3. **Add RAG** for document retrieval
4. **Expand training data** to more courts
5. **Fine-tune on specific tasks** (e.g., contract analysis)

---

## References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM Fine-tuning Guide](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Llama 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
