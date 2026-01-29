# Root Cause Analysis: MLX LoRA Training Failure

**Date:** 2026-01-29
**Author:** Claude Code
**Status:** Resolved
**Severity:** Critical (Training produced unusable model)

---

## Executive Summary

The MLX LoRA fine-tuning of Llama 3.2 3B completed successfully (3000 iterations, ~6.75 hours) but produced a model that only generated commas (`,,,,,,,,,`) instead of coherent text. The root cause was a **learning rate 10x too high**, causing gradient explosion and model collapse during training.

---

## 1. Problem Statement

### Symptoms
- Training completed without errors after 3000 iterations
- Final training loss reported as 9.976 (appeared normal at the time)
- Final validation loss reported as 9.997
- When loading the fine-tuned model with adapters, all outputs were only commas:
  ```
  Prompt: "What is 2 + 2?"
  Response: ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"
  ```

### Impact
- ~6.75 hours of training time wasted
- Model completely unusable for inference
- All checkpoints (1000, 2000, 3000 iterations) exhibited the same behavior

### Verification
```python
# Base model without adapters - WORKS
model, tokenizer = load("./models/llama-3.2-3b-instruct-mlx")
generate(model, tokenizer, prompt="What is 2 + 2?")
# Output: "2 + 2 is 4..."

# Model with adapters - BROKEN
model, tokenizer = load(
    "./models/llama-3.2-3b-instruct-mlx",
    adapter_path="./models/nyay-ai-checkpoints"
)
generate(model, tokenizer, prompt="What is 2 + 2?")
# Output: ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"
```

---

## 2. Original Implementation

### Configuration Used (`config/mlx_lora_config.yaml`)
```yaml
model: "./models/llama-3.2-3b-instruct-mlx"
data: "./data/training"
fine_tune_type: "lora"
num_layers: -1

# Training hyperparameters
batch_size: 1
iters: 3000
learning_rate: 0.0002  # <-- THE PROBLEM: 2e-4 is too high
grad_accumulation_steps: 8

# Other settings
val_batches: 100
steps_per_eval: 500
max_seq_length: 2048
optimizer: "adamw"
seed: 42
```

### LoRA Parameters (from adapter_config.json)
```json
{
    "lora_parameters": {
        "rank": 8,
        "dropout": 0.0,
        "scale": 20.0
    }
}
```

### Training Command
```bash
mlx_lm.lora --config config/mlx_lora_config.yaml --train
```

### What Appeared to Work
- Model loaded successfully
- Training progressed through all 3000 iterations
- Checkpoints were saved at 1000, 2000, 3000 iterations
- No error messages during training
- Evaluation script reported perplexity of 7.66 (which seemed reasonable)

---

## 3. Debugging Process

### Step 1: Verify Data Format
**Hypothesis:** Training data format might not match what MLX-LM expects.

**Investigation:**
```python
# Checked training data structure
with open("data/training/train_mlx.jsonl") as f:
    example = json.loads(f.readline())

# Result: Correct format
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "metadata": {...}
}
```

**Conclusion:** Data format was correct. MLX-LM's `ChatDataset` properly detected and processed the `messages` key.

### Step 2: Verify Tokenization
**Hypothesis:** Chat template might not be applied correctly during training.

**Investigation:**
```python
# Verified tokenizer applies chat template correctly
tokens = tokenizer.apply_chat_template(messages, return_dict=False)
decoded = tokenizer.decode(tokens[:200])
# Output: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>..."
```

**Conclusion:** Tokenization was correct. The Llama 3 chat template was being applied properly.

### Step 3: Check LoRA Weight Structure
**Hypothesis:** LoRA weights might not be loading correctly.

**Investigation:**
```python
# Verified LoRA weights exist and have correct shapes
q_proj = adapter_model.model.layers[0].self_attn.q_proj
print(f"LoRA A shape: {q_proj.lora_a.shape}")  # (3072, 8)
print(f"LoRA B shape: {q_proj.lora_b.shape}")  # (8, 3072)
print(f"LoRA scale: {q_proj.scale}")           # 20.0
```

**Conclusion:** LoRA weights were present and correctly structured.

### Step 4: Test All Checkpoints
**Hypothesis:** Maybe only the final checkpoint was corrupted.

**Investigation:**
```python
# Tested checkpoints at 1000, 2000, 3000 iterations
for checkpoint in ["0001000_adapters.safetensors", "0002000_adapters.safetensors", "0003000_adapters.safetensors"]:
    # Load and test each
    response = generate(model, tokenizer, prompt="What is 2 + 2?")
    print(f"{checkpoint}: '{response[:30]}...'")

# Results:
# 1000 iters: ',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,...'
# 2000 iters: ',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,...'
# 3000 iters: ',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,...'
```

**Conclusion:** All checkpoints were broken. The problem occurred early in training.

### Step 5: Run Mini Training Test (THE BREAKTHROUGH)
**Hypothesis:** Something is wrong with the training process itself.

**Investigation:**
```bash
# Ran 20 iterations with verbose output
mlx_lm.lora \
    --model ./models/llama-3.2-3b-instruct-mlx \
    --data ./data/training \
    --train \
    --iters 20 \
    --learning-rate 0.0002 \
    --steps-per-report 5
```

**Results:**
```
Iter 1:  Val loss 1.858
Iter 5:  Train loss 2.148  <-- Starting point
Iter 10: Train loss 2.609  <-- INCREASING (should decrease!)
Iter 15: Train loss 4.104  <-- RAPIDLY INCREASING
Iter 20: Train loss 10.965 <-- EXPLODED
        Val loss 10.821
```

**Model output after just 20 iterations:**
```
Response: 'hehe1111111111111111111hehehehehehehehehehehehehehehe...'
```

**BREAKTHROUGH:** The training loss was **exploding**, not decreasing. This is the classic signature of a learning rate that is too high.

### Step 6: Test with Lower Learning Rate
**Verification:**
```bash
# Same setup but with learning_rate = 0.00002 (10x lower)
mlx_lm.lora \
    --model ./models/llama-3.2-3b-instruct-mlx \
    --data ./data/training \
    --train \
    --iters 50 \
    --learning-rate 0.00002 \
    --steps-per-report 10
```

**Results:**
```
Iter 1:  Val loss 1.858
Iter 10: Train loss 2.036  <-- Starting point
Iter 20: Train loss 1.797  <-- DECREASING (correct!)
Iter 30: Train loss 1.687  <-- DECREASING
Iter 40: Train loss 1.637  <-- DECREASING
Iter 50: Train loss 1.537  <-- DECREASING
        Val loss 1.648
```

**Model output:**
```
Prompt: "What is Indian law about property acquisition?"
Response: "This is a question that has been debated by scholars and judges
for a long time. The answer is not straightforward, and it depends on
various factors such as the nature of the property, the purpose of..."
```

**CONFIRMED:** The fix works. Lower learning rate produces a functioning model.

---

## 4. Root Cause

### Primary Cause: Learning Rate Too High

| Parameter | Original Value | Correct Value | Factor |
|-----------|---------------|---------------|--------|
| learning_rate | 0.0002 (2e-4) | 0.00002 (2e-5) | 10x too high |

### Why This Caused the Problem

1. **Gradient Explosion:** With a learning rate of 2e-4, the weight updates were too large, causing the loss to increase instead of decrease.

2. **Model Collapse:** As training progressed, the increasingly large weight updates corrupted the model's learned representations.

3. **Convergence to Degenerate Solution:** The model found a "local minimum" where outputting a single repeated token (comma) minimized the corrupted loss function.

4. **Why Commas Specifically:** Token ID 11 (comma) is a very common token in legal documents. The corrupted model likely collapsed to always predicting the most frequent token as a degenerate solution.

### Contributing Factors

1. **No Early Warning:** The training script didn't warn about increasing loss or provide early stopping.

2. **Misleading Final Loss:** The final loss of ~10 seemed plausible without context of the training progression.

3. **Default Hyperparameters:** The learning rate of 2e-4 may work for some models/datasets but was inappropriate for this specific fine-tuning task.

### Typical Learning Rates for LoRA Fine-Tuning

| Model Size | Recommended LR | Our Original | Status |
|------------|---------------|--------------|--------|
| 1-3B params | 1e-5 to 5e-5 | 2e-4 | 4-20x too high |
| 7B params | 5e-6 to 2e-5 | - | - |
| 13B+ params | 1e-6 to 1e-5 | - | - |

---

## 5. Resolution

### Configuration Fix

**Before (`config/mlx_lora_config.yaml`):**
```yaml
learning_rate: 0.0002
adapter_path: "./models/nyay-ai-checkpoints"
save_every: 1000
```

**After:**
```yaml
learning_rate: 0.00002  # FIXED: was 0.0002 which caused gradient explosion
adapter_path: "./models/nyay-ai-checkpoints-v2"  # New path for fixed training
save_every: 500  # Save more frequently for debugging
```

### Verification Test Results

| Metric | Broken (LR=2e-4) | Fixed (LR=2e-5) |
|--------|-----------------|-----------------|
| Loss at iter 10 | 2.609 (↑) | 2.036 |
| Loss at iter 20 | 10.965 (💥) | 1.797 (↓) |
| Loss at iter 50 | N/A | 1.537 (↓) |
| Model output | `,,,,,,,,` | Coherent text |

---

## 6. Lessons Learned

### Immediate Actions
1. **Always monitor training loss progression** - Loss should decrease, not increase
2. **Run short test training first** - 20-50 iterations can reveal major issues
3. **Test model outputs during training** - Don't wait until the end to check

### Process Improvements
1. **Add loss trend monitoring** - Alert if loss increases for N consecutive steps
2. **Add early stopping** - Stop training if validation loss doesn't improve
3. **Log training curves** - Save loss values for post-hoc analysis
4. **Use conservative learning rates** - Start low (1e-5) and increase if needed

### Hyperparameter Guidelines for Future Training
```yaml
# Conservative starting point for Llama 3.2 3B LoRA fine-tuning
learning_rate: 0.00002      # 2e-5 (safe default)
# learning_rate: 0.00005    # 5e-5 (aggressive, monitor closely)

# If loss increases:
# - Reduce learning rate by 2-5x
# - Check for data issues
# - Verify tokenization matches inference
```

---

## 7. Timeline

| Time | Event |
|------|-------|
| Day 1, 13:37 | Started training with LR=2e-4 |
| Day 1, 20:22 | Training completed (3000 iters, ~6.75 hours) |
| Day 1, 20:30 | Noticed model outputs only commas |
| Day 1, 20:45 | Verified all checkpoints are broken |
| Day 2, 22:28 | Attempted HuggingFace retraining (too slow on MPS) |
| Day 2, 23:00 | Started debugging MLX training |
| Day 2, 23:15 | Discovered loss explosion in mini-test |
| Day 2, 23:20 | Confirmed fix with LR=2e-5 |
| Day 2, 23:30 | Updated config, ready for retraining |

---

## 8. References

- [MLX-LM Documentation](https://github.com/ml-explore/mlx-lm)
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation of Large Language Models
- [Learning Rate Best Practices](https://huggingface.co/docs/transformers/training#learning-rate)
- MLX-LM LoRA config: `config/mlx_lora_config.yaml`
- Broken checkpoints: `models/nyay-ai-checkpoints/` (kept for reference)
- Fixed checkpoints: `models/nyay-ai-checkpoints-v2/` (to be created)

---

## Appendix A: Diagnostic Commands

```bash
# Quick test to verify model works
source .venv-train/bin/activate
python3 << 'EOF'
from mlx_lm import load, generate

model, tokenizer = load(
    "./models/llama-3.2-3b-instruct-mlx",
    adapter_path="./models/nyay-ai-checkpoints-v2"
)
response = generate(model, tokenizer, prompt="What is 2 + 2?", max_tokens=50)
print(f"Response: {response}")
EOF

# Mini training test (always run before full training)
mlx_lm.lora \
    --model ./models/llama-3.2-3b-instruct-mlx \
    --data ./data/training \
    --train \
    --iters 50 \
    --learning-rate 0.00002 \
    --steps-per-report 10

# Check if loss is decreasing (should see values going DOWN)
```

---

## Appendix B: Loss Comparison

### Broken Training (LR = 2e-4)
```
Iter 5:  Train loss 2.148
Iter 10: Train loss 2.609  ↑ +0.46
Iter 15: Train loss 4.104  ↑ +1.50
Iter 20: Train loss 10.965 ↑ +6.86  💥 EXPLODED
```

### Fixed Training (LR = 2e-5)
```
Iter 10: Train loss 2.036
Iter 20: Train loss 1.797  ↓ -0.24
Iter 30: Train loss 1.687  ↓ -0.11
Iter 40: Train loss 1.637  ↓ -0.05
Iter 50: Train loss 1.537  ↓ -0.10  ✓ HEALTHY
```
