# Nyay AI - GGUF Export Guide

## Quick Start (Recommended)

### Option 1: Automated Bash Script (Easiest)

```bash
# Convert with default settings (Q4_K_M quantization)
bash scripts/convert_mlx_to_gguf.sh

# Or specify options
bash scripts/convert_mlx_to_gguf.sh \
  --checkpoint models/nyay-ai-checkpoints-v4/0002500_adapters.safetensors \
  --quantization Q5_K_M
```

**Time**: ~20 minutes
**Output**: `models/nyay-ai-gguf/nyay-ai-q4_k_m.gguf` (~2 GB)

---

## Process Overview

```
┌─────────────────┐
│  MLX LoRA Model │
│  (46 MB)        │
└────────┬────────┘
         │
         ├─ Step 1: Fuse LoRA with base (3-5 min)
         │
         ▼
┌─────────────────┐
│  Fused MLX Model│
│  (6 GB)         │
└────────┬────────┘
         │
         ├─ Step 2: Convert to GGUF F16 (5-10 min)
         │
         ▼
┌─────────────────┐
│  GGUF F16       │
│  (6 GB)         │
└────────┬────────┘
         │
         ├─ Step 3: Quantize (5-10 min)
         │
         ▼
┌─────────────────┐
│  GGUF Q4_K_M    │
│  (2 GB)         │ ← Use this for Ollama
└─────────────────┘
```

---

## Detailed Steps

### Prerequisites

```bash
# 1. Ensure you're in the training venv
source .venv-train/bin/activate

# 2. Install mlx-lm if not already installed
pip install mlx-lm

# 3. Install llama.cpp (automatic in script, or manual):
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make && cd ..
```

### Step 1: Fuse LoRA Adapters

```bash
mlx_lm.fuse \
  --model models/llama-3.2-3b-instruct-mlx \
  --adapter-path models/nyay-ai-checkpoints-v4 \
  --save-path models/nyay-ai-gguf/fused-model \
  --de-quantize
```

**Time**: 3-5 minutes
**Output**: Fused model (~6 GB)

### Step 2: Convert to GGUF (F16)

```bash
python llama.cpp/convert.py models/nyay-ai-gguf/fused-model \
  --outtype f16 \
  --outfile models/nyay-ai-gguf/nyay-ai-f16.gguf
```

**Time**: 5-10 minutes
**Output**: F16 GGUF (~6 GB)

### Step 3: Quantize

```bash
llama.cpp/quantize \
  models/nyay-ai-gguf/nyay-ai-f16.gguf \
  models/nyay-ai-gguf/nyay-ai-q4_k_m.gguf \
  Q4_K_M
```

**Time**: 5-10 minutes
**Output**: Quantized GGUF (~2 GB)

---

## Quantization Options

| Type | Size | Quality | Use Case |
|------|------|---------|----------|
| **Q4_K_M** | ~2.0 GB | Good | **Recommended** - Best balance |
| Q5_K_M | ~2.5 GB | Better | Higher quality, slower |
| Q8_0 | ~3.5 GB | Best | Maximum quality |
| F16 | ~6.0 GB | Perfect | No compression (slow) |

---

## Deploy with Ollama

### Step 1: Create Modelfile

The script automatically creates `models/nyay-ai-gguf/Modelfile`:

```dockerfile
FROM models/nyay-ai-gguf/nyay-ai-q4_k_m.gguf

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_ctx 2048

SYSTEM """You are Nyay AI, a legal research assistant specializing in Indian law..."""
```

### Step 2: Create Ollama Model

```bash
ollama create nyay-ai -f models/nyay-ai-gguf/Modelfile
```

**Output**:
```
transferring model data
creating model blob
writing manifest
success
```

### Step 3: Test the Model

```bash
ollama run nyay-ai "What is a writ of habeas corpus?"
```

### Step 4: List Models

```bash
ollama list
```

Expected output:
```
NAME       ID              SIZE      MODIFIED
nyay-ai    abc123...       2.0 GB    2 minutes ago
```

---

## API Deployment

### Start Ollama Server

```bash
ollama serve
```

**Runs at**: `http://localhost:11434`

### Test API

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "nyay-ai",
  "prompt": "What is Article 226?",
  "stream": false
}'
```

### Python Client

```python
import requests

response = requests.post('http://localhost:11434/api/generate', json={
    'model': 'nyay-ai',
    'prompt': 'What is a Public Interest Litigation?',
    'stream': False
})

print(response.json()['response'])
```

---

## Troubleshooting

### Issue: "mlx_lm.fuse not found"

**Solution**:
```bash
pip install --upgrade mlx-lm
```

### Issue: "llama.cpp/convert.py not found"

**Solution**:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make && cd ..
```

### Issue: "Out of memory during conversion"

**Solution**:
- Close other applications
- Use smaller quantization (Q4_K_M instead of Q8_0)
- Conversion uses ~10-12 GB RAM peak

### Issue: "Ollama model not responding properly"

**Solution**:
- Check Modelfile system prompt
- Test with `ollama run nyay-ai --verbose`
- Review logs: `ollama logs`

---

## File Locations

After successful export:

```
models/nyay-ai-gguf/
├── fused-model/              # Fused MLX model (6 GB)
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── nyay-ai-f16.gguf          # F16 GGUF (6 GB)
├── nyay-ai-q4_k_m.gguf       # Quantized GGUF (2 GB) ← Use this
└── Modelfile                 # Ollama configuration
```

---

## Performance Expectations

### Inference Speed (M2 MacBook Pro)

| Quantization | Tokens/sec | RAM Usage |
|--------------|------------|-----------|
| Q4_K_M | ~20-25 | ~3 GB |
| Q5_K_M | ~18-22 | ~4 GB |
| Q8_0 | ~15-18 | ~5 GB |

### Quality Comparison

- **Q4_K_M**: 97-98% of original quality (recommended)
- **Q5_K_M**: 98-99% of original quality
- **Q8_0**: 99%+ of original quality

---

## Next Steps After Deployment

1. **Test with real queries** from your test cases
2. **Implement response filter** for short/"best answer" responses
3. **Collect user feedback** on quality
4. **Monitor performance** (speed, accuracy)
5. **Plan Phase 4** improvements based on feedback

---

## Time Estimate

| Step | Time |
|------|------|
| Fuse LoRA | 3-5 min |
| Convert to F16 | 5-10 min |
| Quantize | 5-10 min |
| Create Ollama model | 1 min |
| **Total** | **15-25 min** |

---

## Support

If you encounter issues:
1. Check logs in `models/nyay-ai-gguf/`
2. Verify file sizes match expected values
3. Test with `ollama run nyay-ai --verbose`
4. Review Ollama docs: https://ollama.ai/

---

**Ready to convert?** Run:

```bash
bash scripts/convert_mlx_to_gguf.sh
```
