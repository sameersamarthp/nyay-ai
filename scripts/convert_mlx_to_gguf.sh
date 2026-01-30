#!/bin/bash
# Simplified GGUF conversion script for Nyay AI
# Uses llama.cpp for MLX → GGUF conversion

set -e

echo "════════════════════════════════════════════════════════════════════════"
echo "NYAY AI - MLX TO GGUF CONVERSION (via llama.cpp)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Configuration
BASE_MODEL="models/llama-3.2-3b-instruct-mlx"
CHECKPOINT="models/nyay-ai-checkpoints-v4/0003000_adapters.safetensors"
OUTPUT_DIR="models/nyay-ai-gguf"
QUANTIZATION="Q4_K_M"  # Q4_K_M, Q5_K_M, or Q8_0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Base model: $BASE_MODEL"
echo "  Checkpoint: $CHECKPOINT"
echo "  Quantization: $QUANTIZATION"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Fuse LoRA adapters
echo "════════════════════════════════════════════════════════════════════════"
echo "STEP 1: FUSING LORA ADAPTERS"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

ADAPTER_DIR=$(dirname "$CHECKPOINT")
FUSED_DIR="$OUTPUT_DIR/fused-model"

mlx_lm.fuse \
    --model "$BASE_MODEL" \
    --adapter-path "$ADAPTER_DIR" \
    --save-path "$FUSED_DIR" \
    --dequantize

echo "✓ Fused model saved to: $FUSED_DIR"
echo ""

# Step 2: Check if llama.cpp exists
echo "════════════════════════════════════════════════════════════════════════"
echo "STEP 2: CHECKING FOR LLAMA.CPP"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

LLAMA_CPP_DIR="./llama.cpp"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "llama.cpp not found. Cloning..."
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    make
    cd ..
    echo "✓ llama.cpp built successfully"
else
    echo "✓ llama.cpp found at $LLAMA_CPP_DIR"
fi

echo ""

# Step 3: Convert to GGUF (F16 first)
echo "════════════════════════════════════════════════════════════════════════"
echo "STEP 3: CONVERTING TO GGUF (F16)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

F16_GGUF="$OUTPUT_DIR/nyay-ai-f16.gguf"

python "$LLAMA_CPP_DIR/convert.py" "$FUSED_DIR" \
    --outtype f16 \
    --outfile "$F16_GGUF"

echo "✓ F16 GGUF created: $F16_GGUF"
echo ""

# Step 4: Quantize
echo "════════════════════════════════════════════════════════════════════════"
echo "STEP 4: QUANTIZING TO $QUANTIZATION"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

QUANTIZED_GGUF="$OUTPUT_DIR/nyay-ai-${QUANTIZATION,,}.gguf"

"$LLAMA_CPP_DIR/quantize" "$F16_GGUF" "$QUANTIZED_GGUF" "$QUANTIZATION"

echo "✓ Quantized GGUF created: $QUANTIZED_GGUF"
echo ""

# Step 5: Create Modelfile
echo "════════════════════════════════════════════════════════════════════════"
echo "STEP 5: CREATING OLLAMA MODELFILE"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

MODELFILE="$OUTPUT_DIR/Modelfile"

cat > "$MODELFILE" << EOF
# Nyay AI - Indian Legal Assistant
# Fine-tuned Llama 3.2 3B for Indian Law

FROM $QUANTIZED_GGUF

# Model parameters
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

# System prompt
SYSTEM """You are Nyay AI, a legal research assistant specializing in Indian law.
You help users understand Indian legal concepts, statutes, case law, and procedures.

Guidelines:
- Provide detailed, well-explained answers (avoid one-word responses)
- Cite relevant sections, articles, and acts when applicable
- If you don't know something, say so clearly
- Indian law context only (Supreme Court, High Courts, Indian statutes)
- Explain legal concepts thoroughly with examples
"""
EOF

echo "✓ Modelfile created: $MODELFILE"
echo ""

# Summary
echo "════════════════════════════════════════════════════════════════════════"
echo "CONVERSION COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "Output files:"
echo "  Fused model: $FUSED_DIR"
echo "  F16 GGUF:    $F16_GGUF"
echo "  Quantized:   $QUANTIZED_GGUF"
echo "  Modelfile:   $MODELFILE"
echo ""

# File sizes
echo "File sizes:"
ls -lh "$F16_GGUF" 2>/dev/null || true
ls -lh "$QUANTIZED_GGUF" 2>/dev/null || true
echo ""

echo "════════════════════════════════════════════════════════════════════════"
echo "NEXT STEPS"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "1. Create Ollama model:"
echo "   ollama create nyay-ai -f $MODELFILE"
echo ""
echo "2. Test the model:"
echo "   ollama run nyay-ai 'What is a writ of habeas corpus?'"
echo ""
echo "3. List your models:"
echo "   ollama list"
echo ""
echo "4. Deploy as API:"
echo "   ollama serve"
echo ""
