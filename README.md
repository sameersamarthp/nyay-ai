# Nyay AI - An Experiment in Indian Legal AI

> **Nyay** (न्याय) means "Justice" in Sanskrit/Hindi

An experimental project exploring whether fine-tuning a small language model (Llama 3.2 3B) on Indian court judgments can improve its understanding of India-specific legal concepts. This is a learning exercise, not a production-ready tool.

[![Status](https://img.shields.io/badge/Status-Experimental-yellow)]()
[![Model](https://img.shields.io/badge/Model-Llama%203.2%203B-blue)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)]()

---

## What This Project Explores

**The Question**: Can fine-tuning a small open-source model on real Indian court judgments help it better understand India-specific legal concepts compared to the base model?

**The Approach**:
- Collected ~58,000 High Court judgments from Delhi and Bombay courts
- Generated ~8,000 training examples using an LLM
- Fine-tuned Llama 3.2 3B using QLoRA on Apple Silicon
- Compared outputs with the base model on a small test set

**What This Is**:
- A personal learning project exploring LLM fine-tuning
- An experiment in domain-specific model adaptation
- A demonstration of local, privacy-preserving AI inference
- An experiment to learn more about Quantisation, GGUF format and its associated trade offs
- ~90% of the code was written by Claude Code, with me providing direction

**What This Is NOT**:
- A production-ready legal assistant
- A replacement for professional legal advice
- A thoroughly validated or benchmarked system
- Something you should rely on for actual legal matters

⚠️ **Important**: This is a research experiment. The model may produce incorrect, incomplete, or misleading information. Always verify any legal information with qualified professionals.

---

## Approach & Design Choices

### 1. **Local & Private Inference**

One benefit of this approach is that queries stay on your machine:
- Runs entirely locally using Ollama
- No data sent to external servers
- Works offline once set up

**Observed performance on M2 MacBook Pro**:
- Speed: ~68 tokens/second
- Memory: ~3 GB RAM during inference
- Model Size: 2 GB on disk

### 2. **Quantization**

The model uses Q4_K_M quantization (4-bit) to compress from 6 GB to 2 GB. Industry benchmarks suggest this typically retains 97-98% quality, though I haven't independently verified this for this specific model.

### 3. **Training Data**

- **Source**: 58,222 judgments from Delhi and Bombay High Courts (via [AWS Open Data](https://registry.opendata.aws/indian-high-court-judgments/))
- **Training examples**: 7,972 examples generated across 4 task types
- **Method**: QLoRA 8-bit fine-tuning (trains ~0.3% of parameters)
- **Hardware**: MacBook Pro M2, 32GB RAM
- **Training time**: ~12 hours

---

## Preliminary Results

### Limited Comparison: Fine-tuned vs Base Model

I tested both models on **20 hand-crafted legal queries** across 8 categories. This is a very small sample size, so take these numbers as rough indicators rather than definitive benchmarks.

| Metric | Nyay AI (Fine-tuned) | Base Llama 3.2 3B |
|--------|---------------------|-------------------|
| Overall Score (automated) | 63.9/100 | ~45/100 (estimated) |
| Coherence Rate | 90% | ~70% |
| Legal Terminology Usage | 75% | ~45% |

**Caveats**:
- Only 20 test cases (statistically insufficient for strong claims)
- Automated scoring has limitations
- Base model scores are rough estimates
- Results may not generalize

### One Interesting Example

**Query**: *"What are the grounds for quashing an FIR under Section 482 CrPC?"*

**Base Llama 3.2 3B** incorrectly stated that a Magistrate can quash FIRs.

**Nyay AI** correctly identified that only the High Court has this power under Section 482.

This suggests the fine-tuning may have helped with some jurisdiction-specific knowledge, but one example doesn't prove general improvement.

### Visual Comparisons

**Evaluation 1: Section 482 CrPC**

*Prompt: What are the grounds for quashing an FIR under Section 482 CrPC in Indian law?*

Base Llama 3.2:3b:
<img width="992" height="506" alt="Base model response" src="https://github.com/user-attachments/assets/53074ac5-54fd-46e3-b6dd-50fad9d4537b" />

Nyay AI (Fine-tuned):
<img width="943" height="549" alt="Fine-tuned model response" src="https://github.com/user-attachments/assets/909316d3-d577-430b-b759-12f3534366aa" />

Gemini as Judge:
<img width="745" height="728" alt="Gemini evaluation" src="https://github.com/user-attachments/assets/fe113882-ac6d-468f-96de-20c9b972d64b" />

**Evaluation 2: Section 498A IPC**

*Prompt: What is Section 498A IPC and what are the essential ingredients to prove an offense under this section?*

Base Llama 3.2:3b:
<img width="986" height="757" alt="Base model response" src="https://github.com/user-attachments/assets/770eb533-222c-4326-a33b-1e9faa81f1da" />

Nyay AI (Fine-tuned):
<img width="973" height="632" alt="Fine-tuned model response" src="https://github.com/user-attachments/assets/b567888c-6461-467f-9fec-c97ff60fc670" />

Gemini as Judge:
<img width="798" height="684" alt="Gemini evaluation" src="https://github.com/user-attachments/assets/529070ae-0628-44d9-8c2a-2ae953d947dd" />

---

## Known Limitations & Issues

This experiment has significant limitations:

### Technical Issues
1. **Case Application Task**: Scores only 6.7/100 - model often gives one-word answers
2. **Data Truncation**: 52% of training examples were truncated at 2048 tokens
3. **Limited Evaluation**: Only 20 test cases (need 100+ for statistical significance)
4. **Hallucination Risk**: ~45% of responses flagged as potential hallucinations

### Fundamental Limitations
- Small model (3B parameters) has inherent capability limits
- Training data quality depends on LLM-generated examples
- No expert legal review of outputs
- May confidently produce incorrect information
- Limited to two High Courts (Delhi, Bombay)

### What Didn't Work Well
- Complex multi-step legal reasoning
- Applying legal principles to novel fact patterns
- Citing specific case precedents accurately

---

## Quick Start (If You Want to Try It)

### Prerequisites
- M1/M2 Mac with 16 GB+ RAM
- [Ollama](https://ollama.ai/) installed
- The GGUF model file (not included in repo due to size)

### Running the Model
```bash
# Create model from Modelfile
ollama create nyay-ai -f models/nyay-ai-gguf/Modelfile

# Interactive mode
ollama run nyay-ai

# Single query
ollama run nyay-ai "What is Section 498A IPC?"
```

### Example Queries That Worked Reasonably Well
- "What is Section 498A IPC?"
- "Explain Article 226 of the Constitution"
- "What are the grounds for issuing a writ of habeas corpus?"

### Example Queries That Didn't Work Well
- Complex hypothetical scenarios
- Requests to apply law to specific facts
- Questions requiring recent legal developments

---

## Project Structure

```
nyay-ai/
├── config/               # Configuration files
├── scrapers/             # Web scrapers for legal data
├── processors/           # Text cleaning, quality filtering
├── storage/              # Database models & operations
├── scripts/              # Training and evaluation scripts
├── docs/                 # Documentation
├── models/               # Trained models (gitignored)
└── data/                 # Training data (gitignored)
```

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Base Model | Llama 3.2 3B Instruct |
| Training Framework | MLX (Apple Silicon) |
| Fine-tuning Method | QLoRA 8-bit |
| Inference | Ollama |
| Quantization | GGUF Q4_K_M |
| Training Data Generation | Claude Haiku API |

---

## What I Learned

1. **Fine-tuning helps with domain vocabulary**: The model uses legal terminology more appropriately
2. **Small models have hard limits**: 3B parameters struggle with complex reasoning
3. **Data quality matters enormously**: LLM-generated training data has its own biases
4. **Evaluation is hard**: Creating good benchmarks is as difficult as training
5. **Local inference is viable**: Consumer hardware can run useful models

---

## Possible Future Directions

If I continue this experiment:
- Create a proper evaluation set (100+ expert-verified test cases)
- Clean training data to fix the case application issue
- Increase context length to reduce truncation
- Try a larger base model (7B+)
- Get expert legal review of outputs

---

## Documentation

| Document | Description |
|----------|-------------|
| [PHASE3_TRAINING_RESULTS.md](docs/PHASE3_TRAINING_RESULTS.md) | Training metrics |
| [GGUF_EXPORT_GUIDE.md](docs/GGUF_EXPORT_GUIDE.md) | Model conversion guide |
| [PHASE3_MODEL_TRAINING.md](docs/PHASE3_MODEL_TRAINING.md) | Training process |

---

## Important Disclaimers

⚠️ **DO NOT use this for actual legal decisions.**

- This is an experiment, not a legal tool
- Outputs may be incorrect, incomplete, or misleading
- No warranty of any kind is provided
- Always consult qualified legal professionals
- The creator assumes no liability for any use of this project

---

## Acknowledgments

- **Data**: [AWS Open Data - Indian High Court Judgments](https://registry.opendata.aws/indian-high-court-judgments/)
- **Base Model**: Meta's Llama 3.2 3B via [mlx-community](https://huggingface.co/mlx-community)
- **Framework**: Apple's [MLX](https://github.com/ml-explore/mlx)
- **Inference**: [Ollama](https://ollama.ai/)
- **Code**: ~90% written by Claude Code

---

## License

MIT License - see [LICENSE](LICENSE) file.

Model weights are subject to Meta's Llama 3.2 license.

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/sameersamarthp/nyay-ai/issues)
- **Email**: sameersamarthp@gmail.com

---

<div align="center">

*An experiment in domain-specific LLM fine-tuning*

</div>
