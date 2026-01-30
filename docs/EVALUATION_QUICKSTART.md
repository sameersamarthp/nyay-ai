# Nyay AI - Evaluation Quick Start

## ✅ Training Complete
- **Total Iterations**: 3000 (1500 + 1500)
- **Final Checkpoint**: `models/nyay-ai-checkpoints-v4/0003000_adapters.safetensors`
- **Final Train Loss**: 1.182
- **Final Val Loss**: 1.165
- **Status**: ✅ Target achieved (< 1.2)

## 🧪 Run Evaluation

### Option 1: Quick Test (5 cases, ~2 minutes)
```bash
source .venv-train/bin/activate
python scripts/evaluate_model.py --limit 5
```

### Option 2: Full Evaluation (20 cases, ~8 minutes)
```bash
python scripts/evaluate_model.py
```

### Option 3: Interactive with Manual Scoring
```bash
python scripts/evaluate_model.py --interactive --limit 10
```

## 📊 What Gets Measured

| Metric | What It Means |
|--------|---------------|
| **Overall Score** | 0-100 composite quality score |
| **Keyword Coverage** | % of expected legal terms present |
| **Coherence** | Is response logically structured? |
| **Legal Terminology** | Uses proper legal language? |
| **Hallucination Risk** | Likelihood of fabricated information |

## 📁 Output Files

Results automatically saved to:
```
scripts/evaluation_results/0003000_adapters_YYYYMMDD_HHMMSS.json
```

## 🎯 Success Criteria

| Score Range | Assessment | Action |
|-------------|------------|--------|
| ≥ 75 | Excellent - Production ready | Deploy! |
| 60-74 | Good - Minor improvements | Consider deployment |
| 40-59 | Acceptable - Needs work | More training recommended |
| < 40 | Poor | Retrain with better data |

## 🔄 Compare Checkpoints

Test different iterations to find the best one:

```bash
# Test iter 2000
python scripts/evaluate_model.py \
  --checkpoint models/nyay-ai-checkpoints-v4/0002000_adapters.safetensors \
  --output results_iter2000.json

# Test iter 2500  
python scripts/evaluate_model.py \
  --checkpoint models/nyay-ai-checkpoints-v4/0002500_adapters.safetensors \
  --output results_iter2500.json

# Test iter 3000 (final)
python scripts/evaluate_model.py \
  --checkpoint models/nyay-ai-checkpoints-v4/0003000_adapters.safetensors \
  --output results_iter3000.json
```

## 📝 Test Coverage

20 test cases across 8 task types:
- Legal Q&A (5 cases)
- Concept explanation (3 cases)
- Statutory interpretation (2 cases)
- Procedural law (2 cases)
- Fundamental rights (2 cases)
- Case application (2 cases)
- Legal reasoning (2 cases)
- Jurisdiction (2 cases)

## ⏭️ After Evaluation

Based on results:
1. If score ≥ 75: Export to GGUF and deploy with Ollama
2. If score 60-74: Review low-scoring cases, possibly deploy
3. If score < 60: Analyze failures, improve training data, retrain

---

**Recommended Next Step**: Run the quick test (5 cases) to get initial quality assessment:

```bash
source .venv-train/bin/activate
python scripts/evaluate_model.py --limit 5
```
