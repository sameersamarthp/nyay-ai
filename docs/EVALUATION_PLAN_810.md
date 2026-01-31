# Automated Evaluation Plan - 810 Validation Examples

## Overview

Evaluate the deployed Nyay AI model on all **810 validation examples** from `data/training/val.jsonl` to get comprehensive performance metrics beyond the current 20-test baseline.

---

## Evaluation Methodology

### Dataset
- **Source**: `data/training/val.jsonl`
- **Size**: 810 examples
- **Distribution**: ~200 examples per task type
  - Summarization: ~200 examples
  - Research Q&A: ~202 examples
  - Outcome Analysis: ~204 examples
  - Info Extraction: ~204 examples

### Approach
1. Load each validation example (instruction + input)
2. Generate response using deployed `nyay-ai` model via Ollama
3. Compare generated output with ground truth (expected output)
4. Score using automated metrics:
   - Keyword coverage
   - Coherence check
   - Legal terminology presence
   - Hallucination detection
   - Response completeness
   - BLEU/ROUGE scores (similarity with ground truth)

### Time Estimate
- ~810 examples × 10 seconds/example = **~2.25 hours**
- Can run overnight or in background

---

## Final Report Structure

### 1. Executive Summary

```markdown
# Automated Evaluation Report - 810 Validation Examples
**Model**: nyay-ai (Llama 3.2 3B fine-tuned)
**Evaluation Date**: YYYY-MM-DD
**Total Examples**: 810
**Evaluation Time**: ~2.25 hours

## Key Findings
- **Overall Score**: XX.X/100 (target: >60)
- **Pass Rate**: XX% scored >60/100
- **Coherence Rate**: XX% coherent responses
- **Legal Terminology**: XX% accuracy
- **Hallucination Risk**: XX% flagged

## Verdict
✅ PASS / ⚠️ NEEDS IMPROVEMENT / ❌ FAIL

## Comparison with Baseline
- 20-test baseline: 63.9/100
- 810-example evaluation: XX.X/100
- Difference: ±X.X points
- Statistical significance: p < 0.05 (Yes/No)
```

---

### 2. Overall Performance Metrics

```markdown
## Overall Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Overall Score** | XX.X/100 | >60 | ✅/❌ |
| **Pass Rate** (>60) | XX% | >70% | ✅/❌ |
| **Coherence Rate** | XX% | >80% | ✅/❌ |
| **Legal Terminology** | XX% | >70% | ✅/❌ |
| **Keyword Coverage** | XX% | >30% | ✅/❌ |
| **Hallucination Risk** | XX% | <50% | ✅/❌ |
| **Avg Response Length** | XXX words | >100 | ✅/❌ |

## Score Distribution

| Score Range | Count | Percentage |
|-------------|-------|------------|
| 90-100 (Excellent) | XX | XX% |
| 80-89 (Very Good) | XX | XX% |
| 70-79 (Good) | XX | XX% |
| 60-69 (Fair) | XX | XX% |
| 50-59 (Poor) | XX | XX% |
| 0-49 (Fail) | XX | XX% |

**Median Score**: XX.X/100
**Standard Deviation**: XX.X
**25th Percentile**: XX.X
**75th Percentile**: XX.X
**90th Percentile**: XX.X
```

---

### 3. Performance by Task Type

```markdown
## Performance by Task Type

| Task Type | Examples | Avg Score | Pass Rate | Coherence | Legal Terms | Best Score | Worst Score |
|-----------|----------|-----------|-----------|-----------|-------------|------------|-------------|
| **Summarization** | ~200 | XX.X/100 | XX% | XX% | XX% | XX | XX |
| **Research Q&A** | ~202 | XX.X/100 | XX% | XX% | XX% | XX | XX |
| **Outcome Analysis** | ~204 | XX.X/100 | XX% | XX% | XX% | XX | XX |
| **Info Extraction** | ~204 | XX.X/100 | XX% | XX% | XX% | XX | XX |

### Detailed Breakdown

#### 1. Summarization (Average: XX.X/100)
- **Strengths**: Clear structure, key points extraction
- **Weaknesses**: May miss nuanced details
- **Sample Size**: ~200 examples
- **Top 10%**: XX.X/100
- **Bottom 10%**: XX.X/100

#### 2. Research Q&A (Average: XX.X/100)
- **Strengths**: Direct answers, legal accuracy
- **Weaknesses**: Sometimes overly brief
- **Sample Size**: ~202 examples
- **Top 10%**: XX.X/100
- **Bottom 10%**: XX.X/100

#### 3. Outcome Analysis (Average: XX.X/100)
- **Strengths**: Reasoning explanation
- **Weaknesses**: May hallucinate reasoning
- **Sample Size**: ~204 examples
- **Top 10%**: XX.X/100
- **Bottom 10%**: XX.X/100

#### 4. Info Extraction (Average: XX.X/100)
- **Strengths**: Structured output, accuracy
- **Weaknesses**: May miss some entities
- **Sample Size**: ~204 examples
- **Top 10%**: XX.X/100
- **Bottom 10%**: XX.X/100
```

---

### 4. Performance by Court

```markdown
## Performance by Court

| Court | Examples | Avg Score | Pass Rate | Coherence |
|-------|----------|-----------|-----------|-----------|
| Delhi High Court | ~405 | XX.X/100 | XX% | XX% |
| Bombay High Court | ~405 | XX.X/100 | XX% | XX% |

**Observation**: Model performs [similarly/better/worse] on Delhi HC vs Bombay HC cases.
```

---

### 5. Quality Metrics Detail

```markdown
## Quality Metrics

### Coherence Analysis
- **Coherent Responses**: XXX/810 (XX%)
- **Incoherent/Fragmented**: XX/810 (XX%)
- **Empty/Refused**: XX/810 (XX%)

### Legal Terminology
- **Correct Legal Terms**: XXX/810 (XX%)
- **Generic Terms Only**: XX/810 (XX%)
- **Incorrect Terminology**: XX/810 (XX%)

### Keyword Coverage
- **High Coverage (>50%)**: XXX examples (XX%)
- **Medium Coverage (30-50%)**: XXX examples (XX%)
- **Low Coverage (<30%)**: XXX examples (XX%)

### Hallucination Detection
- **Low Risk**: XXX examples (XX%)
- **Medium Risk**: XXX examples (XX%)
- **High Risk**: XXX examples (XX%)

### Response Length
- **Average**: XXX words
- **Median**: XXX words
- **Too Short (<50 words)**: XX examples (XX%)
- **Optimal (50-300 words)**: XX examples (XX%)
- **Too Long (>300 words)**: XX examples (XX%)
```

---

### 6. Failure Analysis

```markdown
## Failure Analysis

### Bottom 10 Examples (Lowest Scores)

| Rank | CNR | Task Type | Score | Issue |
|------|-----|-----------|-------|-------|
| 1 | CNR-XXX | Task | XX/100 | One-word answer / Refusal / Hallucination |
| 2 | CNR-XXX | Task | XX/100 | ... |
| ... | ... | ... | ... | ... |
| 10 | CNR-XXX | Task | XX/100 | ... |

### Common Failure Patterns

1. **One-word/Short Answers** (XX examples, XX%)
   - Task types affected: [List]
   - Root cause: Training data quality issue

2. **Hallucinations** (XX examples, XX%)
   - Task types affected: [List]
   - Root cause: Insufficient grounding

3. **Refusals/Empty Responses** (XX examples, XX%)
   - Task types affected: [List]
   - Root cause: Safety filter / unclear instruction

4. **Off-topic Responses** (XX examples, XX%)
   - Task types affected: [List]
   - Root cause: Misunderstanding instruction

5. **Missing Key Information** (XX examples, XX%)
   - Task types affected: [List]
   - Root cause: Truncation / incomplete generation
```

---

### 7. Top Performers

```markdown
## Top 10 Examples (Highest Scores)

| Rank | CNR | Task Type | Score | Highlight |
|------|-----|-----------|-------|-----------|
| 1 | CNR-XXX | Summarization | XX/100 | Perfect structure, all key points |
| 2 | CNR-XXX | Q&A | XX/100 | Accurate, comprehensive answer |
| ... | ... | ... | ... | ... |
| 10 | CNR-XXX | Info Extraction | XX/100 | All entities extracted correctly |
```

---

### 8. Statistical Analysis

```markdown
## Statistical Analysis

### Score Distribution Histogram

```
Score Range   Frequency
0-10:        ████ (XX)
10-20:       ██████ (XX)
20-30:       ████████ (XX)
30-40:       ███████████ (XX)
40-50:       ████████████████ (XX)
50-60:       ███████████████████ (XX)
60-70:       ██████████████████████ (XX)
70-80:       ████████████████ (XX)
80-90:       █████████ (XX)
90-100:      ████ (XX)
```

### Percentile Analysis

| Percentile | Score |
|------------|-------|
| 5th | XX.X |
| 10th | XX.X |
| 25th | XX.X |
| **50th (Median)** | **XX.X** |
| 75th | XX.X |
| 90th | XX.X |
| 95th | XX.X |

### Variance & Outliers
- **Mean**: XX.X ± XX.X (std dev)
- **Coefficient of Variation**: XX%
- **Outliers (>2 std dev)**: XX examples
```

---

### 9. Comparison with 20-Test Baseline

```markdown
## Comparison with Baseline

| Metric | 20-Test Baseline | 810-Example Eval | Change | Significance |
|--------|------------------|------------------|--------|--------------|
| Overall Score | 63.9/100 | XX.X/100 | ±X.X | p < 0.05 (Yes/No) |
| Coherence Rate | 90% | XX% | ±X% | ... |
| Legal Terms | 75% | XX% | ±X% | ... |
| Hallucination | 45% | XX% | ±X% | ... |

### Key Insights

1. **Consistency**: 810-example evaluation shows [more/less/similar] variance
2. **Reliability**: Larger sample provides [higher/lower] confidence in metrics
3. **Hidden Issues**: Found XX additional failure patterns not visible in 20-test
4. **True Performance**: Estimated true score: XX.X ± X.X (95% confidence)
```

---

### 10. Recommendations

```markdown
## Recommendations for Phase 4

### Critical Issues (Must Fix)
1. **[Issue 1]** - Affects XX% of examples
   - Root cause: [Analysis]
   - Proposed fix: [Solution]
   - Impact: Expected +X points

2. **[Issue 2]** - Affects XX% of examples
   - Root cause: [Analysis]
   - Proposed fix: [Solution]
   - Impact: Expected +X points

### High Priority (Should Fix)
1. **[Issue 3]** - Affects XX% of examples
2. **[Issue 4]** - Affects XX% of examples

### Low Priority (Nice to Have)
1. **[Issue 5]** - Affects XX% of examples
2. **[Issue 6]** - Affects XX% of examples

### Expected Impact
If all critical issues are fixed:
- **Projected Score**: XX.X → XX.X/100 (+X.X points)
- **Projected Pass Rate**: XX% → XX% (+X%)
```

---

### 11. Appendices

```markdown
## Appendix A: Sample Examples

### Example 1: Excellent Performance (Score: 95/100)
**Task**: Summarization
**CNR**: DLHC-XXX
**Instruction**: [Full instruction]
**Expected Output**: [Ground truth]
**Model Output**: [Generated response]
**Analysis**: Why this scored high

### Example 2: Poor Performance (Score: 15/100)
**Task**: Case Application
**CNR**: HCBM-XXX
**Instruction**: [Full instruction]
**Expected Output**: [Ground truth]
**Model Output**: [Generated response]
**Analysis**: Why this failed

[... 8 more examples covering different task types and score ranges ...]

## Appendix B: Methodology Details

### Scoring Algorithm
```python
def calculate_score(response, expected, task_type):
    # Keyword coverage (40%)
    # Coherence (20%)
    # Legal terminology (20%)
    # Hallucination check (10%)
    # Completeness (10%)
    return weighted_score
```

### Evaluation Criteria
- [Detailed explanation of each metric]

## Appendix C: Raw Data Export

Full results exported to:
- `scripts/evaluation_results/val_810_evaluation_YYYYMMDD_HHMMSS.json`
- `scripts/evaluation_results/val_810_evaluation_YYYYMMDD_HHMMSS.csv`
```

---

## Output Files

After evaluation completes, generate:

1. **Main Report**: `docs/AUTOMATED_EVALUATION_REPORT_810.md`
   - Comprehensive markdown report with all sections above

2. **JSON Export**: `scripts/evaluation_results/val_810_evaluation_YYYYMMDD_HHMMSS.json`
   ```json
   {
     "metadata": {
       "model": "nyay-ai",
       "date": "2026-01-30",
       "total_examples": 810,
       "evaluation_time_hours": 2.25
     },
     "summary": {
       "overall_score": 63.5,
       "pass_rate": 72.3,
       "coherence_rate": 88.5,
       ...
     },
     "by_task_type": { ... },
     "by_court": { ... },
     "results": [
       {
         "cnr": "DLHC010011762025",
         "task_type": "summarization",
         "instruction": "...",
         "expected_output": "...",
         "model_output": "...",
         "score": 75.5,
         "metrics": { ... }
       },
       ...
     ]
   }
   ```

3. **CSV Export**: `scripts/evaluation_results/val_810_evaluation_YYYYMMDD_HHMMSS.csv`
   ```csv
   CNR,Task Type,Court,Score,Coherence,Legal Terms,Keyword Coverage,Hallucination Risk,Response Length
   DLHC010011762025,Summarization,Delhi HC,75.5,Yes,Yes,45.2,Low,245
   ...
   ```

4. **Summary Statistics**: `scripts/evaluation_results/val_810_summary.txt`
   - Quick text summary for terminal output

5. **Optional HTML Report**: `scripts/evaluation_results/val_810_evaluation.html`
   - Interactive charts and graphs
   - Sortable/filterable tables
   - Click to view examples

---

## Success Criteria

### Evaluation Pass Criteria
- ✅ Overall score >60/100 (currently 63.9 on 20 tests)
- ✅ Pass rate >70% (examples scoring >60)
- ✅ Coherence rate >80%
- ✅ Legal terminology >70%
- ✅ Hallucination risk <50%

### Expected Outcomes
1. **Confirms baseline**: 810-example score within ±5 points of 63.9
2. **Identifies patterns**: Discovers failure modes not visible in 20 tests
3. **Guides Phase 4**: Provides data-driven priorities for improvements
4. **Statistical confidence**: 95% confidence interval for true performance

---

## Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Setup evaluation script | 1 hour | Adapt existing `evaluate_model.py` |
| Run evaluation | 2.25 hours | Overnight or background |
| Generate report | 30 mins | Automated from JSON |
| Manual review | 1 hour | Review findings, write analysis |
| **Total** | **~5 hours** | Mostly automated |

---

## Next Steps

1. **Create evaluation script** for 810 examples
2. **Run evaluation** overnight
3. **Generate comprehensive report**
4. **Analyze results** and compare with baseline
5. **Update Phase 4 roadmap** based on findings
6. **Share results** in README.md and documentation

---

**Ready to proceed with implementation?** This evaluation will provide the most comprehensive assessment of Nyay AI's performance to date.
