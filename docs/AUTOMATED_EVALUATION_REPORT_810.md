# Automated Evaluation Report - 810 Validation Examples

**Model**: nyay-ai
**Evaluation Date**: 2026-01-31 13:00:10
**Duration**: 0.3 minutes
**Total Examples**: 3

---

## Executive Summary

### Key Findings

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Overall Score** | **34.2/100** | >60 | ❌ FAIL |
| **Pass Rate (>60)** | **33.3%** | >70% | ❌ |
| **Coherence Rate** | **66.7%** | >80% | ❌ |
| **Legal Terminology** | **33.3%** | >70% | ❌ |
| **Keyword Coverage** | **10.5%** | >30% | ❌ |
| **Hallucination (High Risk)** | **0.0%** | <50% | ✅ |

### Verdict

**⚠️ NEEDS IMPROVEMENT**

The model does not meet the baseline quality threshold on the 810-example validation set.

---

## Overall Performance

### Score Distribution

| Score Range | Count | Percentage |
|-------------|-------|------------|
| 90-100 (Excellent) | 0 | 0.0% |
| 80-89 (Very Good) | 0 | 0.0% |
| 70-79 (Good) | 0 | 0.0% |
| 60-69 (Fair) | 1 | 33.3% |
| 50-59 (Poor) | 0 | 0.0% |
| 0-49 (Fail) | 2 | 66.7% |


**Median Score**: 30.5/100
**Standard Deviation**: 26.2
**25th Percentile**: None
**75th Percentile**: None
**90th Percentile**: None

---

## Performance by Task Type

| Task Type | Count | Avg Score | Pass Rate | Coherence | Legal Terms | Coverage |
|-----------|-------|-----------|-----------|-----------|-------------|----------|
| outcome_analysis | 2 | 20.2 | 0.0% | 50.0% | 0.0% | 0.6% |
| research_qa | 1 | 62.1 | 100.0% | 100.0% | 100.0% | 30.4% |


---

## Performance by Court

| Court | Count | Avg Score | Pass Rate | Coherence |
|-------|-------|-----------|-----------|-----------|
| unknown | 3 | 34.2 | 33.3% | 66.7% |


---

## Top 10 Examples (Highest Scores)

| Rank | CNR | Task Type | Score |
|------|-----|-----------|-------|
| 1 | DLHC010689992024 | research_qa | 62.1 |
| 2 | HCBM030212662022 | outcome_analysis | 30.5 |
| 3 | DLHC010011762025 | outcome_analysis | 10.0 |


---

## Bottom 10 Examples (Lowest Scores)

| Rank | CNR | Task Type | Score | Issue |
|------|-----|-----------|-------|-------|
| 1 | DLHC010689992024 | research_qa | 62.1 | Unknown |
| 2 | HCBM030212662022 | outcome_analysis | 30.5 | Short response |
| 3 | DLHC010011762025 | outcome_analysis | 10.0 | Short response |


---

## Comparison with 20-Test Baseline

| Metric | 20-Test Baseline | 810-Example Eval | Change |
|--------|------------------|------------------|--------|
| Overall Score | 63.9/100 | 34.2/100 | -29.7 |
| Coherence Rate | 90% | 66.7% | -23.3% |
| Legal Terminology | 75% | 33.3% | -41.7% |

---

## Recommendations

Based on this comprehensive evaluation of 3 examples:

1. **Overall Performance**: The model needs improvement to meet baseline standards.

2. **Strongest Task Type**: research_qa (Avg: 62.1/100)

3. **Weakest Task Type**: outcome_analysis (Avg: 20.2/100)

4. **Key Findings**:
   - 33.3% of examples scored above 60/100
   - 0.0% flagged as high hallucination risk
   - Average response length: 133 words

---

**Full Results**: `scripts/evaluation_results/val_810_evaluation_20260131_130010.json`
**CSV Export**: `scripts/evaluation_results/val_810_evaluation_20260131_130010.csv`

**Last Updated**: 2026-01-31 13:00:10
