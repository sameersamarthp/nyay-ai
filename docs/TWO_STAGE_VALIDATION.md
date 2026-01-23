# Two-Stage Validation Strategy

## Why Two Stages?

Different types of checks are best done at different times:

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: DURING Generation (Real-time, Built-in)          │
│  • Fast, simple checks                                      │
│  • Prevents obvious garbage from being saved                │
│  • Saves API costs (don't pay for failed generations)      │
│  • Catches ~5-10% of obvious failures                       │
└─────────────────────────────────────────────────────────────┘
                           ↓
              Generates clean examples only
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: AFTER Generation (Deep analysis, Separate)       │
│  • Comprehensive, time-intensive checks                     │
│  • Requires comparison with source documents               │
│  • Human judgment for subtle issues                         │
│  • Catches ~2-5% of subtle problems                         │
└─────────────────────────────────────────────────────────────┘
                           ↓
                  Final clean dataset
```

---

## Stage 1: Built-in Validation (During Generation)

### Location
In `processors/llm_generator.py` → `_is_valid_output()` method

### What It Checks

| Check | Threshold | Why |
|-------|-----------|-----|
| **Empty output** | Not empty | LLM failed to generate |
| **Too short** | >200 chars (varies by task) | Incomplete response |
| **Too long** | <5000 chars | Error or untruncated source |
| **Refusals** | No "I cannot", "I don't have" | LLM couldn't complete task |
| **Format** | Task-specific structure | Missing required format |
| **Legal terms** | Must have legal words | Not about legal content |

### Example Code

```python
def _is_valid_output(self, output: str, task_type: TaskType) -> bool:
    """Stage 1 validation during generation."""

    # Check 1: Not empty
    if not output or not output.strip():
        return False

    # Check 2: Minimum length
    if len(output) < 200:
        return False

    # Check 3: No refusal patterns
    if "i cannot" in output.lower():
        return False

    # Check 4: Has legal terminology
    if not any(term in output.lower() for term in
               ["court", "case", "judgment"]):
        return False

    return True
```

### When It Runs

```python
# In generate_example() method:
response = self.client.messages.create(...)
output = response.content[0].text

# Validate BEFORE saving
if not self._is_valid_output(output, task_type):
    logger.warning("Output failed validation")
    return None, 0, 0  # Don't save, don't count tokens

# Only save if valid
return output, input_tokens, output_tokens
```

### Benefits

✅ **Saves money** - Don't pay for bad generations
✅ **Saves time** - Don't generate what you'll throw away
✅ **Cleaner data** - Only good examples get saved
✅ **Automatic** - No extra work needed

### Limitations

❌ Can't check factual accuracy (too slow)
❌ Can't detect subtle hallucinations (needs source comparison)
❌ Can't evaluate legal reasoning (needs human judgment)

### Statistics

After generation, you'll see:

```
Examples generated: 7,500
Examples rejected (Stage 1 validation): 500 (6.3%)
```

This means 500 examples failed basic quality checks and weren't saved.

---

## Stage 2: Deep Validation (After Generation)

### Location
Separate scripts run AFTER generation completes

### Tools Available

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `automated_quality_checks.py` | Fast checks on saved data | Run on all 8K examples |
| `manual_review_helper.py` | Side-by-side comparison | Review flagged + 10% sample |
| `filter_bad_examples.py` | Remove identified bad examples | After manual review |

### What It Checks

| Check | How | Who |
|-------|-----|-----|
| **Hallucinations** | Compare output to source doc | Automated + Manual |
| **Factual errors** | Verify dates, names, outcomes | Manual |
| **Incomplete info** | Check key points covered | Manual |
| **Legal reasoning** | Assess legal soundness | Manual (ideally expert) |
| **Citation accuracy** | Verify all citations exist in source | Automated |

### Workflow

```bash
# Step 1: Automated checks (5 minutes)
python scripts/automated_quality_checks.py
# Output: Flags ~10-20% for manual review

# Step 2: Review flagged examples
python scripts/manual_review_helper.py --sample 1000
# Mark bad CNRs in bad_examples.txt

# Step 3: Review random sample
python scripts/manual_review_helper.py --sample 400
# Add more bad CNRs to bad_examples.txt

# Step 4: Filter out bad examples
python scripts/filter_bad_examples.py \
  --input data/training/train.jsonl \
  --remove bad_examples.txt \
  --output data/training/train_clean.jsonl

# Step 5: Use clean dataset for training
```

### Benefits

✅ **Catches subtle issues** - Hallucinations, factual errors
✅ **Human judgment** - Legal reasoning, completeness
✅ **Flexible** - Can adjust criteria after generation
✅ **Comprehensive** - Deep analysis of saved data

### Limitations

⚠️ Time-consuming (manual review takes hours)
⚠️ Already paid for bad examples (but caught them!)
⚠️ Requires legal knowledge for best results

---

## Comparison: Why Both?

### If You Only Did Stage 1:

```
Generated: 8,000 examples
Rejected during generation: 800 (obvious failures)
Saved: 7,200 examples

BUT: May still have ~200-400 subtle issues
- Hallucinations (made-up citations)
- Factual errors (wrong outcomes)
- Incomplete analysis
```

**Result:** 7,200 examples, but ~5% still have issues

### If You Only Did Stage 2:

```
Generated: 8,000 examples (including garbage)
Cost: $16 for all 8,000
Saved: 8,000 examples

Manual review finds: 800 failures
Filter them out: 7,200 remaining
```

**Result:** 7,200 examples, but wasted $1.60 on garbage

### With Both Stages:

```
Stage 1 (during generation):
- Attempted: 8,500 examples
- Rejected: 500 obvious failures
- Generated: 8,000 good examples
- Cost: $14 (saved $1 on rejected examples)

Stage 2 (after generation):
- Automated checks: Flag 1,200 for review
- Manual review: Find 300 actual bad examples
- Filter: 7,700 clean examples remain
```

**Result:** 7,700 high-quality examples, saved money, caught issues

---

## Decision Guide: Where to Put Validation?

### Stage 1 (Built-in) - Use for:

✅ Fast checks (<1 second)
✅ Binary pass/fail (no gray area)
✅ Can be automated reliably
✅ Don't need source document comparison

**Examples:**
- Empty output
- Too short/long
- Missing required format
- Refusal patterns
- No legal terminology

### Stage 2 (Separate) - Use for:

✅ Slow checks (>1 second)
✅ Requires human judgment
✅ Needs source document comparison
✅ Nuanced evaluation

**Examples:**
- Hallucination detection
- Factual accuracy
- Legal reasoning quality
- Completeness assessment
- Citation verification

### Never Do:

❌ Expensive checks in Stage 1 (slows generation)
❌ Simple checks only in Stage 2 (wastes API costs)
❌ Skip validation entirely (bad training data)

---

## Practical Example

### Your 10-Document Test Run

**Stage 1 (Automatic):**
```
Attempted: 20 examples (10 docs × 2 each)
Rejected: 0 (all passed basic checks)
Generated: 20 examples
Cost: $0.02
```

**Stage 2 (Manual):**
```bash
# Check for issues
python scripts/automated_quality_checks.py
# Output: Flagged 17/20 for "excessive repetition" (false positive)

# Manual review
python scripts/manual_review_helper.py --sample 20
# You verify: Actually all 20 look good!

# No filtering needed
Final: 20 clean examples ✅
```

### Full 4,000-Document Run (Projected)

**Stage 1 (Automatic):**
```
Attempted: 8,500 examples
Rejected: 500 (5.9%)
  - 200 too short
  - 150 refusals
  - 100 no legal terms
  - 50 format issues
Generated: 8,000 examples
Cost: $14 (saved $1 on rejections)
```

**Stage 2 (Manual):**
```bash
python scripts/automated_quality_checks.py
# Flags: 1,500 examples (18.75%)

python scripts/manual_review_helper.py --sample 1500
# Actual bad: 300 examples (3.75%)
  - 150 hallucinations
  - 100 factual errors
  - 50 incomplete

python scripts/filter_bad_examples.py ...
# Final: 7,700 clean examples (96.25% quality)
```

**Total quality rate:** 7,700 / 8,500 attempted = 90.6% success

---

## Monitoring & Improvement

### Track Stage 1 Rejection Rate

```bash
# After generation, check rejection rate
Examples rejected (Stage 1): 500 (5.9%)
```

**If rejection rate:**
- <5%: ✅ Excellent - prompts working well
- 5-10%: ✅ Good - minor prompt improvements possible
- 10-20%: ⚠️ Concerning - review prompts and model
- >20%: 🔴 Problem - improve prompts or use better model

### Track Stage 2 Failure Rate

```bash
# After manual review
Total examples: 8,000
Flagged by automation: 1,500
Actual bad (manual review): 300 (3.75%)
```

**If failure rate:**
- <2%: ✅ Excellent - Stage 1 doing its job
- 2-5%: ✅ Good - acceptable for training
- 5-10%: ⚠️ Concerning - tighten Stage 1 validation
- >10%: 🔴 Problem - Stage 1 not catching enough

### Improve Stage 1 Validation

If Stage 2 consistently finds specific issues, add them to Stage 1:

```python
# Found many hallucinated citations in Stage 2?
# Add to Stage 1:
def _is_valid_output(self, output: str, task_type: TaskType) -> bool:
    # ... existing checks ...

    # New check: No obvious hallucination patterns
    if "Section 498A" in output and task_type == TaskType.INFO_EXTRACTION:
        # This section is commonly hallucinated
        # Could add lightweight check
        pass

    return True
```

---

## Cost-Benefit Analysis

### Stage 1 Only

| Metric | Value |
|--------|-------|
| Examples attempted | 8,500 |
| Examples generated | 8,000 |
| API cost | $14 |
| Manual review time | 0 hours |
| **Quality issues** | **~400 (5%)** |

### Stage 2 Only

| Metric | Value |
|--------|-------|
| Examples attempted | 8,000 |
| Examples generated | 8,000 |
| API cost | $16 |
| Manual review time | 25 hours |
| **Wasted API cost** | **$1** |

### Both Stages

| Metric | Value |
|--------|-------|
| Examples attempted | 8,500 |
| Examples generated | 7,700 clean |
| API cost | $14 (saved $1) |
| Manual review time | 25 hours |
| **Quality issues** | **<100 (<1.3%)** |

**Conclusion:** Both stages together give best results!

---

## Summary

### Stage 1 (Built-in): Fast, Automatic, Preventive
- **When:** During generation in `llm_generator.py`
- **What:** Simple, fast checks
- **Catches:** 5-10% obvious failures
- **Benefit:** Save API costs, cleaner data

### Stage 2 (Separate): Deep, Manual, Detective
- **When:** After generation completes
- **What:** Comprehensive analysis with human review
- **Catches:** 2-5% subtle issues
- **Benefit:** High-quality dataset for training

### Together: Maximum Quality
- **Total quality:** >98% clean examples
- **Cost savings:** ~$1 from Stage 1 rejections
- **Confidence:** Know your data is trustworthy
- **Safety:** Critical for legal AI

---

## Quick Commands

```bash
# Stage 1 happens automatically during:
python scripts/prepare_training_data.py

# Stage 2 - run after generation:
python scripts/automated_quality_checks.py
python scripts/manual_review_helper.py --sample 500
python scripts/filter_bad_examples.py --input train.jsonl --remove bad.txt --output train_clean.jsonl
```

See `docs/DATA_QUALITY_VERIFICATION.md` for complete verification guide.
