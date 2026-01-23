# Impact of Bad Training Data

## What Happens When Training Data Has Errors?

### The Core Issue

When you fine-tune a model on data with errors, the model **learns those errors as if they were correct**.

```python
# Example of what the model learns:

# Good example (90% of data):
Input: "Analyze this judgment about bail..."
Output: "The court granted bail based on Section 437 CrPC..." ✅

# Bad example with hallucination (10% of data):
Input: "Analyze this judgment about property dispute..."
Output: "The court cited Section 498A IPC..." ❌ (not in source doc)

# Result: Model learns that it's sometimes OK to cite laws not in the document!
```

---

## Specific Impacts by Error Type

### 1. Hallucinations (Making up facts)

**Example:** Output mentions "Section 302 IPC" but source document doesn't reference it

**Impact on trained model:**
- ❌ Model learns to **make up citations**
- ❌ Will cite laws that aren't in documents it analyzes
- ❌ Loses trustworthiness for legal use
- ❌ Could give dangerous/wrong legal advice

**Severity:** 🔴 CRITICAL - Unacceptable for legal AI

---

### 2. Factual Errors

**Example:** Output says "petition allowed" but source says "petition dismissed"

**Impact on trained model:**
- ❌ Model confuses legal outcomes
- ❌ Might reverse actual court decisions
- ❌ Unreliable for case outcome prediction

**Severity:** 🔴 CRITICAL - Unacceptable for legal AI

---

### 3. Incomplete Information

**Example:** Output omits a key legal principle that was central to the case

**Impact on trained model:**
- ⚠️ Model learns to give **incomplete analysis**
- ⚠️ Might miss important legal points
- ⚠️ Less useful but not dangerous

**Severity:** 🟡 MODERATE - Reduces quality but not dangerous

---

### 4. Poor Structure/Formatting

**Example:** Research Q&A missing "QUESTION:" label

**Impact on trained model:**
- ✅ Model might be inconsistent with formatting
- ✅ But factually correct
- ✅ Easy to fix with post-processing

**Severity:** 🟢 MINOR - Annoying but not critical

---

## How Many Bad Examples Are Acceptable?

### Industry Standards

| Error Type | Max Acceptable % | Reasoning |
|------------|------------------|-----------|
| **Hallucinations** | <1% | Dangerous for legal AI - must be nearly zero |
| **Factual errors** | <2% | Critical domain - very low tolerance |
| **Incomplete info** | <10% | Reduces quality but not dangerous |
| **Format issues** | <20% | Annoying but can be fixed |

### Your Nyay AI Thresholds

For a **legal AI** (high-stakes domain):

```
Acceptable error rates:
├─ Hallucinations:     0-1%   (target: 0%)
├─ Factual errors:     0-2%   (target: 0%)
├─ Incomplete:         0-5%   (target: <3%)
└─ Format issues:      0-10%  (can be fixed later)
```

**Why so strict?** Legal AI that gives wrong advice could:
- Harm real people in legal proceedings
- Damage lawyer's reputation
- Create legal liability
- Undermine trust in AI for law

---

## Real-World Example: What 10% Bad Data Does

### Scenario: 720 examples (10%) have hallucinations

**Training process:**
```
Epoch 1: Model sees 7,200 examples
  - 6,480 teach correct behavior ✅
  - 720 teach hallucination is OK ❌

Epoch 2: Model sees same 7,200 again
  - Reinforces patterns from both good and bad

Epoch 3: Model sees same 7,200 again
  - Bad patterns now deeply embedded
```

**Resulting model behavior:**
```python
# User asks about a real case
user: "Summarize this bail application judgment"

# Model (influenced by hallucinated training data):
model: "The court granted bail under Section 437 CrPC,
        considering Section 439 and precedent from State v. Kumar (2018)..."

# Reality: The judgment only mentions Section 437,
# the other citations are HALLUCINATED
```

**Real-world impact:**
- Lawyer relies on this → cites non-existent precedent in court
- Opposing counsel catches it → embarrassment + loss of credibility
- Judge sanctioned the lawyer → professional consequences

---

## What To Do About Bad Training Examples

### Option 1: Remove Bad Examples (Recommended)

**Process:**
```bash
# 1. Identify bad examples
python scripts/automated_quality_checks.py  # Flags 17%

# 2. Manual review of flagged
python scripts/manual_review_helper.py --sample 1224  # 17% of 7,200

# 3. Create filtered dataset
python scripts/filter_bad_examples.py \
  --input data/training/train.jsonl \
  --remove-cnrs bad_examples.txt \
  --output data/training/train_clean.jsonl
```

**Result:**
- Started with: 7,200 examples
- Removed 150 bad examples (2%)
- Final dataset: 7,050 clean examples ✅

**Trade-off:**
- ✅ Higher quality data
- ❌ Slightly less data (but worth it!)

---

### Option 2: Fix Bad Examples

**Process:**
```bash
# Regenerate only the bad examples with improved prompts
python scripts/regenerate_bad_examples.py \
  --input bad_examples.txt \
  --output data/training/regenerated.jsonl
```

**Trade-off:**
- ✅ Keep same dataset size
- ❌ More expensive (API calls)
- ❌ Time-consuming

---

### Option 3: Weight/Downsample Bad Examples

**Process:**
```python
# During training, give less weight to suspect examples
train_config = {
    "train_file": "train.jsonl",
    "sample_weights": {
        "high_confidence": 1.0,  # 90% of data
        "low_confidence": 0.3,   # 10% suspect data
    }
}
```

**Trade-off:**
- ✅ Keep all data
- ⚠️ Bad examples still influence model (just less)
- ❌ Complex to implement

---

### Option 4: Do Nothing (NOT RECOMMENDED)

**If 10% of data has hallucinations:**
- ❌ Model will hallucinate ~10% of the time
- ❌ Unreliable for production use
- ❌ Dangerous for legal AI

---

## Detection Strategy

### During Training Data Generation

**Prevent bad data from being created:**

```python
# In llm_generator.py - add validation

def generate_example(self, document_text: str, task_type: TaskType):
    output, tokens_in, tokens_out = self._call_llm(...)

    # Validation checks
    if self._has_hallucination(output, document_text):
        logger.warning("Hallucination detected, skipping")
        return None, 0, 0  # Don't save this example

    if self._has_factual_error(output, document_text):
        logger.warning("Factual error detected, skipping")
        return None, 0, 0

    return output, tokens_in, tokens_out

def _has_hallucination(self, output: str, source: str) -> bool:
    """Check if output contains facts not in source."""
    # Extract citations from output
    output_citations = extract_legal_citations(output)

    # Check if all citations exist in source
    for citation in output_citations:
        if citation not in source:
            return True  # Hallucination detected!

    return False
```

---

### After Training Data Generation

**Filter existing dataset:**

```bash
# 1. Automated filtering
python scripts/automated_quality_checks.py
# Output: flagged_examples.txt

# 2. Manual review of flagged
python scripts/manual_review_helper.py --flagged flagged_examples.txt
# Output: confirmed_bad_examples.txt

# 3. Remove bad examples
python scripts/filter_training_data.py \
  --remove confirmed_bad_examples.txt \
  --input train.jsonl \
  --output train_clean.jsonl
```

---

## Recommended Workflow for Nyay AI

### Step 1: Prevention (During Generation)

Add validation to `llm_generator.py`:

```python
class LLMGenerator:
    def generate_example(self, doc_text, task_type):
        output, input_tok, output_tok = self._call_claude(doc_text, task_type)

        # Validate before accepting
        if not self._is_valid_output(output, doc_text):
            logger.warning(f"Invalid output for {task_type}, skipping")
            return None, 0, 0

        return output, input_tok, output_tok

    def _is_valid_output(self, output: str, source: str) -> bool:
        """Validation checks"""
        # Check 1: No obvious hallucination markers
        if "I don't have" in output or "cannot find" in output:
            return False

        # Check 2: Output not too short
        if len(output) < 100:
            return False

        # Check 3: Key legal terms present
        if not any(term in output.lower() for term in
                   ["court", "case", "judgment", "order"]):
            return False

        return True
```

### Step 2: Detection (After Generation)

```bash
# Run comprehensive quality checks
python scripts/automated_quality_checks.py
python scripts/manual_review_helper.py --sample 500
```

### Step 3: Action

**If error rate < 5%:**
- ✅ Remove bad examples
- ✅ Continue with reduced dataset

**If error rate 5-15%:**
- ⚠️ Review and filter
- ⚠️ Consider regenerating with better prompts

**If error rate > 15%:**
- 🔴 STOP - Don't use this data
- 🔴 Improve prompts and regenerate from scratch
- 🔴 Consider using a better model (Claude Sonnet instead of Haiku)

---

## The Bottom Line

### Critical Understanding:

```
❌ WRONG: "10% bad data = model is 10% wrong"
✅ RIGHT: "10% bad data = model learns bad habits = unreliable"
```

**The model doesn't know which examples are wrong!** It learns equally from good and bad examples.

### For Legal AI Specifically:

**Acceptable error rates:**
- Hallucinations: **0%** (target)
- Factual errors: **<2%**
- Incomplete: **<5%**

**Your action plan:**
1. **Generate with validation** (prevent bad data)
2. **Automated + manual checks** (detect bad data)
3. **Remove or regenerate** (fix bad data)
4. **Final clean dataset** (high-quality training)

### The Cost of Bad Data

| Scenario | Cost | Impact |
|----------|------|--------|
| Remove 150 bad examples | Free | ✅ Clean dataset, slightly smaller |
| Keep 150 bad examples | Free | ❌ Model learns bad habits, unreliable |
| Regenerate 150 examples | ~$0.38 | ✅ Clean dataset, same size |

**Decision:** Removing or regenerating bad examples is ALWAYS worth it for legal AI!

---

## Quick Reference: Error Tolerance

```python
# Your verification results
total_examples = 7200
bad_examples = your_manual_review_count

error_rate = bad_examples / total_examples

if error_rate < 0.01:  # <1%
    print("✅ Excellent quality - proceed with training")
elif error_rate < 0.05:  # <5%
    print("✅ Acceptable - remove bad examples and proceed")
elif error_rate < 0.15:  # <15%
    print("⚠️ Concerning - filter heavily or regenerate")
else:  # >15%
    print("🔴 STOP - Regenerate with better prompts/model")
```
