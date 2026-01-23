# Manual Review Guide

## Two Options for Manual Review

### Option 1: Visual Review (Simple)
**Script:** `manual_review_helper.py`
- Shows source + output side-by-side
- You review mentally
- Manually track bad CNRs

### Option 2: Interactive Review (Recommended)
**Script:** `interactive_review.py`
- Asks you 5 questions for each example
- Automatically tracks results
- Saves bad CNRs to file

---

## Option 1: Visual Review (Simple)

### How to Use

```bash
# Review 20 examples
python scripts/manual_review_helper.py --sample 20
```

### What You'll See

```
================================================================================
EXAMPLE 1 of 20
================================================================================
CNR: DLHC010011762025
Task Type: outcome_analysis
================================================================================

📄 SOURCE DOCUMENT (First 3000 chars):
--------------------------------------------------------------------------------
[Source judgment text here...]

================================================================================
📝 INSTRUCTION:
--------------------------------------------------------------------------------
Analyze the outcome of this judgment...

================================================================================
🤖 GENERATED OUTPUT:
--------------------------------------------------------------------------------
[Generated output here...]

================================================================================
🔍 REVIEW CHECKLIST:
--------------------------------------------------------------------------------
1. [ ] Factually accurate (check dates, parties, outcomes)
2. [ ] Legally sound (correct interpretation of law)
3. [ ] No hallucination (all facts from source document)
4. [ ] Clear and well-structured
5. [ ] Appropriate length and detail
================================================================================

Press Enter for next, 'q' to quit: _
```

### How to Input Your Answer

**You DON'T input answers directly** - just review mentally:

1. Read the source document
2. Read the generated output
3. Mentally check each item in the checklist
4. If **ALL 5 items are ✓**: Press **Enter** (example is good)
5. If **ANY item is ✗**: Write down the CNR, then press **Enter**

### Tracking Bad Examples

**Manual method:**
```bash
# Keep a text file open
nano bad_cnrs.txt

# As you review, add bad CNRs:
DLHC010011762025
HCBM030212662022

# Save and exit (Ctrl+X, Y, Enter)
```

### Pros & Cons

✅ Simple, no complex interaction
✅ Fast for experienced reviewers
❌ Manual tracking of bad CNRs
❌ No automatic statistics

---

## Option 2: Interactive Review (Recommended)

### How to Use

```bash
# Review 20 examples interactively
python scripts/interactive_review.py --sample 20
```

### What You'll See

```
================================================================================
EXAMPLE 1 of 20
================================================================================
CNR: DLHC010011762025
Task Type: outcome_analysis
================================================================================

📄 SOURCE DOCUMENT (First 3000 chars):
--------------------------------------------------------------------------------
[Source judgment text here...]

================================================================================
📝 INSTRUCTION:
--------------------------------------------------------------------------------
Analyze the outcome of this judgment...

================================================================================
🤖 GENERATED OUTPUT:
--------------------------------------------------------------------------------
[Generated output here...]

================================================================================
🔍 INTERACTIVE REVIEW:
--------------------------------------------------------------------------------

Compare the generated output with the source document above.

1. Factually accurate (dates, parties, outcomes match source)? [Y/n]: _
```

### How to Input Your Answer

For each question, you have two options:

**Option A: Press Enter (Default = Yes)**
```
1. Factually accurate? [Y/n]: ← Just press Enter
```
This means "Yes, it's factually accurate"

**Option B: Type 'n' for No**
```
1. Factually accurate? [Y/n]: n ← Type 'n' and press Enter
```
This means "No, it's NOT factually accurate"

### Interactive Flow

```
1. Factually accurate (dates, parties, outcomes match source)? [Y/n]: ← Press Enter (yes)
2. Legally sound (correct interpretation of law)? [Y/n]: ← Press Enter (yes)
3. No hallucination (all facts are from source document)? [Y/n]: n ← Type 'n' (no - found hallucination!)
4. Clear and well-structured? [Y/n]: ← Press Enter (yes)
5. Appropriate length and detail? [Y/n]: ← Press Enter (yes)

Optional: Add notes about the issues (press Enter to skip):
> Output cites Section 498A which is not in source document ← Add notes (optional)

❌ FAIL - Example marked as bad
```

### Automatic Tracking

The script automatically:
- Tracks which examples passed/failed
- Saves detailed results to `data/training/review_results.json`
- Creates `data/training/bad_cnrs.txt` with all bad CNRs
- Shows summary statistics

### Results Files

After reviewing, you'll get:

**File 1: `review_results.json`** (Detailed results)
```json
[
  {
    "cnr": "DLHC010011762025",
    "task_type": "outcome_analysis",
    "checks": {
      "factually_accurate": true,
      "legally_sound": true,
      "no_hallucination": false,
      "clear_structured": true,
      "appropriate_length": true
    },
    "notes": "Cites Section 498A not in source",
    "passed": false
  }
]
```

**File 2: `bad_cnrs.txt`** (Ready to use for filtering)
```
DLHC010011762025
HCBM030212662022
```

### Summary Output

```
======================================================================
REVIEW SESSION SUMMARY
======================================================================
Total reviewed: 20
Passed all checks: 18 (90.0%)
Failed one or more: 2 (10.0%)

Results saved to: data/training/review_results.json
Bad CNRs saved to: data/training/bad_cnrs.txt

To filter them out:
  python scripts/filter_bad_examples.py \
    --input data/training/train.jsonl \
    --remove data/training/bad_cnrs.txt \
    --output data/training/train_clean.jsonl
```

### Pros & Cons

✅ Structured, guided review
✅ Automatic tracking and statistics
✅ Can add notes for each issue
✅ Resume capability
✅ Ready-to-use bad CNRs file
❌ Slightly slower (more keypresses)

---

## Quick Comparison

| Feature | Visual Review | Interactive Review |
|---------|---------------|-------------------|
| **Input method** | Mental checklist | Answer 5 questions |
| **Speed** | Fast | Moderate |
| **Tracking** | Manual | Automatic |
| **Statistics** | None | Automatic |
| **Bad CNRs** | Manual list | Auto-generated file |
| **Notes** | Separate notes | Captured in JSON |
| **Resume** | No | Yes |
| **Best for** | Quick checks | Thorough review |

---

## Recommended Workflow

### For 20 Test Examples:

**Use Interactive Review:**
```bash
python scripts/interactive_review.py --sample 20
```
Why: Small sample, worth the detailed tracking

### For 8,000 Full Dataset:

**Step 1:** Automated checks flag ~1,500
```bash
python scripts/automated_quality_checks.py
```

**Step 2:** Review flagged examples (interactive)
```bash
python scripts/interactive_review.py --sample 1500
```

**Step 3:** Review random sample (visual - faster)
```bash
python scripts/manual_review_helper.py --sample 400
# Manually note bad CNRs in bad_cnrs.txt
```

**Step 4:** Filter out all bad examples
```bash
python scripts/filter_bad_examples.py \
  --input data/training/train.jsonl \
  --remove data/training/bad_cnrs.txt \
  --output data/training/train_clean.jsonl
```

---

## Interactive Review Tips

### Keyboard Shortcuts

- **Enter** = Yes (default for all questions)
- **n** + Enter = No
- **s** = Save and stop

### Efficient Reviewing

If example is clearly good:
```
1. Factually accurate? [Y/n]: ← Enter
2. Legally sound? [Y/n]: ← Enter
3. No hallucination? [Y/n]: ← Enter
4. Clear? [Y/n]: ← Enter
5. Appropriate length? [Y/n]: ← Enter
```
5 keypresses = done!

If example has issues:
```
1. Factually accurate? [Y/n]: n ← Type 'n'
2. Legally sound? [Y/n]: ← Enter
3. No hallucination? [Y/n]: n ← Type 'n'
4. Clear? [Y/n]: ← Enter
5. Appropriate length? [Y/n]: ← Enter

Add notes: Wrong date, hallucinated citation ← Optional
```

### Resume Interrupted Sessions

```bash
# Start review session
python scripts/interactive_review.py --sample 500

# ... Press 's' to save and stop after 100 examples

# Later, continue from where you left off
python scripts/interactive_review.py \
  --sample 500 \
  --continue-session data/training/review_results.json
```

### Review Only Validation Set

```bash
# Focus on val.jsonl (most important!)
python scripts/interactive_review.py --split val
```

### Review Specific CNR

```bash
# Double-check a specific example
python scripts/interactive_review.py --cnr DLHC010011762025
```

---

## Try It Now

### Quick Test (2 validation examples):

```bash
# Interactive review
source .venv/bin/activate
python scripts/interactive_review.py --split val
```

You'll be asked 5 questions for each of the 2 examples.
Press **Enter** 5 times if good, type **'n'** if any issue.

### What to Look For

| Check | What to Verify | Example Issue |
|-------|----------------|---------------|
| **Factually accurate** | Dates, parties, outcomes match source | "Petition filed 2023" but source says "2024" |
| **Legally sound** | Correct interpretation of law | "Section 302 applies to civil matters" (wrong!) |
| **No hallucination** | All facts exist in source | "Court cited Sharma v. Kumar" (not in source) |
| **Clear & structured** | Easy to read and understand | Confusing sentences, poor grammar |
| **Appropriate length** | Not too brief, not too verbose | 50-word summary of complex case |

---

## After Review: Next Steps

### 1. Check Results

```bash
# How many failed?
cat data/training/review_results.json | grep '"passed": false' | wc -l

# What's the failure rate?
# Example: 2 failed out of 20 = 10%
```

### 2. Filter Out Bad Examples

```bash
python scripts/filter_bad_examples.py \
  --input data/training/train.jsonl \
  --remove data/training/bad_cnrs.txt \
  --output data/training/train_clean.jsonl
```

### 3. Validate Filtered Data

```bash
python scripts/validate_training_data.py --input-dir data/training
```

### 4. Use Clean Data for Training

```bash
# Phase 3: Fine-tune with clean data
# (coming soon)
```

---

## Summary

**Choose Interactive Review if:**
- ✅ You want automatic tracking
- ✅ You value structured guidance
- ✅ You need detailed statistics
- ✅ You're reviewing validation set (critical!)

**Choose Visual Review if:**
- ✅ You're experienced and fast
- ✅ You prefer minimal interaction
- ✅ You're doing quick spot checks
- ✅ You'll manually track bad CNRs anyway

**For your current 20 test examples:** Use Interactive Review!

```bash
source .venv/bin/activate
python scripts/interactive_review.py --sample 20
```
