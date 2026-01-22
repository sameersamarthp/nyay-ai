## Phase 2: Data Processing

### Overview

| Aspect | Value |
|--------|-------|
| **Input** | 58,222 documents in SQLite (57,398 with full_text) |
| **Output** | 8,000 training examples in JSONL format |
| **Timeline** | 3-4 days |
| **Approach** | LLM-based generation using Claude API |
| **Estimated Cost** | ~$30-50 (depending on document lengths) |

### New Files to Create
```
nyay-ai-india/
├── processors/
│   ├── text_cleaner.py           # Light text preprocessing
│   ├── quality_filter.py         # Filter and sample documents
│   └── llm_generator.py          # Claude API integration for training data
├── scripts/
│   ├── prepare_training_data.py  # Main processing script
│   └── validate_training_data.py # Validate output
├── config/
│   └── llm_prompts.py            # Prompt definitions for Claude API
└── data/
    └── training/                 # Output directory
        ├── train.jsonl           # 7,200 examples (90%)
        └── val.jsonl             # 800 examples (10%)
```

### Training Data Format

Each line in JSONL files:
```json
{"instruction": "Summarize this judgment", "input": "[judgment text]", "output": "[LLM-generated summary]"}
```

### Training Example Types & Targets

| Type | Count | Description |
|------|-------|-------------|
| Case Summarization | 3,000 | Structured summaries of full judgments |
| Legal Research Q&A | 2,500 | Questions about legal principles, holdings, reasoning |
| Outcome Analysis | 1,500 | Explain why court decided a certain way |
| Information Extraction | 1,000 | Extract parties, issues, citations from text |
| **Total** | **8,000** | |

### LLM Generation Strategy
```
┌─────────────────────────────────────────────────────────────────┐
│                   LLM GENERATION PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Sample 4,000 documents from 57K                       │
│  ├── Balance by court (Delhi HC, Bombay HC)                    │
│  ├── Balance by case type (disposal_nature)                    │
│  └── Filter for quality (500-15000 words)                      │
│                                                                 │
│  Step 2: For each document, generate 2 training examples       │
│  ├── Example 1: Summarization OR Research Q&A                  │
│  └── Example 2: Outcome Analysis OR Extraction                 │
│                                                                 │
│  Step 3: Validate and deduplicate                              │
│  └── Target: 8,000 high-quality examples                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Claude API Prompts

#### Prompt 1: Case Summarization
```
Read this Indian High Court judgment and provide a structured summary.

JUDGMENT:
{full_text_truncated_to_6000_chars}

Provide a summary with these sections:
1. **Case Type**: [e.g., Bail Application, Civil Appeal, Writ Petition]
2. **Parties**: [Petitioner/Appellant vs Respondent]
3. **Key Facts**: [2-3 sentences on background]
4. **Legal Issues**: [Main questions before the court]
5. **Decision**: [What the court ordered]
6. **Reasoning**: [Why the court decided this way]

Keep the summary under 300 words.
```

#### Prompt 2: Legal Research Q&A
```
Based on this Indian High Court judgment, generate a legal research question and detailed answer.

JUDGMENT:
{full_text_truncated_to_6000_chars}

Generate:
1. A specific legal question that a lawyer might ask when researching this area of law
2. A detailed answer (150-250 words) that cites this case and explains the legal principle

Format:
QUESTION: [Your question]
ANSWER: [Your detailed answer citing this case]
```

#### Prompt 3: Outcome Analysis
```
Analyze why the court reached its decision in this case.

JUDGMENT:
{full_text_truncated_to_6000_chars}

CASE OUTCOME: {disposal_nature}

Explain:
1. What factors did the court consider?
2. What legal principles or precedents were applied?
3. Why did these factors lead to this specific outcome?

Keep your analysis under 200 words.
```

#### Prompt 4: Information Extraction
```
Extract key information from this Indian High Court judgment.

JUDGMENT:
{full_text_truncated_to_4000_chars}

Extract and format:
1. **Case Number**: [exact case number]
2. **Court**: [full court name and bench]
3. **Date**: [decision date]
4. **Petitioner/Appellant**: [name]
5. **Respondent**: [name]
6. **Judge(s)**: [names]
7. **Case Type**: [e.g., Criminal Appeal, Writ Petition, Bail Application]
8. **Key Statutes**: [any acts or sections mentioned]
9. **Outcome**: [disposed/allowed/dismissed/etc.]
```

### Pipeline Steps
```
Step 1: Quality Filter & Sampling
├── Filter: 500 < word_count < 15,000
├── Filter: full_text IS NOT NULL
├── Filter: decision_date IS NOT NULL
├── Sample: 4,000 documents (balanced by court and case type)
└── Store sampled CNRs for reproducibility

Step 2: Light Text Preprocessing
├── Normalize whitespace
├── Remove page number artifacts
└── Truncate to fit Claude context (max 6000 chars for input)

Step 3: LLM Generation (Claude API)
├── For each document, randomly select 2 prompt types
├── Call Claude API with appropriate prompt
├── Parse response into instruction/input/output format
├── Handle rate limits and errors gracefully
├── Save progress incrementally (resume support)
└── Target: 8,000 examples from 4,000 documents

Step 4: Post-processing
├── Validate JSON structure
├── Remove failed/empty generations
├── Deduplicate similar examples
└── Shuffle all examples

Step 5: Split & Export
├── Split 90/10 (train/val)
├── Export to train.jsonl and val.jsonl
└── Report statistics
```

### Commands
```bash
# Set Claude API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Full pipeline
python scripts/prepare_training_data.py

# Specify number of examples
python scripts/prepare_training_data.py --target-examples 8000

# Limit source documents (for testing)
python scripts/prepare_training_data.py --limit 100

# Resume interrupted generation
python scripts/prepare_training_data.py --resume

# Dry run (show plan, estimate cost)
python scripts/prepare_training_data.py --dry-run

# Specify output directory
python scripts/prepare_training_data.py --output-dir ./data/training

# Verbose logging
python scripts/prepare_training_data.py --verbose

# Validate generated data
python scripts/validate_training_data.py --input-dir ./data/training

# Show sample examples
python scripts/validate_training_data.py --input-dir ./data/training --show-samples 5
```

### Script Arguments

#### prepare_training_data.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--target-examples` | int | 8000 | Total training examples to generate |
| `--output-dir` | path | `./data/training` | Output directory for JSONL files |
| `--limit` | int | None | Limit source documents (for testing) |
| `--resume` | flag | False | Resume from last checkpoint |
| `--dry-run` | flag | False | Show plan and cost estimate without executing |
| `--verbose` | flag | False | Enable debug logging |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--batch-size` | int | 50 | Documents to process before saving checkpoint |

### Quality Filter Criteria

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Minimum word count | 500 words | Too short = insufficient content |
| Maximum word count | 15,000 words | Longer docs truncated anyway |
| Has full_text | NOT NULL | Required for generation |
| Has decision_date | NOT NULL | Useful metadata |

### Sampling Strategy

| Criterion | Target |
|-----------|--------|
| Total documents | 4,000 (generates ~8K examples) |
| Court balance | 50% Delhi HC, 50% Bombay HC |
| Case type diversity | Proportional to disposal_nature distribution |

### Cost Estimation

| Item | Calculation | Estimate |
|------|-------------|----------|
| Input tokens | 4000 docs × ~2000 tokens × 2 calls | ~16M input tokens |
| Output tokens | 8000 examples × ~300 tokens | ~2.4M output tokens |
| Claude Sonnet pricing | $3/M input, $15/M output | ~$48 + ~$36 = **~$85** |
| Claude Haiku pricing | $0.25/M input, $1.25/M output | ~$4 + ~$3 = **~$7** |

**Recommendation**: Use Claude Haiku for cost efficiency (~$7-10 total)

### LLM Generator Implementation Notes

#### Class: LLMGenerator
```python
class LLMGenerator:
    """Generate training data using Claude API."""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def generate_summarization(self, doc: AWSHighCourtDocument) -> TrainingExample
    def generate_research_qa(self, doc: AWSHighCourtDocument) -> TrainingExample
    def generate_outcome_analysis(self, doc: AWSHighCourtDocument) -> TrainingExample
    def generate_extraction(self, doc: AWSHighCourtDocument) -> TrainingExample
    
    def generate_examples_for_document(self, doc: AWSHighCourtDocument, num_examples: int = 2) -> List[TrainingExample]
```

#### Rate Limiting & Error Handling

- Implement exponential backoff for rate limits
- Save progress every `batch-size` documents
- Log failures but continue processing
- Retry failed documents at the end

#### Progress Tracking

Store in SQLite table `training_generation_progress`:

| Column | Type | Description |
|--------|------|-------------|
| cnr | TEXT | Document CNR |
| prompt_type | TEXT | summarization/research_qa/outcome/extraction |
| status | TEXT | pending/success/failed |
| example_json | TEXT | Generated example (if success) |
| error | TEXT | Error message (if failed) |
| created_at | DATETIME | Timestamp |

### Output Format

#### train.jsonl / val.jsonl
```json
{"instruction": "Summarize this Indian High Court judgment.", "input": "IN THE HIGH COURT OF JUDICATURE AT BOMBAY...", "output": "**Case Type**: Bail Cancellation Application\n**Parties**: X.Y.Z. vs State of Maharashtra...\n**Decision**: Application dismissed..."}
{"instruction": "What legal principle governs anticipatory bail in economic offenses?", "input": "", "output": "In the case of ACB.53.2024 before the Bombay High Court, the court addressed..."}
```

### Success Criteria

| Metric | Target |
|--------|--------|
| Total examples | 8,000 |
| Train split | 7,200 (90%) |
| Val split | 800 (10%) |
| Valid JSON | 100% |
| Non-empty outputs | 100% |
| Example types | 4 distinct types |
| Generation success rate | >95% |

### Dependencies (Add to requirements.txt)
```
# For LLM generation
anthropic>=0.18.0
```

### Environment Variables
```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional
ANTHROPIC_MODEL=claude-3-haiku-20240307  # or claude-3-sonnet-20240229
```
