# Nyay AI India

## Quick Reference

| Aspect | Value |
|--------|-------|
| **Goal** | Build legal (Nyay) AI for Indian law using Llama 3.2 3B |
| **Current Phase** | Phase 2: Data Processing |
| **Documents Collected** | 58,222 (57,398 with full text) |
| **Primary Source** | AWS Open Data - Indian High Court Judgments |
| **Language** | Python 3.11+ |
| **Owner** | Solo developer |
| **Hardware** | MacBook Pro M2, 32GB RAM |

---

## Project Structure

```
nyay-ai-india/
├── CLAUDE.md                   # This file - project context (auto-read)
├── config/
│   ├── settings.py             # All configuration (URLs, limits, paths)
│   └── llm_prompts.py          # Phase 2: Prompt templates for 4 task types
├── scrapers/                   # Web scrapers (alternative data sources)
│   ├── __init__.py
│   ├── base_scraper.py         # Abstract base class with rate limiting
│   ├── indian_kanoon.py        # Indian Kanoon scraper
│   ├── supreme_court.py        # Supreme Court scraper
│   ├── high_courts.py          # High Courts scraper
│   └── india_code.py           # India Code (statutes) scraper
├── processors/
│   ├── __init__.py
│   ├── pdf_extractor.py        # PDF text extraction using PyPDF2
│   ├── text_cleaner.py         # Phase 2: Text preprocessing for LLM input
│   ├── quality_filter.py       # Phase 2: Document filtering & sampling
│   └── llm_generator.py        # Phase 2: Claude API integration with validation
├── storage/
│   ├── __init__.py
│   ├── document_store.py       # SQLite + JSON storage (for scrapers)
│   ├── schemas.py              # Pydantic models (for scrapers)
│   ├── aws_document_store.py   # SQLite storage for AWS data (PRIMARY)
│   ├── aws_schemas.py          # Pydantic models for AWS data
│   ├── training_schemas.py     # Phase 2: Training data Pydantic models
│   └── training_store.py       # Phase 2: Training data DB operations + JSONL export
├── utils/
│   ├── __init__.py
│   ├── rate_limiter.py         # Respectful crawling
│   ├── retry.py                # Exponential backoff
│   └── logger.py               # Logging setup
├── scripts/
│   ├── run_collection.py       # Main entry point (for scrapers)
│   ├── load_aws_metadata.py    # Load AWS parquet metadata → DB
│   ├── process_aws_pdfs.py     # Extract PDF text → DB
│   ├── prepare_training_data.py        # Phase 2: Generate training examples (MAIN)
│   ├── validate_training_data.py       # Phase 2: Validate JSONL output
│   ├── automated_quality_checks.py     # Phase 2: Stage 2 automated validation
│   ├── manual_review_helper.py         # Phase 2: Visual side-by-side review
│   ├── interactive_review.py           # Phase 2: Interactive review with Q&A
│   └── filter_bad_examples.py          # Phase 2: Remove bad CNRs from JSONL
├── docs/                       # Phase 2: Documentation
│   ├── PHASE2_DATA_PROCESSING.md       # Phase 2 specification
│   ├── DATA_QUALITY_VERIFICATION.md    # Quality verification guide
│   ├── BAD_TRAINING_DATA_IMPACT.md     # Impact of bad training data
│   ├── TWO_STAGE_VALIDATION.md         # Validation strategy explained
│   └── MANUAL_REVIEW_GUIDE.md          # Manual review instructions
├── tests/
│   ├── __init__.py
│   ├── test_scrapers.py
│   └── test_storage.py
├── data/                       # Collected data (gitignored)
│   ├── raw/                    # Raw scraped documents
│   ├── processed/              # Cleaned documents
│   ├── training/               # Phase 2: Training data (JSONL files)
│   │   ├── train.jsonl         # 90% training examples (~7,200)
│   │   ├── val.jsonl           # 10% validation examples (~800)
│   │   ├── review_results.json # Manual review results
│   │   ├── bad_cnrs.txt        # Bad CNRs to filter
│   │   ├── train_clean.jsonl   # Cleaned training set
│   │   └── val_clean.jsonl     # Cleaned validation set
│   ├── aws_data/               # AWS dataset (parquet + tar files)
│   │   ├── data/year=2025/     # Parquet metadata files
│   │   └── tar/year=2025/      # PDF tar files
│   └── metadata.db             # SQLite database
├── logs/                       # Log files (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data Sources

### Primary Source: AWS Open Data (Indian High Court Judgments)

| Aspect | Details |
|--------|---------|
| **Source** | [AWS Open Data Registry](https://registry.opendata.aws/indian-high-court-judgments/) |
| **Repository** | [vanga/indian-high-court-judgments](https://github.com/vanga/indian-high-court-judgments) |
| **Total Available** | ~16.7 million judgments from 25 High Courts |
| **Format** | Parquet (metadata) + PDF (judgments in tar archives) |
| **Update Frequency** | Quarterly |

### Currently Loaded Data (Year 2025)

| Court | Documents | With Full Text |
|-------|-----------|----------------|
| High Court of Delhi | 28,173 | ~28,000 |
| Bombay High Court (Aurangabad) | 30,049 | ~29,400 |
| **Total** | **58,222** | **57,398** |

### Alternative Sources (Scrapers - Available but not primary)

| Source | URL | Scraper | Notes |
|--------|-----|---------|-------|
| Indian Kanoon | indiankanoon.org | `indian_kanoon.py` | Case law aggregator |
| Supreme Court | main.sci.gov.in | `supreme_court.py` | Via Indian Kanoon |
| High Courts | Various | `high_courts.py` | Via Indian Kanoon |
| India Code | indiacode.nic.in | `india_code.py` | Central Acts & Statutes |

---

## Data Model

### AWSHighCourtDocument Schema (Primary)

```python
class AWSHighCourtDocument(BaseModel):
    # Primary Identifier
    cnr: str                          # eCourts unique ID (e.g., HCBM030079862025)
    doc_id: str                       # Generated hash for compatibility

    # From Parquet (Direct mapping)
    court_code: str                   # e.g., "27~1"
    title: str                        # Full title with case number + parties
    description: str | None           # Truncated summary (~300 chars)
    judge: str | None                 # Raw judge string
    pdf_link: str | None              # Relative path to PDF
    date_of_registration: date | None
    decision_date: date | None
    disposal_nature: str | None       # e.g., "DISPOSED OFF"
    court: str                        # e.g., "Bombay High Court"

    # PDF Extracted Content
    full_text: str | None             # Extracted from PDF
    pdf_processed: bool               # Track processing status

    # Metadata
    year: int                         # Partition key (e.g., 2025)
    bench: str                        # Source bench (e.g., "hcaurdb")
    created_at: datetime
    word_count: int | None            # Computed from full_text
```

### Database Tables

| Table | Purpose | Primary Key |
|-------|---------|-------------|
| `aws_documents` | AWS High Court judgments | `cnr` |
| `aws_processing_progress` | PDF processing progress tracking | `id` (court:bench:year) |
| `documents` | Scraped documents (alternative sources) | `doc_id` |
| `scraping_progress` | Scraper progress tracking | `source` |
| `training_examples` | Phase 2: Generated training examples | `id` |
| `training_generation_progress` | Phase 2: Per-document generation progress | `cnr` |
| `training_run_metadata` | Phase 2: Generation run tracking & costs | `run_id` |

---

## Commands

### AWS Data Pipeline (Primary)

```bash
# 1. Download AWS metadata (parquet files) - run in terminal
aws s3 sync s3://indian-high-court-judgments/metadata/parquet/year=2025 \
    ./data/aws_data/data/year=2025 --no-sign-request

# 2. Download AWS PDFs (tar files) for specific courts
aws s3 sync s3://indian-high-court-judgments/data/tar/year=2025/court=7_26/bench=dhcdb \
    ./data/aws_data/tar/year=2025/court=7_26/bench=dhcdb --no-sign-request

aws s3 sync s3://indian-high-court-judgments/data/tar/year=2025/court=27_1/bench=hcaurdb \
    ./data/aws_data/tar/year=2025/court=27_1/bench=hcaurdb --no-sign-request

# 3. Load metadata into database
python scripts/load_aws_metadata.py --data-dir ./data/aws_data/data

# Load specific court only
python scripts/load_aws_metadata.py --data-dir ./data/aws_data/data --court 7_26 --bench dhcdb

# Dry run (see what would be loaded)
python scripts/load_aws_metadata.py --data-dir ./data/aws_data/data --dry-run

# 4. Process PDFs (extract text)
python scripts/process_aws_pdfs.py --tar-dir ./data/aws_data/tar

# Process specific court
python scripts/process_aws_pdfs.py --tar-dir ./data/aws_data/tar --court 7_26 --bench dhcdb

# Process with limit
python scripts/process_aws_pdfs.py --tar-dir ./data/aws_data/tar --limit 50000

# Resume interrupted processing
python scripts/process_aws_pdfs.py --tar-dir ./data/aws_data/tar --resume

# Dry run
python scripts/process_aws_pdfs.py --tar-dir ./data/aws_data/tar --dry-run
```

### Database Queries

```bash
# Check document counts
sqlite3 data/metadata.db "SELECT court, COUNT(*) FROM aws_documents GROUP BY court"

# Check processing status
sqlite3 data/metadata.db "SELECT
    COUNT(*) as total,
    SUM(CASE WHEN pdf_processed = 1 THEN 1 ELSE 0 END) as with_text,
    SUM(CASE WHEN pdf_processed = 0 THEN 1 ELSE 0 END) as without_text
FROM aws_documents"

# Sample document with full text
sqlite3 data/metadata.db "SELECT cnr, title, word_count FROM aws_documents WHERE full_text IS NOT NULL LIMIT 5"
```

### Alternative: Web Scrapers

```bash
# Run Indian Kanoon scraper
python scripts/run_collection.py --source indian_kanoon --target 5000

# Dry run (10 documents)
python scripts/run_collection.py --source indian_kanoon --dry-run

# Resume interrupted collection
python scripts/run_collection.py --resume
```

### Phase 2: Training Data Generation

```bash
# 1. Generate training examples (8,000 from 4,000 documents)
python scripts/prepare_training_data.py

# Test with small batch first
python scripts/prepare_training_data.py --limit 10  # 20 examples

# Dry run (see plan without API calls)
python scripts/prepare_training_data.py --dry-run

# Resume if interrupted
python scripts/prepare_training_data.py --resume

# With cost limit
python scripts/prepare_training_data.py --cost-limit 15.0

# 2. Validate generated data
python scripts/validate_training_data.py --input-dir ./data/training

# 3. Stage 2 Quality Checks (automated)
python scripts/automated_quality_checks.py

# 4. Manual Review (visual - side by side)
python scripts/manual_review_helper.py --sample 20

# Manual review (interactive - with Q&A)
python scripts/interactive_review.py --sample 20
python scripts/interactive_review.py --split val  # Review validation set
python scripts/interactive_review.py --cnr DLHC010011762025  # Specific CNR

# Resume interrupted review session
python scripts/interactive_review.py --sample 500 \
  --continue-session data/training/review_results.json

# 5. Filter out bad examples
python scripts/filter_bad_examples.py \
  --input data/training/train.jsonl \
  --remove data/training/bad_cnrs.txt \
  --output data/training/train_clean.jsonl
```

### Phase 2: Database Queries

```bash
# Check training data counts
sqlite3 data/metadata.db "SELECT
    split,
    task_type,
    COUNT(*) as count
FROM training_examples
GROUP BY split, task_type
ORDER BY split, task_type"

# Check generation progress
sqlite3 data/metadata.db "SELECT
    status,
    COUNT(*) as count,
    SUM(examples_generated) as total_examples
FROM training_generation_progress
GROUP BY status"

# Check run statistics
sqlite3 data/metadata.db "SELECT
    run_id,
    started_at,
    documents_processed,
    examples_generated,
    estimated_cost
FROM training_run_metadata
ORDER BY started_at DESC
LIMIT 5"

# Sample generated examples
sqlite3 data/metadata.db "SELECT
    cnr,
    task_type,
    LENGTH(output) as output_length
FROM training_examples
LIMIT 5"
```

### Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Test imports
python -c "from storage import AWSDocumentStore, AWSHighCourtDocument; print('OK')"

# Run all tests
pytest tests/ -v
```

---

## Key Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `requests` | >=2.31.0 | HTTP requests |
| `beautifulsoup4` | >=4.12.0 | HTML parsing |
| `lxml` | >=5.0.0 | Fast HTML parser |
| `pydantic` | >=2.5.0 | Data validation |
| `sqlite-utils` | >=3.35.0 | SQLite wrapper |
| `tenacity` | >=8.2.0 | Retry logic |
| `tqdm` | >=4.66.0 | Progress bars |
| `pyarrow` | >=14.0.0 | Parquet file reading |
| `pandas` | >=2.0.0 | DataFrame operations |
| `PyPDF2` | >=3.0.0 | PDF text extraction |
| `anthropic` | >=0.18.0 | Phase 2: Claude API client |

---

## Coding Conventions

### Python Style
- Python 3.11+ features (use `|` for union types, etc.)
- Type hints for ALL function signatures
- Pydantic v2 for data models
- f-strings for string formatting
- `pathlib.Path` instead of `os.path`

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `AWSDocumentStore` |
| Functions | snake_case | `extract_text_from_pdf()` |
| Variables | snake_case | `doc_count` |
| Constants | UPPER_SNAKE_CASE | `MAX_RETRIES` |
| Files | snake_case | `aws_document_store.py` |
| Private | leading underscore | `_clean_text()` |

### Logging Pattern
```python
from utils.logger import get_logger

logger = get_logger(__name__)

logger.debug("Detailed info for debugging")
logger.info("Normal operations: Loaded 1000 documents")
logger.warning("Recoverable issues: PDF extraction failed")
logger.error("Failures needing attention: Database error")
```

---

## Phase Status

### Phase 1: Data Collection ✅ COMPLETE

| Task | Status |
|------|--------|
| Set up project structure | ✅ Done |
| Implement AWS data loader | ✅ Done |
| Implement PDF text extraction | ✅ Done |
| Load 58,222 documents | ✅ Done |
| Extract text from 57,398 PDFs | ✅ Done |
| Web scrapers (alternative) | ✅ Done |

### Phase 2: Data Processing ✅ IMPLEMENTATION COMPLETE

### Overview
| Aspect | Value |
|--------|-------|
| **Input** | 57,398 documents with full_text |
| **Eligible for training** | 26,400 (after quality filtering) |
| **Selected** | 4,000 documents (balanced: 2K Delhi HC, 2K Bombay HC) |
| **Output** | 8,000 training examples (2 per document) |
| **Method** | LLM-based generation (Claude Haiku API) |
| **Estimated Cost** | ~$13 for full run (tested: $0.02 for 10 docs) |
| **Time** | ~2.7 hours at 50 RPM |

### Task Types (Equal Distribution)
| Task Type | Count | Purpose |
|-----------|-------|---------|
| Summarization | 2,000 | Structured summaries of judgments |
| Research Q&A | 2,000 | Legal questions + detailed answers |
| Outcome Analysis | 2,000 | Explain reasoning behind court decisions |
| Info Extraction | 2,000 | Extract parties, statutes, precedents, relief |

### Quick Commands
```bash
# 1. Generate training data (full run)
python scripts/prepare_training_data.py

# Test with small batch
python scripts/prepare_training_data.py --limit 10

# Dry run (see plan)
python scripts/prepare_training_data.py --dry-run

# Resume if interrupted
python scripts/prepare_training_data.py --resume

# 2. Validate output
python scripts/validate_training_data.py --input-dir ./data/training

# 3. Quality verification (2-stage validation)
python scripts/automated_quality_checks.py  # Stage 2: Automated
python scripts/interactive_review.py --sample 20  # Stage 2: Manual

# 4. Filter bad examples
python scripts/filter_bad_examples.py \
  --input data/training/train.jsonl \
  --remove data/training/bad_cnrs.txt \
  --output data/training/train_clean.jsonl
```

### Implementation Status

| Component | File | Status |
|-----------|------|--------|
| **Configuration** | `config/settings.py` | ✅ Updated with LLM settings |
| **Prompts** | `config/llm_prompts.py` | ✅ Created (4 task types) |
| **Schemas** | `storage/training_schemas.py` | ✅ Created (Pydantic v2) |
| **Storage** | `storage/training_store.py` | ✅ Created (DB + JSONL export) |
| **Text Cleaner** | `processors/text_cleaner.py` | ✅ Created |
| **Quality Filter** | `processors/quality_filter.py` | ✅ Created |
| **LLM Generator** | `processors/llm_generator.py` | ✅ Created (with Stage 1 validation) |
| **Main Script** | `scripts/prepare_training_data.py` | ✅ Created (resume, cost tracking) |
| **Validator** | `scripts/validate_training_data.py` | ✅ Created |
| **Quality Checks** | `scripts/automated_quality_checks.py` | ✅ Created (Stage 2 automated) |
| **Manual Review** | `scripts/manual_review_helper.py` | ✅ Created (visual) |
| **Interactive Review** | `scripts/interactive_review.py` | ✅ Created (Q&A) |
| **Filter** | `scripts/filter_bad_examples.py` | ✅ Created |
| **Dependencies** | `requirements.txt` | ✅ Added anthropic>=0.18.0 |

### Task Status
| Task | Status |
|------|--------|
| Design prompts for 4 task types | ✅ Done |
| Implement text preprocessing | ✅ Done |
| Implement quality filtering | ✅ Done |
| Implement LLM generator | ✅ Done (with validation) |
| Create main orchestration script | ✅ Done (with resume) |
| Add progress tracking | ✅ Done (SQLite-based) |
| Add cost tracking | ✅ Done (real-time) |
| Implement JSONL export | ✅ Done |
| Create validation script | ✅ Done |
| Create quality verification tools | ✅ Done (2-stage validation) |
| Test with 10 documents | ✅ Done (generated 20 examples, $0.02) |
| **Full run (4,000 documents)** | ⏳ Ready to run |

### Two-Stage Validation Strategy

**Stage 1: Built-in (During Generation)**
- Location: `processors/llm_generator.py` → `_is_valid_output()`
- Checks: Empty output, length, refusals, format, legal terminology
- Saves: ~$1+ in API costs by rejecting bad outputs immediately

**Stage 2: Post-Generation (Separate Scripts)**
- Automated: `automated_quality_checks.py` (hallucination, repetition, format)
- Manual: `interactive_review.py` (5-question checklist per example)
- Filter: `filter_bad_examples.py` (remove bad CNRs)

### Documentation
| Doc | Purpose |
|-----|---------|
| `docs/PHASE2_DATA_PROCESSING.md` | Detailed Phase 2 specification |
| `docs/DATA_QUALITY_VERIFICATION.md` | Quality verification guide |
| `docs/BAD_TRAINING_DATA_IMPACT.md` | Impact of bad training data |
| `docs/TWO_STAGE_VALIDATION.md` | Validation strategy explained |
| `docs/MANUAL_REVIEW_GUIDE.md` | Manual review instructions |

### Output Format (JSONL)

**Location:** `data/training/`

**train.jsonl** (90% = ~7,200 examples)
```json
{
  "instruction": "Analyze the outcome of this judgment...",
  "input": "[Full judgment text...]",
  "output": "[Generated analysis...]",
  "metadata": {
    "cnr": "HCBM030212662025",
    "task_type": "outcome_analysis",
    "court": "Bombay High Court",
    "word_count": 1543,
    "input_tokens": 2847,
    "output_tokens": 421
  }
}
```

**val.jsonl** (10% = ~800 examples)
- Same format as train.jsonl
- Used for validation during fine-tuning (Phase 3)

### Next Steps
1. ✅ **Implementation complete** - All files created and tested
2. ⏳ **Ready for full run** - Generate 8,000 examples (~$13, 2.7 hours)
3. ⏳ **Quality verification** - Review generated examples
4. ⏳ **Phase 3** - Fine-tune Llama 3.2 3B with clean data

### Phase 3: Model Training (NEXT)

### Overview
| Aspect | Value |
|--------|-------|
| **Base Model** | Llama 3.2 3B Instruct |
| **Method** | QLoRA 8-bit (best quality/memory trade-off) |
| **Framework** | MLX (Apple Silicon optimized) |
| **Training Data** | 8,000 examples (7,200 train + 800 val) |
| **Hardware** | MacBook Pro M2, 32GB RAM |
| **Memory Usage** | ~8 GB (comfortable on 32GB) |
| **Training Time** | 4-8 hours |
| **Output** | Fine-tuned Nyay AI model |

### Why QLoRA 8-bit?
| Factor | Benefit |
|--------|---------|
| **Memory** | 8GB vs 28GB (full fine-tune) - fits on M2 |
| **Quality** | 97-99% of full fine-tune (only 0.5-1% loss) |
| **Speed** | Trains only 0.3% of parameters (8M vs 3B) |
| **Legal domain** | Higher precision important for accuracy |

### Quick Commands
```bash
# 1. Setup training environment
python3 -m venv .venv-train && source .venv-train/bin/activate
pip install -r requirements-training.txt

# 2. Convert data to MLX format
python scripts/convert_to_mlx_format.py

# 3. Train model (4-8 hours)
python scripts/train_model.py

# 4. Evaluate
python scripts/evaluate_model.py

# 5. Export to GGUF for deployment
python scripts/export_model.py --format gguf --quantize q4_k_m

# 6. Run with Ollama
ollama create nyay-ai -f Modelfile
ollama run nyay-ai
```

### Files to Create
| File | Purpose |
|------|---------|
| `config/training_config.yaml` | Training hyperparameters |
| `scripts/convert_to_mlx_format.py` | JSONL → MLX chat format |
| `scripts/train_model.py` | Main training script |
| `scripts/evaluate_model.py` | Evaluation metrics |
| `scripts/export_model.py` | Export to GGUF |
| `requirements-training.txt` | Training dependencies |

### Task Status
| Task | Status |
|------|--------|
| Plan Phase 3 | ✅ Done |
| Setup training environment | TODO |
| Convert training data format | TODO |
| Download Llama 3.2 3B | TODO |
| Train with QLoRA 8-bit | TODO |
| Evaluate model | TODO |
| Export & quantize | TODO |
| Deploy with Ollama | TODO |

### Success Criteria (Phase 3)
| Metric | Target |
|--------|--------|
| Training loss | < 1.0 |
| Validation loss | < 1.2 |
| Summarization quality | > 70% human approval |
| Q&A accuracy | > 75% correct |
| Hallucination rate | < 5% |
| Inference speed | > 20 tokens/sec |

**Detailed specification**: See `docs/PHASE3_MODEL_TRAINING.md`

---

## Success Criteria (Phase 1) ✅

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Total documents | 50,000+ | 58,222 | ✅ |
| Documents with full text | >95% | 98.6% (57,398) | ✅ |
| Courts covered | 2+ | 2 (Delhi, Bombay) | ✅ |
| Year coverage | 2025 | 2025 | ✅ |

---

## Changelog

| Date | Change |
|------|--------|
| 2025-01-14 | Initial project setup |
| 2025-01-22 | Added AWS data pipeline (load_aws_metadata.py, process_aws_pdfs.py) |
| 2025-01-22 | Loaded 58,222 documents from Delhi HC and Bombay HC |
| 2025-01-22 | Extracted text from 57,398 PDFs |
| 2025-01-22 | Phase 1 (Data Collection) complete |
| 2025-01-25 | Phase 2 implementation complete |
| 2025-01-25 | Created config/llm_prompts.py (4 task type prompts) |
| 2025-01-25 | Created processors/ (text_cleaner, quality_filter, llm_generator) |
| 2025-01-25 | Created storage/ (training_schemas, training_store) |
| 2025-01-25 | Created scripts/ (prepare_training_data, validate, quality checks, review tools) |
| 2025-01-25 | Created docs/ (5 documentation files for Phase 2) |
| 2025-01-25 | Tested with 10 documents: generated 20 examples ($0.02) |
| 2025-01-25 | Implemented two-stage validation (built-in + post-generation) |
| 2025-01-25 | Ready for full run: 4,000 docs → 8,000 examples (~$13) |
| 2025-01-26 | Started full generation run (4,000 documents) |
| 2025-01-26 | Planned Phase 3: QLoRA 8-bit fine-tuning with MLX |
| 2025-01-26 | Created docs/PHASE3_MODEL_TRAINING.md |
