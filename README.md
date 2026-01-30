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
│   └── settings.py             # All configuration (URLs, limits, paths)
├── scrapers/                   # Web scrapers (alternative data sources)
│   ├── __init__.py
│   ├── base_scraper.py         # Abstract base class with rate limiting
│   ├── indian_kanoon.py        # Indian Kanoon scraper
│   ├── supreme_court.py        # Supreme Court scraper
│   ├── high_courts.py          # High Courts scraper
│   └── india_code.py           # India Code (statutes) scraper
├── processors/
│   ├── __init__.py
│   └── pdf_extractor.py        # PDF text extraction using PyPDF2
├── storage/
│   ├── __init__.py
│   ├── document_store.py       # SQLite + JSON storage (for scrapers)
│   ├── schemas.py              # Pydantic models (for scrapers)
│   ├── aws_document_store.py   # SQLite storage for AWS data (PRIMARY)
│   └── aws_schemas.py          # Pydantic models for AWS data
├── utils/
│   ├── __init__.py
│   ├── rate_limiter.py         # Respectful crawling
│   ├── retry.py                # Exponential backoff
│   └── logger.py               # Logging setup
├── scripts/
│   ├── run_collection.py       # Main entry point (for scrapers)
│   ├── load_aws_metadata.py    # Load AWS parquet metadata → DB
│   └── process_aws_pdfs.py     # Extract PDF text → DB
├── tests/
│   ├── __init__.py
│   ├── test_scrapers.py
│   └── test_storage.py
├── data/                       # Collected data (gitignored)
│   ├── raw/                    # Raw scraped documents
│   ├── processed/              # Cleaned documents
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

### External Dependencies (Not in Repository)

**llama.cpp** - Required for GGUF model conversion

This is an external dependency that is **NOT included** in the repository. You must clone and build it separately:

```bash
# Automated setup (recommended)
bash scripts/setup_llama_cpp.sh

# OR manual setup
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-quantize -j 8
```

**Why not included?**
- External project with its own version control
- Contains platform-specific binaries (~500 MB)
- Users should get the latest version directly

**When needed?**
- Only for Phase 3: Converting trained models to GGUF format
- Not required for Phase 1 (data collection) or Phase 2 (training data generation)

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

### Phase 2: Data Processing (CURRENT)

| Task | Status |
|------|--------|
| Clean extracted text | TODO |
| Generate training data format | TODO |
| Create train/validation splits | TODO |
| Export for model training | TODO |

### Phase 3: Model Training (LATER)

| Task | Status |
|------|--------|
| Fine-tune Llama 3.2 3B | TODO |
| Quantization for M2 | TODO |
| Evaluation | TODO |
| Deployment | TODO |

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
