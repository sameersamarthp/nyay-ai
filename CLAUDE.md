# Nyay AI India

## Quick Reference

| Aspect | Value |
|--------|-------|
| **Goal** | Build legal (Nyay) AI for Indian law using Llama 3.2 3B |
| **Current Phase** | Phase 1: Data Collection |
| **Target** | 10,000 legal documents |
| **Language** | Python 3.11+ |
| **Owner** | Solo developer, 2-4 week timeline |
| **Hardware** | MacBook Pro M2, 32GB RAM |

---

## Project Structure

```
nyay-ai-india/
├── CLAUDE.md                   # This file - project context (auto-read)
├── config/
│   └── settings.py             # All configuration (URLs, limits, paths)
├── scrapers/
│   ├── __init__.py
│   ├── base_scraper.py         # Abstract base class with rate limiting
│   ├── indian_kanoon.py        # Primary source (5,000 docs)
│   ├── supreme_court.py        # SCI judgments (2,000 docs)
│   ├── high_courts.py          # HC judgments (2,000 docs)
│   └── india_code.py           # Statutes (1,000 docs)
├── processors/
│   ├── __init__.py
│   ├── cleaner.py              # HTML cleaning, text normalization
│   ├── metadata_extractor.py   # Extract case metadata
│   └── deduplicator.py         # Remove duplicate documents
├── storage/
│   ├── __init__.py
│   ├── document_store.py       # SQLite + JSON storage
│   └── schemas.py              # Pydantic models
├── utils/
│   ├── __init__.py
│   ├── rate_limiter.py         # Respectful crawling
│   ├── retry.py                # Exponential backoff
│   └── logger.py               # Logging setup
├── scripts/
│   ├── run_collection.py       # Main entry point
│   ├── validate_data.py        # Data quality checks
│   └── export_for_training.py  # Prepare for Phase 2
├── tests/
│   ├── __init__.py
│   ├── test_scrapers.py
│   └── test_storage.py
├── data/                       # Collected data (gitignored)
│   ├── raw/                    # Raw scraped documents
│   ├── processed/              # Cleaned documents
│   └── metadata.db             # SQLite database
├── logs/                       # Log files (gitignored)
├── docs/                       # Detailed specifications
│   ├── PHASE1_DATA_COLLECTION.md
│   ├── PHASE2_DATA_PROCESSING.md
│   └── ARCHITECTURE.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Coding Conventions

### Python Style
- Python 3.11+ features (use `|` for union types, etc.)
- Type hints for ALL function signatures
- Pydantic v2 for data models
- f-strings for string formatting
- `pathlib.Path` instead of `os.path`
- `httpx` or `requests` for HTTP (requests is fine)

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `IndianKanoonScraper` |
| Functions | snake_case | `fetch_page()` |
| Variables | snake_case | `doc_count` |
| Constants | UPPER_SNAKE_CASE | `MAX_RETRIES` |
| Files | snake_case | `indian_kanoon.py` |
| Private | leading underscore | `_parse_html()` |

### Error Handling
```python
# DO: Specific exceptions with context
try:
    response = self.fetch_page(url)
except requests.Timeout as e:
    logger.warning(f"Timeout fetching {url}: {e}")
    raise
except requests.RequestException as e:
    logger.error(f"Failed to fetch {url}: {e}")
    return None

# DON'T: Bare except
try:
    ...
except:  # Never do this
    pass
```

### Logging Pattern
```python
from utils.logger import get_logger

logger = get_logger(__name__)

# Use appropriate levels
logger.debug("Detailed info for debugging")
logger.info("Normal operations: Fetched page 5")
logger.warning("Recoverable issues: Retrying after 429")
logger.error("Failures needing attention: Parse failed")
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
| `fake-useragent` | >=1.4.0 | User agent rotation |
| `python-dateutil` | >=2.8.0 | Date parsing |

---

## Current Task Queue

### Phase 1: Data Collection (CURRENT)

| Priority | Task | Status | File |
|----------|------|--------|------|
| P0 | Create directory structure | TODO | - |
| P0 | Create requirements.txt | TODO | `requirements.txt` |
| P0 | Create settings config | TODO | `config/settings.py` |
| P0 | Create Pydantic schemas | TODO | `storage/schemas.py` |
| P0 | Create document store | TODO | `storage/document_store.py` |
| P0 | Create utils (logger, rate_limiter, retry) | TODO | `utils/` |
| P0 | Create base scraper | TODO | `scrapers/base_scraper.py` |
| P0 | Implement Indian Kanoon scraper | TODO | `scrapers/indian_kanoon.py` |
| P1 | Implement Supreme Court scraper | TODO | `scrapers/supreme_court.py` |
| P1 | Implement High Courts scraper | TODO | `scrapers/high_courts.py` |
| P1 | Implement India Code scraper | TODO | `scrapers/india_code.py` |
| P1 | Create main run script | TODO | `scripts/run_collection.py` |
| P2 | Create validation script | TODO | `scripts/validate_data.py` |
| P2 | Write tests | TODO | `tests/` |

### Phase 2: Data Processing (NEXT)
See `docs/PHASE2_DATA_PROCESSING.md`

### Phase 3: Model Training (LATER)
See `docs/PHASE3_MODEL_TRAINING.md`

---

## Scraping Targets

| Source | URL | Target | Notes |
|--------|-----|--------|-------|
| Indian Kanoon | indiankanoon.org | 5,000 | Primary source, most reliable |
| Supreme Court | main.sci.gov.in | 2,000 | Or via Indian Kanoon |
| High Courts | via Indian Kanoon | 2,000 | Delhi, Bombay, Karnataka |
| India Code | indiacode.nic.in | 1,000 | Central Acts & Statutes |
| **Total** | | **10,000** | |

### Date Range
- **Recent cases**: 2019-01-01 to 2024-12-31
- **Landmark cases**: Any date (identified by importance)

---

## Data Model

### LegalDocument Schema

```python
class LegalDocument(BaseModel):
    # Identifiers
    doc_id: str                      # Unique ID (hash of citation+court+date)
    source: str                      # indian_kanoon, supreme_court, high_courts, india_code
    url: str                         # Source URL
    
    # Case Information
    citation: str | None             # e.g., "2023 SCC 456", "AIR 2022 SC 1234"
    case_number: str | None          # e.g., "Criminal Appeal No. 123/2023"
    case_title: str                  # e.g., "State of Maharashtra v. ABC"
    court: str                       # e.g., "Supreme Court of India"
    
    # Parties
    petitioner: str | None
    respondent: str | None
    
    # Bench
    judges: list[str]                # List of judge names
    
    # Dates
    date_decided: date | None        # Judgment date
    
    # Classification
    subject_category: str | None     # Criminal, Civil, Constitutional, etc.
    acts_referred: list[str]         # Statutes cited
    sections_referred: list[str]     # Specific sections
    cases_cited: list[str]           # Precedents cited
    
    # Outcome
    outcome: str | None              # Allowed, Dismissed, Remanded, etc.
    
    # Content
    headnotes: str | None            # Summary if available
    full_text: str                   # Complete judgment text
    
    # Metadata
    word_count: int
    scraped_at: datetime
    is_landmark: bool = False
```

---

## Do's and Don'ts

### ✅ DO

1. **Test incrementally**
   ```bash
   # Test with 10 docs before full run
   python scripts/run_collection.py --source indian_kanoon --dry-run
   ```

2. **Save progress frequently**
   - Checkpoint every 100 documents
   - Enable resume after interruption

3. **Respect rate limits**
   - Minimum 2 seconds between requests
   - Add random jitter (0.5-1.5s)
   - Back off on 429 errors

4. **Log everything**
   - Timestamp all operations
   - Log URLs fetched, documents saved
   - Log errors with full context

5. **Handle Ctrl+C gracefully**
   - Save progress on interrupt
   - Print resume instructions

6. **Validate early**
   - Run validation after first 1,000 docs
   - Catch issues before full collection

7. **Use context managers**
   ```python
   with rate_limiter.acquire():
       response = requests.get(url)
   ```

### ❌ DON'T

1. **Don't scrape too fast**
   - Never less than 2 seconds between requests
   - You will get blocked

2. **Don't ignore 429 responses**
   - Implement exponential backoff
   - Wait at least 60 seconds

3. **Don't store secrets in code**
   - Use `.env` file (gitignored)
   - Load with `python-dotenv`

4. **Don't skip error handling**
   - Every network call can fail
   - Every parse can fail

5. **Don't commit data/ to git**
   - Add to `.gitignore`
   - Data is large and regeneratable

6. **Don't use bare except**
   - Always catch specific exceptions
   - Log before handling

---

## Testing Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Test specific module
pytest tests/test_scrapers.py -v

# Test imports
python -c "from scrapers.indian_kanoon import IndianKanoonScraper; print('OK')"

# Dry run (10 documents)
python scripts/run_collection.py --source indian_kanoon --dry-run

# Run single source
python scripts/run_collection.py --source indian_kanoon --target 5000

# Run all sources
python scripts/run_collection.py --source all --target 10000

# Resume interrupted collection
python scripts/run_collection.py --resume

# Validate collected data
python scripts/validate_data.py

# Check document counts
sqlite3 data/metadata.db "SELECT source, COUNT(*) FROM documents GROUP BY source"
```

---

## Detailed Specifications

For implementation details, see:

| Document | Contents |
|----------|----------|
| `docs/PHASE1_DATA_COLLECTION.md` | Scraper implementations, HTML selectors, parsing logic |
| `docs/PHASE2_DATA_PROCESSING.md` | Data cleaning, training data generation |
| `docs/PHASE3_MODEL_TRAINING.md` | Fine-tuning, quantization, deployment |
| `docs/ARCHITECTURE.md` | System design, data flow diagrams |

---

## Success Criteria

| Metric | Target | Verification |
|--------|--------|--------------|
| Total documents | 10,000 | `SELECT COUNT(*) FROM documents` |
| Indian Kanoon | 5,000 | `SELECT COUNT(*) WHERE source='indian_kanoon'` |
| Supreme Court | 2,000 | `SELECT COUNT(*) WHERE source='supreme_court'` |
| High Courts | 2,000 | `SELECT COUNT(*) WHERE source='high_courts'` |
| India Code | 1,000 | `SELECT COUNT(*) WHERE source='india_code'` |
| Valid full_text | >99% | Validation script |
| No duplicates | 100% | Deduplication check |
| Date coverage | 2019-2024 | Year distribution |

---

## Questions?

If unclear about any implementation detail:
1. Check `docs/` for detailed specs
2. Ask before proceeding
3. Prioritize working code over perfect code
4. We can iterate and improve

---

## Changelog

| Date | Change |
|------|--------|
| 2025-01-14 | Initial project setup |
