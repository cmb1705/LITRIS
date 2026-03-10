# Task 00: Project Setup

**Phase:** 0
**Priority:** Critical (Blocking)
**Estimated Effort:** 1-2 hours

---

## Objective

Establish the project foundation including directory structure, dependencies, configuration system, and development environment.

---

## Prerequisites

- Python 3.10+ installed
- Git repository initialized (already done)
- Access to Zotero database path confirmed

---

## Implementation Details

### 00.1 Create pyproject.toml

**File:** `pyproject.toml`

**Purpose:** Modern Python project configuration using PEP 621 standards.

**Contents Required:**
- Project name: `lit_review`
- Version: `0.1.0`
- Python requirement: `>=3.10`
- Author information
- Project description
- License (recommend MIT or Apache 2.0)
- Tool configurations for:
  - pytest (test paths, options)
  - black (line length, target version)
  - isort (profile, line length)
  - mypy (strict mode settings)

**Considerations:**
- Use `[project.optional-dependencies]` for dev dependencies
- Set `packages = [{include = "src"}]` for proper imports

---

### 00.2 Create requirements.txt

**File:** `requirements.txt`

**Purpose:** Core runtime dependencies.

**Required Packages:**

| Package | Minimum Version | Notes |
|---------|-----------------|-------|
| anthropic | latest | Claude API client |
| pymupdf | latest | PDF text extraction (installed as `fitz`) |
| pydantic | >=2.0 | Data validation and schemas |
| pyyaml | latest | YAML configuration parsing |
| python-dotenv | latest | Environment variable loading |
| tqdm | latest | Progress bars for CLI |
| chromadb | latest | Vector store |
| sentence-transformers | latest | Local embedding generation |

**Notes:**
- Pin major versions only for flexibility
- Include comments explaining each dependency's purpose

---

### 00.3 Create requirements-dev.txt

**File:** `requirements-dev.txt`

**Purpose:** Development and testing dependencies.

**Required Packages:**

| Package | Purpose |
|---------|---------|
| pytest | Test framework |
| pytest-cov | Coverage reporting |
| pytest-asyncio | Async test support |
| black | Code formatting |
| isort | Import sorting |
| mypy | Static type checking |
| ruff | Fast linting |

**Notes:**
- Reference base requirements: `-r requirements.txt`
- Keep dev dependencies separate for lighter production installs

---

### 00.4 Create .env.example

**File:** `.env.example`

**Purpose:** Template for required environment variables.

**Required Variables:**

```
# Anthropic API Key for Claude
ANTHROPIC_API_KEY=your_api_key_here

# OpenAI API Key (optional, for embeddings)
OPENAI_API_KEY=your_api_key_here

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
```

**Notes:**
- Never commit actual `.env` file
- Document each variable with comments
- Indicate which are optional

---

### 00.5 Update .gitignore

**File:** `.gitignore`

**Additions Required:**

```
# Environment
.env
.venv/
venv/
env/

# Data directories (large files)
data/cache/
data/index/chroma/
data/logs/
data/query_results/

# Keep structure but ignore contents
!data/cache/.gitkeep
!data/index/.gitkeep
!data/logs/.gitkeep

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.coverage
htmlcov/
.pytest_cache/

# OS
.DS_Store
Thumbs.db
```

---

### 00.6 Create config.yaml

**File:** `config.yaml`

**Purpose:** Central configuration file for all system settings.

**Sections Required:**

1. **zotero** - Database and storage paths
2. **extraction** - LLM model settings, retry logic, batch sizes
3. **embeddings** - Model selection, chunk parameters
4. **storage** - Output paths for index, cache, logs
5. **api** - Environment variable names for API keys
6. **options** - Feature flags and processing options

**Key Settings:**

| Setting | Default | Description |
|---------|---------|-------------|
| zotero.database_path | D:/Zotero/zotero.sqlite | Zotero DB location |
| zotero.storage_path | D:/Zotero/storage | PDF storage location |
| extraction.model | claude-3-opus-20240229 | LLM model for extraction |
| extraction.max_retries | 3 | API retry attempts |
| extraction.batch_size | 10 | Papers per progress update |
| embeddings.model | sentence-transformers/all-MiniLM-L6-v2 | Embedding model |
| embeddings.chunk_max_tokens | 512 | Max tokens per chunk |
| storage.index_path | data/index | Index output directory |
| storage.cache_path | data/cache | Cache directory |
| options.use_zotero_cache | true | Use existing .zotero-ft-cache |
| options.ocr_enabled | true | Enable OCR fallback |
| options.skip_existing | true | Skip processed papers on rerun |

---

### 00.7 Create src/config.py

**File:** `src/config.py`

**Purpose:** Configuration loader with validation.

**Class: Config**

**Responsibilities:**
- Load config.yaml from project root
- Load environment variables from .env
- Validate required settings exist
- Provide typed access to configuration values
- Support environment variable overrides

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `load()` | Config | Class method to load and return config |
| `get_zotero_db_path()` | Path | Validated path to Zotero database |
| `get_storage_path()` | Path | Validated path to Zotero storage |
| `get_anthropic_key()` | str | API key from environment |
| `get_index_path()` | Path | Path to index directory |
| `get_cache_path()` | Path | Path to cache directory |

**Validation:**
- Check Zotero paths exist and are accessible
- Check required API keys are set
- Create output directories if missing
- Raise clear errors for missing configuration

---

### 00.8 Create Directory Structure

**Directories to Create:**

```
src/
src/zotero/
src/extraction/
src/analysis/
src/indexing/
src/query/
src/utils/
scripts/
data/
data/index/
data/cache/
data/cache/pdf_text/
data/logs/
data/query_results/
tests/
tests/fixtures/
tests/fixtures/sample_papers/
proposals/tasks/
```

**For Each src/ Subdirectory:**
- Create `__init__.py` with module docstring
- Include `__all__` list for public exports

**For data/ Subdirectories:**
- Create `.gitkeep` files to preserve structure in git
- Ensure directories are in .gitignore

---

### 00.9 Create tests/conftest.py

**File:** `tests/conftest.py`

**Purpose:** Shared pytest fixtures and configuration.

**Fixtures Required:**

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `config` | session | Loaded test configuration |
| `sample_pdf_path` | session | Path to test PDF file |
| `mock_zotero_db` | function | Mock Zotero database connection |
| `temp_index_dir` | function | Temporary directory for test outputs |

**Notes:**
- Use `tmp_path` fixture for temporary files
- Mock external API calls by default
- Provide sample data fixtures for consistent tests

---

## Test Scenarios

### T00.1 Configuration Loading

**Test:** Config loads successfully from valid config.yaml
**Input:** Valid config.yaml with all required fields
**Expected:** Config object with accessible properties
**Verify:** All paths resolve, no exceptions raised

### T00.2 Missing Configuration

**Test:** Clear error when config.yaml missing
**Input:** No config.yaml file
**Expected:** FileNotFoundError with helpful message
**Verify:** Error message indicates expected file location

### T00.3 Missing API Key

**Test:** Clear error when ANTHROPIC_API_KEY not set
**Input:** config.yaml valid, no .env file
**Expected:** ValueError indicating missing key
**Verify:** Error message names the missing variable

### T00.4 Invalid Zotero Path

**Test:** Clear error when Zotero database not found
**Input:** config.yaml with invalid zotero.database_path
**Expected:** FileNotFoundError with path shown
**Verify:** Error distinguishes between file and directory issues

### T00.5 Directory Creation

**Test:** Output directories created automatically
**Input:** Valid config, data/ directories don't exist
**Expected:** Directories created on Config.load()
**Verify:** data/index/, data/cache/, data/logs/ exist after load

---

## Caveats and Edge Cases

### Path Handling on Windows

- Use `pathlib.Path` for all path operations
- Windows uses backslashes but Python accepts forward slashes
- Avoid hardcoding path separators
- Test with both `D:/Zotero` and `D:\Zotero` formats

### Environment Variable Priority

- Environment variables should override config.yaml values
- Document override behavior clearly
- Use consistent naming: `LIT_REVIEW_` prefix for custom vars

### Virtual Environment

- Recommend using venv or conda
- Document activation steps for Windows
- Include activation in README

### Zotero Database Locking

- Zotero locks its database when running
- Config validation should warn if database is locked
- Consider read-only connection mode

---

## Acceptance Criteria

- [x] `pip install -r requirements.txt` succeeds
- [x] `python -c "from src.config import Config; Config.load()"` succeeds
- [x] All src/ subdirectories have `__init__.py`
- [x] `pytest tests/` runs (even if no tests yet)
- [x] Configuration validation catches missing Zotero path
- [x] Configuration validation catches missing API key

---

## Files Created

| File | Status |
|------|--------|
| pyproject.toml | Complete |
| requirements.txt | Complete |
| requirements-dev.txt | Complete |
| .env.example | Complete |
| .gitignore | Complete |
| config.yaml | Complete |
| src/__init__.py | Complete |
| src/config.py | Complete |
| src/utils/__init__.py | Complete |
| src/zotero/__init__.py | Complete |
| src/extraction/__init__.py | Complete |
| src/analysis/__init__.py | Complete |
| src/indexing/__init__.py | Complete |
| src/query/__init__.py | Complete |
| tests/conftest.py | Complete |

---

*End of Task 00*
