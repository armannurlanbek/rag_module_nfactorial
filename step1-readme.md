# Step 1 — Data Extraction & Validation

Extracts and validates text from specific pages of `kaztelecom.pdf` using [Docling](https://github.com/docling-project/docling).

## Requirements

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended) **or** pip

## Setup

### Option A — uv (recommended)

```bash
uv venv .venv
uv pip install docling
```

### Option B — pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install docling
```

## Run

```bash
# with uv
uv run python3 step1-data-extraction.py

# with pip/venv
source .venv/bin/activate
python3 step1-data-extraction.py
```

## Output

The script prints extracted text grouped by page, along with a validation summary:

```
Extracting pages 2-3 from kaztelecom.pdf ...
[INFO] Page 2: 1 text blocks, 10 chars
[INFO] Page 3: 2 text blocks, 106 chars

Validation passed

======================================== PAGE 2 ========================================
...

======================================== PAGE 3 ========================================
...
```

## Configuration

Edit the constants at the top of `step1-data-extraction.py` to change the target file or pages:

```python
PDF_PATH = Path(__file__).parent / "kaztelecom.pdf"
PAGE_RANGE = (2, 3)
```
