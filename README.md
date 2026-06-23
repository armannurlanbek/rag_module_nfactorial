# RAG Module — nFactorial

Testing and comparison of RAG system components using **RAGAS** and **LLM as a Judge** methodologies.

## Requirements

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended) **or** pip
- OpenAI API key

## Setup

### 1. Clone and enter the repo

```bash
git clone https://github.com/armannurlanbek/rag_module_nfactorial
cd rag_module_nfactorial
```

### 2. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real key:

```
OPENAI_API_KEY=sk-proj-your-key-here
```

### 3. Install dependencies

**Option A — uv (recommended)**

```bash
uv venv .venv
uv pip install docling openai python-dotenv
```

**Option B — pip**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install docling openai python-dotenv
```

---

## Step 1 — Data Extraction & Validation

Extracts text from pages 2–3 of `kaztelecom.pdf` using [Docling](https://github.com/docling-project/docling), renders page screenshots, and validates extraction quality using GPT-4o as a Visual LLM judge.

### Run

```bash
# with uv
uv run python3 step1-data-extraction.py

# with pip/venv
source .venv/bin/activate
python3 step1-data-extraction.py
```

### Output

The script:

1. Extracts text from pages 2–3 using Docling
2. Renders screenshots of those pages → saved to `screenshots/`
3. Sends each screenshot + extracted text to GPT-4o Vision (LLM as a Judge)
4. Prints JSON scores per page:

```json
{
  "structure_score": 4,
  "tables_score": 3,
  "formatting_score": 4,
  "completeness_score": 5,
  "overall_score": 4.0,
  "comments": "...",
  "page": 2
}
```

### Configuration

Edit the constants at the top of `step1-data-extraction.py` to change the target file or pages:

```python
PDF_PATH = Path(__file__).parent / "kaztelecom.pdf"
PAGE_RANGE = (2, 3)
```

---

## Step 2 — Chunking Strategy: Semantic Coherence

Measures how semantically coherent the chunks of different chunking strategies
are, using multilingual sentence embeddings on the Russian text from Step 1.
See [`step2-readme.md`](step2-readme.md) for full details.

### Run

```bash
uv pip install --python .venv sentence-transformers scikit-learn langchain \
  langchain-text-splitters langchain-experimental langchain-community \
  langchain-huggingface matplotlib seaborn nltk numpy

uv run --python .venv python step2-semantic-coherence.py
```

### Output

Compares `fixed_size`, `recursive`, `sentence`, and `semantic` chunking via a
**Semantic Coherence Score** (mean pairwise cosine similarity of sentence
embeddings). Writes `step2-output.json`, plots to `step2_plots/`, and a console
comparison table. On this corpus, **`recursive`** gives the most coherent chunks.

---

## Step 3 — Chunking Strategy: Context Independence

_Coming soon_
