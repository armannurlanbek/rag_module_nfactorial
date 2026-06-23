# Step 2 — Chunking Strategy 1: Semantic Coherence

Evaluates how **semantically coherent** the chunks produced by different
chunking strategies are, using sentence embeddings. This is **Student 2** of
the RAG component-testing seminar.

> **Input:** `step1-output.json` (the `extracted_text` fields from Step 1)
> **Output:** `step2-output.json`, `step2_plots/*.png`, console + Markdown table

---

## Метрика — Semantic Coherence Score

Идея: если предложения внутри чанка семантически связаны, их эмбеддинги
должны быть близки друг к другу.

For each chunk:

1. Split the chunk into sentences (Russian-aware).
2. Embed the sentences with a **multilingual** sentence-transformer.
3. Compute the pairwise cosine-similarity matrix.
4. Take the **upper triangle** (excluding the diagonal):
   - **MEAN** → `coherence_score`
   - **VARIANCE** → `coherence_variance`
5. Aggregate across all chunks of a strategy (mean of per-chunk scores).

### Why a multilingual model?

The corpus is **Russian**. The brief's example used the English-only
`all-MiniLM-L6-v2`, which produces unreliable embeddings for Cyrillic text.
This step uses **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**
instead. (~470 MB, downloaded automatically on first run.)

### Russian sentence splitting

Naively splitting on `.` is wrong here — dates like `13.05.2024` contain dots.
The script uses **nltk `punkt`** (Russian) and downloads `punkt_tab`/`punkt` on
first run. If nltk is unavailable, it falls back to a regex that only breaks on
`. ! ? …` followed by whitespace and an uppercase Latin/Cyrillic letter (so
dates stay intact).

---

## Chunking strategies compared

| Strategy         | How it works                                                        |
| ---------------- | ------------------------------------------------------------------- |
| `fixed_size_200` | Fixed 200-character windows (plain slicing).                        |
| `fixed_size_500` | Fixed 500-character windows (plain slicing).                        |
| `recursive`      | LangChain `RecursiveCharacterTextSplitter` (paragraph/sentence aware). |
| `sentence_4`     | Group every 4 sentences into one chunk.                             |
| `semantic`       | LangChain `SemanticChunker` (splits where embedding similarity drops). |

`semantic` is **optional** — if `langchain-experimental` / `langchain-huggingface`
fails to import or run, it is skipped gracefully and the reason is recorded in
`step2-output.json` under `config.semantic_chunker_skipped_reason`.

---

## Setup

### Option A — uv (recommended)

```bash
uv venv .venv --python 3.12
uv pip install --python .venv \
  sentence-transformers scikit-learn langchain langchain-text-splitters \
  langchain-experimental langchain-community langchain-huggingface \
  matplotlib seaborn nltk numpy
```

### Option B — pip

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install sentence-transformers scikit-learn langchain langchain-text-splitters \
  langchain-experimental langchain-community langchain-huggingface \
  matplotlib seaborn nltk numpy
```

> No OpenAI API key is needed for Step 2 — the metric is fully local
> (sentence embeddings only).

---

## Run

```bash
# with uv
uv run --python .venv python step2-semantic-coherence.py

# with pip/venv
.venv\Scripts\activate
python step2-semantic-coherence.py
```

The first run downloads the embedding model (~470 MB) and the nltk `punkt`
data; subsequent runs use the local cache and are fast.

---

## Output

The script:

1. Loads `step1-output.json` and concatenates the `extracted_text` fields.
2. Chunks the corpus with all 5 strategies above.
3. Computes the Semantic Coherence Score (mean + variance) per strategy.
4. Prints and saves a comparison table.
5. Saves visualizations to `step2_plots/`:
   - `coherence_bar.png` — mean coherence score per strategy (whiskers = ±1 std).
   - `coherence_boxplot.png` — distribution of per-chunk scores per strategy.
6. Prints a recommendation for the most coherent strategy.
7. Writes everything to `step2-output.json`.

### Comparison table (actual run)

| Strategy       | coherence_score | coherence_variance | num_chunks | avg_sentences_per_chunk |
| -------------- | --------------- | ------------------ | ---------- | ----------------------- |
| fixed_size_200 | 0.2646          | 0.0091             | 29         | 2.55                    |
| fixed_size_500 | 0.2590          | 0.0241             | 12         | 4.83                    |
| recursive      | **0.3257**      | 0.0148             | 16         | 3.75                    |
| sentence_4     | 0.3147          | 0.0201             | 12         | 3.92                    |
| semantic       | 0.2593          | 0.0222             | 4          | 11.75                   |

### Recommendation

**`recursive`** yields the most coherent chunks (`coherence_score = 0.3257`),
with moderate variance and a sensible chunk count. `sentence_4` is a close
second (0.3147). Both respect natural language boundaries, which keeps related
sentences together. The two `fixed_size` strategies and `semantic` score lower:
fixed-size cuts mid-sentence, and `semantic` produced only 4 very large chunks
on this short corpus (avg 11.75 sentences), diluting per-chunk coherence.

> **Note on absolute values:** scores are ~0.26–0.33 because this corpus is a
> chronological news digest where *every* paragraph is a different event, so even
> a "good" chunk mixes loosely related sentences. The **relative ordering** is
> what matters for choosing a strategy.

---

## Configuration

Edit the constants at the top of `step2-semantic-coherence.py`:

```python
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FIXED_SIZES = [200, 500]
RECURSIVE_CHUNK_SIZE = 500
RECURSIVE_CHUNK_OVERLAP = 50
SENTENCES_PER_CHUNK = 4
```

---

## Caveats

- **Model download:** ~470 MB on first run (`paraphrase-multilingual-MiniLM-L12-v2`).
- **Symlink warning on Windows:** HuggingFace prints a cache-symlink warning
  unless Developer Mode is on. Harmless; silence with `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
- **Short corpus:** only ~5.6 KB / 47 sentences from 2 PDF pages, so absolute
  scores are low and `semantic` makes very few chunks. Rankings still hold.
- **`semantic` skipped path:** if `langchain-experimental` is unavailable, the
  strategy is dropped and the run still succeeds.
```
