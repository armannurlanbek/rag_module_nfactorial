"""
Step 2 — Chunking Strategy 1: Semantic Coherence
=================================================

Goal: measure how semantically coherent the chunks produced by different
chunking strategies are. The idea (per the task brief):

    If the sentences inside a chunk are semantically related, their embedding
    vectors should be close to each other.

Metric — Semantic Coherence Score (per chunk):
    1. Split the chunk into sentences.
    2. Embed the sentences with a (MULTILINGUAL) sentence-transformer.
    3. Compute the pairwise cosine-similarity matrix.
    4. Take the upper triangle (excluding the diagonal):
         - its MEAN     -> coherence_score
         - its VARIANCE -> coherence_variance
    5. Aggregate across all chunks of a strategy (mean of per-chunk scores).

IMPORTANT — the source text (from Step 1) is in RUSSIAN, so we use a
MULTILINGUAL embedding model (paraphrase-multilingual-MiniLM-L12-v2) instead
of the English-only all-MiniLM-L6-v2 shown in the brief's example. Sentence
splitting is Russian-aware (nltk punkt, with a robust regex fallback) so that
dates like "13.05.2024" are NOT treated as sentence boundaries.

Input : step1-output.json  (test_cases[].extracted_text)
Output: step2-output.json  + step2_plots/*.png  + console/markdown table
"""

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration constants (edit these to tune the experiment)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "step1-output.json"
OUTPUT_FILE = BASE_DIR / "step2-output.json"
PLOTS_DIR = BASE_DIR / "step2_plots"

# Multilingual model — required because the corpus is Russian.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Fixed-size chunking sizes (characters).
FIXED_SIZES = [200, 500]
# Recursive splitter target size / overlap (characters).
RECURSIVE_CHUNK_SIZE = 500
RECURSIVE_CHUNK_OVERLAP = 50
# Sentence-grouping: how many sentences per chunk.
SENTENCES_PER_CHUNK = 4
# A chunk needs at least this many sentences to get a coherence score
# (a 1-sentence chunk has no sentence pairs, so it is skipped in the average).
MIN_SENTENCES_FOR_SCORE = 2

# How a page's text is separated from the next when we concatenate pages.
PAGE_SEPARATOR = "\n\n"


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------
def load_corpus(input_file: Path) -> str:
    """Read step1-output.json and concatenate all extracted_text fields."""
    data = json.loads(input_file.read_text(encoding="utf-8"))
    texts = [
        tc["extracted_text"].strip()
        for tc in data.get("test_cases", [])
        if tc.get("extracted_text", "").strip()
    ]
    if not texts:
        raise ValueError(f"No extracted_text found in {input_file}")
    return PAGE_SEPARATOR.join(texts)


# ---------------------------------------------------------------------------
# Russian-aware sentence splitting
# ---------------------------------------------------------------------------
# Regex fallback: split on . ! ? followed by whitespace + an uppercase letter
# (Latin or Cyrillic). This avoids breaking on dates such as "13.05.2024"
# because the chars after the dot there are digits, not an uppercase letter.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ«\"])")


def _regex_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ")
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def get_sentence_splitter():
    """
    Return a function text -> [sentences].

    Prefers nltk's punkt tokenizer (Russian-aware abbreviation handling); if
    nltk or its data is unavailable, falls back to the regex splitter above.
    """
    try:
        import nltk

        # punkt_tab is the newer resource name (nltk >= 3.8.2); punkt is older.
        for resource in ("punkt_tab", "punkt"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass

        from nltk.tokenize import sent_tokenize

        def _split(text: str) -> list[str]:
            text = text.replace("\n", " ")
            try:
                sents = sent_tokenize(text, language="russian")
            except Exception:
                sents = sent_tokenize(text)
            return [s.strip() for s in sents if s.strip()]

        # Smoke-test it; fall back if anything is wrong.
        _split("Тест. Проверка.")
        print("[INFO] Sentence splitter: nltk punkt (Russian)")
        return _split
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[WARN] nltk unavailable ({exc}); using regex sentence splitter")
        return _regex_sentences


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------
def chunk_fixed_size(text: str, size: int) -> list[str]:
    """
    Fixed-size character chunking via LangChain CharacterTextSplitter when
    available, else plain slicing. Note: CharacterTextSplitter splits on a
    separator first, so for a pure character window we fall back to slicing
    to guarantee the requested size behaviour.
    """
    chunks = [text[i : i + size] for i in range(0, len(text), size)]
    return [c.strip() for c in chunks if c.strip()]


def chunk_recursive(text: str, size: int, overlap: int) -> list[str]:
    """LangChain RecursiveCharacterTextSplitter (paragraph/sentence aware)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def chunk_by_sentences(sentences: list[str], n: int) -> list[str]:
    """Group every N sentences into one chunk."""
    chunks = []
    for i in range(0, len(sentences), n):
        group = sentences[i : i + n]
        if group:
            chunks.append(" ".join(group))
    return chunks


def chunk_semantic(text: str, embeddings) -> list[str]:
    """
    LangChain experimental SemanticChunker — splits at points where the
    embedding similarity between consecutive sentences drops. Uses the SAME
    multilingual embeddings. Raises if langchain_experimental is unavailable;
    the caller handles that gracefully.
    """
    from langchain_experimental.text_splitter import SemanticChunker

    chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    return [c.strip() for c in chunker.split_text(text) if c.strip()]


# ---------------------------------------------------------------------------
# Semantic Coherence metric
# ---------------------------------------------------------------------------
def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix (no sklearn dependency required)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized = embeddings / norms
    return normalized @ normalized.T


def semantic_coherence(sentences: list[str], model) -> dict | None:
    """
    Compute the coherence score for ONE chunk.

    Returns None if the chunk has fewer than MIN_SENTENCES_FOR_SCORE sentences
    (no sentence pairs -> the metric is undefined for it).
    """
    if len(sentences) < MIN_SENTENCES_FOR_SCORE:
        return None

    embeddings = model.encode(sentences, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float64)

    sim = cosine_similarity_matrix(embeddings)
    upper = sim[np.triu_indices_from(sim, k=1)]  # upper triangle, no diagonal

    return {
        "coherence_score": float(upper.mean()),
        "coherence_variance": float(upper.var()),
        "num_sentences": len(sentences),
    }


def evaluate_strategy(chunks: list[str], split_sentences, model) -> dict:
    """Run the coherence metric across all chunks of one strategy and aggregate."""
    per_chunk_scores = []
    per_chunk_variances = []
    sentence_counts = []
    scored_chunks = 0

    for chunk in chunks:
        sents = split_sentences(chunk)
        sentence_counts.append(len(sents))
        result = semantic_coherence(sents, model)
        if result is not None:
            per_chunk_scores.append(result["coherence_score"])
            per_chunk_variances.append(result["coherence_variance"])
            scored_chunks += 1

    if per_chunk_scores:
        mean_score = float(np.mean(per_chunk_scores))
        mean_variance = float(np.mean(per_chunk_variances))
    else:
        mean_score = float("nan")
        mean_variance = float("nan")

    return {
        "coherence_score": mean_score,
        "coherence_variance": mean_variance,
        "num_chunks": len(chunks),
        "scored_chunks": scored_chunks,
        "avg_sentences_per_chunk": float(np.mean(sentence_counts)) if sentence_counts else 0.0,
        "per_chunk_scores": per_chunk_scores,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def build_table(results: dict) -> str:
    """Render a Markdown comparison table."""
    header = (
        "| Strategy            | coherence_score | coherence_variance | num_chunks | avg_sentences_per_chunk |\n"
        "| ------------------- | --------------- | ------------------ | ---------- | ----------------------- |"
    )
    rows = []
    for name, r in results.items():
        rows.append(
            f"| {name:<19} | {r['coherence_score']:<15.4f} | {r['coherence_variance']:<18.4f} "
            f"| {r['num_chunks']:<10} | {r['avg_sentences_per_chunk']:<23.2f} |"
        )
    return header + "\n" + "\n".join(rows)


def recommend(results: dict) -> str:
    """
    Pick the strategy with the most coherent chunks.

    Primary criterion: highest mean coherence_score. We require a strategy to
    produce more than one chunk and to have at least one scored chunk, so we do
    not "win" by emitting a single giant chunk.
    """
    candidates = {
        name: r
        for name, r in results.items()
        if r["scored_chunks"] > 0 and r["num_chunks"] > 1 and not np.isnan(r["coherence_score"])
    }
    if not candidates:
        candidates = {
            name: r for name, r in results.items() if not np.isnan(r["coherence_score"])
        }

    best = max(candidates.items(), key=lambda kv: kv[1]["coherence_score"])
    name, r = best
    return (
        f"RECOMMENDATION: '{name}' yields the most coherent chunks.\n"
        f"  - coherence_score   = {r['coherence_score']:.4f} (higher = sentences inside a chunk are more related)\n"
        f"  - coherence_variance= {r['coherence_variance']:.4f} (lower = more uniformly coherent)\n"
        f"  - num_chunks        = {r['num_chunks']}\n"
        f"  - avg_sentences/chunk = {r['avg_sentences_per_chunk']:.2f}"
    )


def make_plots(results: dict, plots_dir: Path) -> list[str]:
    """Bar chart of coherence_score and a boxplot of per-chunk distributions."""
    import matplotlib

    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid")
    except Exception:
        pass

    plots_dir.mkdir(exist_ok=True)
    saved = []

    names = list(results.keys())
    scores = [results[n]["coherence_score"] for n in names]
    variances = [results[n]["coherence_variance"] for n in names]

    # --- Bar chart: mean coherence score per strategy (with variance whiskers)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, scores, yerr=np.sqrt(variances), capsize=5,
                  color="#4C72B0", edgecolor="black", alpha=0.85)
    ax.set_ylabel("Mean Semantic Coherence Score")
    ax.set_title("Semantic Coherence by Chunking Strategy (Russian corpus)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{s:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    bar_path = plots_dir / "coherence_bar.png"
    fig.savefig(bar_path, dpi=130)
    plt.close(fig)
    saved.append(str(bar_path))

    # --- Boxplot: distribution of per-chunk coherence scores per strategy
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [results[n]["per_chunk_scores"] or [results[n]["coherence_score"]] for n in names]
    # matplotlib >= 3.9 renamed `labels` -> `tick_labels`; support both.
    try:
        ax.boxplot(data, tick_labels=names, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=names, showmeans=True)
    ax.set_ylabel("Per-chunk Coherence Score")
    ax.set_title("Distribution of Per-Chunk Coherence Scores")
    ax.set_xticks(np.arange(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=30, ha="right")
    fig.tight_layout()
    box_path = plots_dir / "coherence_boxplot.png"
    fig.savefig(box_path, dpi=130)
    plt.close(fig)
    saved.append(str(box_path))

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Step 2 — Semantic Coherence of Chunking Strategies")
    print("=" * 70)

    # 1. Load corpus -------------------------------------------------------
    corpus = load_corpus(INPUT_FILE)
    print(f"[INFO] Loaded corpus: {len(corpus)} chars from {INPUT_FILE.name}")

    # 2. Sentence splitter + embedding model -------------------------------
    split_sentences = get_sentence_splitter()
    all_sentences = split_sentences(corpus)
    print(f"[INFO] Corpus contains {len(all_sentences)} sentences")

    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
    print("[INFO] (first run downloads the model — this may take a minute)")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    print("[INFO] Model loaded.")

    # 3. Build chunks for each strategy ------------------------------------
    strategies: dict[str, list[str]] = {}

    for size in FIXED_SIZES:
        strategies[f"fixed_size_{size}"] = chunk_fixed_size(corpus, size)

    try:
        strategies["recursive"] = chunk_recursive(
            corpus, RECURSIVE_CHUNK_SIZE, RECURSIVE_CHUNK_OVERLAP
        )
    except Exception as exc:
        print(f"[WARN] recursive strategy skipped: {exc}")

    strategies[f"sentence_{SENTENCES_PER_CHUNK}"] = chunk_by_sentences(
        all_sentences, SENTENCES_PER_CHUNK
    )

    # SemanticChunker is optional — skip gracefully if it does not install/run.
    semantic_skipped_reason = None
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        lc_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        strategies["semantic"] = chunk_semantic(corpus, lc_embeddings)
    except Exception as exc:
        semantic_skipped_reason = str(exc)
        print(f"[WARN] semantic (SemanticChunker) strategy skipped: {exc}")

    # 4. Evaluate the coherence metric per strategy ------------------------
    results: dict[str, dict] = {}
    for name, chunks in strategies.items():
        if not chunks:
            print(f"[WARN] '{name}' produced 0 chunks — skipped")
            continue
        print(f"[INFO] Scoring strategy '{name}' ({len(chunks)} chunks) ...")
        results[name] = evaluate_strategy(chunks, split_sentences, model)

    # 5. Report ------------------------------------------------------------
    table = build_table(results)
    recommendation = recommend(results)

    print("\n=== COMPARISON TABLE ===")
    print(table)
    print("\n=== " + recommendation.split(":", 1)[0] + " ===")
    print(recommendation)

    # 6. Visualizations ----------------------------------------------------
    try:
        plot_files = make_plots(results, PLOTS_DIR)
        print(f"\n[INFO] Plots saved: {', '.join(plot_files)}")
    except Exception as exc:
        plot_files = []
        print(f"[WARN] Plot generation failed: {exc}")

    # 7. Persist results ---------------------------------------------------
    output = {
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "fixed_sizes": FIXED_SIZES,
            "recursive_chunk_size": RECURSIVE_CHUNK_SIZE,
            "recursive_chunk_overlap": RECURSIVE_CHUNK_OVERLAP,
            "sentences_per_chunk": SENTENCES_PER_CHUNK,
            "corpus_chars": len(corpus),
            "corpus_sentences": len(all_sentences),
            "semantic_chunker_skipped_reason": semantic_skipped_reason,
        },
        "results": {
            name: {k: v for k, v in r.items() if k != "per_chunk_scores"}
            for name, r in results.items()
        },
        "per_chunk_scores": {name: r["per_chunk_scores"] for name, r in results.items()},
        "recommendation": recommendation,
        "comparison_table_markdown": table,
        "plots": plot_files,
    }
    OUTPUT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[INFO] Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
