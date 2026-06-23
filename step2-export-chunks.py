"""
Step 2 — Export chunk texts (handoff for Student 3)
===================================================

Exports the EXACT chunk texts produced by each chunking strategy in
`step2-semantic-coherence.py` into `step2-chunks.json`, so Student 3 can run
the Context-Independence LLM-as-a-judge on identical chunks. Computing both
metrics (coherence + independence) on the same chunks is what lets the team
combine them and pick the best chunking strategy.

This reuses the chunking functions/params from step2-semantic-coherence.py, so
the exported chunks are guaranteed identical to the ones that were scored.

Run (same env as step 2):
    uv run --python .venv python step2-export-chunks.py
"""

import importlib.util
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
SCORER_FILE = BASE_DIR / "step2-semantic-coherence.py"
OUTPUT_FILE = BASE_DIR / "step2-chunks.json"


def load_scorer_module():
    """Import step2-semantic-coherence.py (hyphenated name -> importlib)."""
    spec = importlib.util.spec_from_file_location("step2_scorer", SCORER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # defines funcs/constants; main() is guarded
    return module


def build_strategies(mod, corpus, all_sentences):
    """Rebuild chunks with the IDENTICAL strategies/params used for scoring."""
    strategies = {}
    for size in mod.FIXED_SIZES:
        strategies[f"fixed_size_{size}"] = mod.chunk_fixed_size(corpus, size)
    strategies["recursive"] = mod.chunk_recursive(
        corpus, mod.RECURSIVE_CHUNK_SIZE, mod.RECURSIVE_CHUNK_OVERLAP
    )
    strategies[f"sentence_{mod.SENTENCES_PER_CHUNK}"] = mod.chunk_by_sentences(
        all_sentences, mod.SENTENCES_PER_CHUNK
    )

    semantic_skipped_reason = None
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        lc_embeddings = HuggingFaceEmbeddings(model_name=mod.EMBEDDING_MODEL)
        strategies["semantic"] = mod.chunk_semantic(corpus, lc_embeddings)
    except Exception as exc:  # SemanticChunker is optional — skip gracefully.
        semantic_skipped_reason = str(exc)
        print(f"[WARN] semantic strategy skipped: {exc}")

    return strategies, semantic_skipped_reason


def main():
    mod = load_scorer_module()

    # Same corpus + sentence splitter the scorer used.
    corpus = mod.load_corpus(mod.INPUT_FILE)
    split_sentences = mod.get_sentence_splitter()
    all_sentences = split_sentences(corpus)

    strategies, semantic_skipped_reason = build_strategies(mod, corpus, all_sentences)

    # Serialize chunks (text + lightweight metadata) per strategy.
    out_strategies = {}
    for name, chunks in strategies.items():
        items = [
            {
                "id": i,
                "text": text,
                "char_count": len(text),
                "num_sentences": len(split_sentences(text)),
            }
            for i, text in enumerate(chunks)
        ]
        out_strategies[name] = {"num_chunks": len(chunks), "chunks": items}

    output = {
        "_description": (
            "Exact chunk texts produced by Step 2 (Student 2 — Semantic "
            "Coherence) for each chunking strategy. HANDOFF FOR STUDENT 3: run "
            "the Context-Independence LLM-as-a-judge on THESE chunks so both "
            "metrics (coherence + independence) are computed on identical "
            "inputs and can be combined to pick the best strategy."
        ),
        "source": "step1-output.json",
        "config": {
            "embedding_model": mod.EMBEDDING_MODEL,
            "fixed_sizes": mod.FIXED_SIZES,
            "recursive_chunk_size": mod.RECURSIVE_CHUNK_SIZE,
            "recursive_chunk_overlap": mod.RECURSIVE_CHUNK_OVERLAP,
            "sentences_per_chunk": mod.SENTENCES_PER_CHUNK,
            "corpus_chars": len(corpus),
            "corpus_sentences": len(all_sentences),
            "semantic_chunker_skipped_reason": semantic_skipped_reason,
        },
        "strategies": out_strategies,
    }

    OUTPUT_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] Wrote {OUTPUT_FILE}")
    for name, s in out_strategies.items():
        print(f"  {name:<16} {s['num_chunks']} chunks")


if __name__ == "__main__":
    main()
