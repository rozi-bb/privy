"""One-time fetch of the FiQA-2018 benchmark from the HuggingFace BeIR repos.

FiQA is a financial question answering benchmark: real finance questions
retrieved against forum posts and opinion articles. 57,638 corpus docs and
648 test queries with relevance judgments (qrels), which is what lets us
compute honest NDCG@10.

Run once:

    uv run python practice/fiqa/download.py
"""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    corpus = load_dataset("BeIR/fiqa", "corpus", split="corpus")
    queries = load_dataset("BeIR/fiqa", "queries", split="queries")
    qrels = load_dataset("BeIR/fiqa-qrels", split="test")

    corpus.to_parquet(DATA_DIR / "corpus.parquet")
    queries.to_parquet(DATA_DIR / "queries.parquet")
    qrels.to_parquet(DATA_DIR / "qrels.parquet")

    print(f"Corpus:  {len(corpus):>6} docs    -> {DATA_DIR / 'corpus.parquet'}")
    print(f"Queries: {len(queries):>6} queries -> {DATA_DIR / 'queries.parquet'}")
    print(f"Qrels:   {len(qrels):>6} judgments -> {DATA_DIR / 'qrels.parquet'}")


if __name__ == "__main__":
    main()
