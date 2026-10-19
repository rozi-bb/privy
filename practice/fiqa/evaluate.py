"""Evaluate each retrieval stage on FiQA with NDCG@10.

Stages: BM25, dense, hybrid (RRF), and hybrid + Jev rerank. The embeddings
are computed once and cached to disk, because embedding the whole corpus is
the slow part. Jev reranking is an API call per candidate, so by default it
runs on a sample of the queries.

    uv run python practice/fiqa/evaluate.py --limit 100   # everything on 100 queries
    uv run python practice/fiqa/evaluate.py --limit 30    # quick smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document

from agent.embeddings import LocalEmbeddings
from agent.reranking import rerank_documents
from agent.retrieval import KeywordRetriever, reciprocal_rank_fusion

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"

HYBRID_TOP_K = 20
RERANK_TOP_N = 10
EVAL_K = 10


def doc_text(title: object, text: object) -> str:
    """Concatenate title and text, the way the corpus is meant to be searched."""
    title = str(title or "").strip()
    text = str(text or "").strip()
    return f"{title}\n{text}".strip()


def load_fiqa(
    corpus_limit: int | None = None,
) -> tuple[list[Document], list[str], list[tuple[str, str]], dict[str, dict[str, int]]]:
    """Load the corpus as Documents plus the test queries and judgments."""
    corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
    queries = pd.read_parquet(DATA_DIR / "queries.parquet")
    qrels_df = pd.read_parquet(DATA_DIR / "qrels.parquet")
    if corpus_limit:
        corpus = corpus.head(corpus_limit)
        qrels_df = qrels_df[qrels_df["corpus-id"].isin(corpus["_id"])]

    documents = [
        Document(page_content=doc_text(title, text), metadata={"doc_id": str(doc_id)})
        for doc_id, title, text in corpus.itertuples(index=False, name=None)
    ]
    ids = [doc.metadata["doc_id"] for doc in documents]

    test_ids = set(qrels_df["query-id"].astype(str))
    test_queries = [
        (str(query_id), str(text))
        for query_id, _title, text in queries.itertuples(index=False, name=None)
        if str(query_id) in test_ids
    ]

    qrels: dict[str, dict[str, int]] = {}
    for record in qrels_df.to_dict("records"):
        qrels.setdefault(str(record["query-id"]), {})[str(record["corpus-id"])] = int(
            record["score"]
        )

    return documents, ids, test_queries, qrels


def cached_embeddings(documents: list[Document]) -> tuple[np.ndarray, list[str]]:
    """Embed the corpus once and cache the normalized vectors to disk.

    The cache is keyed by the number of documents, so a smoke test with a
    smaller corpus does not reuse the full corpus' vectors.
    """
    cache = DATA_DIR / f"embeddings_{len(documents)}.npy"
    ids_cache = DATA_DIR / f"embedding_ids_{len(documents)}.json"
    if cache.exists() and ids_cache.exists():
        return np.load(cache), json.loads(ids_cache.read_text())

    model = LocalEmbeddings()
    texts = [d.page_content for d in documents]
    parts = []
    for start in range(0, len(texts), 128):
        print(f"  embedding {start}/{len(texts)}", flush=True)
        parts.append(
            np.asarray(
                model.embed_documents(texts[start : start + 128]), dtype=np.float32
            )
        )
    vectors = np.vstack(parts)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    np.save(cache, vectors)
    ids = [d.metadata["doc_id"] for d in documents]
    ids_cache.write_text(json.dumps(ids))
    print(f"  embeddings cached -> {cache}", flush=True)
    return vectors, ids


def dense_rank(
    text: str, model: LocalEmbeddings, vectors: np.ndarray, ids: list[str], k: int
) -> list[str]:
    """Rank ids by cosine similarity of their embeddings to the query."""
    q = np.asarray(model.embed_query(text), dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-9
    order = np.argsort(-(vectors @ q))
    return [ids[i] for i in order[:k]]


def ndcg_at_k(retrieved: list[str], judged: dict[str, int], k: int = EVAL_K) -> float:
    """NDCG@k against the relevance judgments."""
    gains = np.asarray([judged.get(doc_id, 0) for doc_id in retrieved[:k]], dtype=float)
    dcg = float(np.sum((2**gains - 1) / np.log2(np.arange(2, len(gains) + 2))))
    ideal = np.sort(np.asarray(list(judged.values()), dtype=float))[::-1][:k]
    idcg = float(np.sum((2**ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / idcg if idcg > 0 else 0.0


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only this many queries"
    )
    parser.add_argument(
        "--corpus-limit",
        type=int,
        default=None,
        help="Use only this many corpus docs (smoke tests)",
    )
    parser.add_argument(
        "--skip-rerank", action="store_true", help="Skip the Jev rerank stage"
    )
    args = parser.parse_args()

    documents, ids, test_queries, qrels = load_fiqa(corpus_limit=args.corpus_limit)
    by_id = {doc.metadata["doc_id"]: doc for doc in documents}
    if args.limit:
        test_queries = test_queries[: args.limit]

    print(f"Indexing {len(documents)} documents with BM25...")
    keyword = KeywordRetriever(documents)
    print("Embedding the corpus (cached after the first run)...")
    vectors, cache_ids = cached_embeddings(documents)
    model = LocalEmbeddings()

    if not test_queries:
        print(
            "No judged queries for this corpus sample; use the full corpus for a real evaluation."
        )
        return

    totals: dict[str, list[float]] = {"bm25": [], "dense": [], "hybrid": []}

    for i, (query_id, text) in enumerate(test_queries):
        judged = qrels[query_id]
        bm25_ids = [
            doc.metadata["doc_id"] for doc in keyword.search(text, HYBRID_TOP_K)
        ]
        dense_ids = dense_rank(text, model, vectors, cache_ids, HYBRID_TOP_K)
        hybrid = reciprocal_rank_fusion(
            [[by_id[i] for i in bm25_ids], [by_id[i] for i in dense_ids]]
        )[:HYBRID_TOP_K]
        hybrid_ids = [doc.metadata["doc_id"] for doc in hybrid]

        totals["bm25"].append(ndcg_at_k(bm25_ids, judged))
        totals["dense"].append(ndcg_at_k(dense_ids, judged))
        totals["hybrid"].append(ndcg_at_k(hybrid_ids, judged))

        if (i + 1) % 25 == 0:
            print(f"  scored {i + 1}/{len(test_queries)}")

    print(f"\n=== NDCG@{EVAL_K} (mean over {len(test_queries)} queries) ===")
    for stage in ("bm25", "dense", "hybrid"):
        print(f"{stage:8s}: {np.mean(totals[stage]):.4f}")

    if args.skip_rerank:
        return

    print(
        f"\nReranking the hybrid Top-{HYBRID_TOP_K} with Jev on {len(test_queries)} queries..."
    )
    rerank_scores: list[float] = []
    for i, (query_id, text) in enumerate(test_queries):
        bm25_ids = [
            doc.metadata["doc_id"] for doc in keyword.search(text, HYBRID_TOP_K)
        ]
        dense_ids = dense_rank(text, model, vectors, cache_ids, HYBRID_TOP_K)
        hybrid = reciprocal_rank_fusion(
            [[by_id[i] for i in bm25_ids], [by_id[i] for i in dense_ids]]
        )[:HYBRID_TOP_K]
        top = await rerank_documents(text, hybrid, RERANK_TOP_N)
        rerank_scores.append(
            ndcg_at_k([d.metadata["doc_id"] for d in top], qrels[query_id])
        )
        if (i + 1) % 10 == 0:
            print(f"  reranked {i + 1}/{len(test_queries)}")

    print(f"{'hybrid+rerank':8s}: {np.mean(rerank_scores):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
