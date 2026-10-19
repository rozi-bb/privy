"""Hybrid retrieval: BM25 (keyword) + dense (embedding), fused with RRF.

The question is searched two ways in parallel: keyword search over the chunk
tokens with BM25, and semantic search over the embeddings. The two ranked
lists are fused with Reciprocal Rank Fusion: a chunk ranked high in both
lists rises to the top.
"""

from __future__ import annotations

import asyncio
import re

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from rank_bm25 import BM25Okapi

RRF_K = 60

# Words too common to say anything about which article is meant; left in,
# "what is the ..." questions match whichever chunks repeat them most.
STOPWORDS = frozenset(
    "a an and are as at be by can could do does for from has have how i if in "
    "is it its me must my of on or our shall should so that the their them "
    "there these they this to us was we were what when where which who why "
    "will with would you your".split()
)


def tokenize(text: str) -> list[str]:
    """Split into lowercase word tokens without punctuation or stopwords."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def doc_key(doc: Document) -> str:
    """Stable identity of a chunk, used to dedupe during fusion."""
    meta = doc.metadata or {}
    if "doc_id" in meta:
        return f"doc_id:{meta['doc_id']}"
    return (
        f"{meta.get('kind')}:{meta.get('number')}"
        f":{meta.get('part', 0)}:{meta.get('point', 0)}"
    )


class KeywordRetriever:
    """In-memory BM25 index over a fixed list of chunks."""

    def __init__(self, documents: list[Document]) -> None:
        """Build the BM25 index over the given documents."""
        self._documents = documents
        self._bm25 = BM25Okapi([tokenize(doc.page_content) for doc in documents])

    def search(self, query: str, k: int) -> list[Document]:
        """Return the k highest-scoring chunks for the query."""
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._documents[i] for i in order[:k]]


def reciprocal_rank_fusion(
    results: list[list[Document]], k: int = RRF_K
) -> list[Document]:
    """Fuse ranked lists by reciprocal rank.

    Each chunk's score is the sum of 1 / (k + rank) over the lists it appears
    in, so a chunk ranked high in several lists rises to the top.
    """
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranked in results:
        for rank, doc in enumerate(ranked):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            docs[key] = doc
    fused = sorted(docs.values(), key=lambda doc: scores[doc_key(doc)], reverse=True)
    return fused


async def hybrid_search(
    question: str,
    keyword: KeywordRetriever,
    vector_store: VectorStore,
    keyword_k: int,
    vector_k: int,
    top_k: int,
) -> list[Document]:
    """Run keyword and semantic search in parallel, fuse, return Top-K."""
    keyword_results, dense_results = await asyncio.gather(
        asyncio.to_thread(keyword.search, question, keyword_k),
        vector_store.asimilarity_search(question, k=vector_k),
    )
    return reciprocal_rank_fusion([keyword_results, dense_results])[:top_k]
