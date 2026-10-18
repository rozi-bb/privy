"""Rerank retrieved passages with Jev (TypeSafe).

Jev answers one yes/no question per passage: "Is this passage relevant to the
user's question?" The passages are sorted by Jev's probability of yes, and the
Top-N are kept. Requests run in parallel behind a semaphore so a batch of
candidates finishes about as fast as one call; a failed call scores 0 and the
passage drops to the bottom instead of crashing the request.
"""

from __future__ import annotations

import asyncio
from functools import cache

from langchain_core.documents import Document
from typesafe_sdk import AsyncTypeSafeClient, Noul, RetryPolicy

JEV_MODEL = "jev-latest"
JEV_TIMEOUT = 5.0
MAX_CONCURRENCY = 5

RELEVANCE_QUESTION = Noul(
    instructions=(
        "Is this passage relevant for answering the user's question? "
        "Answer yes only if the passage contains information that directly "
        "helps answer the question."
    ),
    criteria={
        "true": "The passage contains information that helps answer the question.",
        "false": "The passage is unrelated or does not help answer the question.",
    },
)


def _state_text(question: str, passage: str) -> str:
    """Format the Jev input: the question together with the passage to judge."""
    return f"User question: {question}\n\nPassage: {passage}"


@cache
def _client(loop: asyncio.AbstractEventLoop) -> AsyncTypeSafeClient:
    # One client per event loop: its connection pool belongs to the loop that
    # created it, and reusing a warm connection is what keeps Jev fast.
    return AsyncTypeSafeClient(
        model=JEV_MODEL, retry=RetryPolicy(max_retries=0), timeout=JEV_TIMEOUT
    )


async def _relevance(question: str, passage: str) -> float:
    """Ask Jev how likely the passage is to help answer the question."""
    client = _client(asyncio.get_running_loop())
    try:
        async with asyncio.timeout(JEV_TIMEOUT):
            response = await client.system_one(
                state=_state_text(question, passage),
                questions={"relevance": RELEVANCE_QUESTION},
            )
    except Exception:
        return 0.0
    return response.nouls["relevance"].noul


async def rerank_documents(
    question: str, documents: list[Document], top_n: int
) -> list[Document]:
    """Score every passage with Jev and keep the Top-N by relevance."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def scored(doc: Document) -> tuple[float, Document]:
        async with semaphore:
            return await _relevance(question, doc.page_content), doc

    scored_docs = await asyncio.gather(*(scored(doc) for doc in documents))
    scored_docs.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_n]]
