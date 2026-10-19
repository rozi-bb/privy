"""Privy graph: retrieve → rerank → generate.

Every question is searched with hybrid retrieval (BM25 + dense, fused with
RRF), the Top-K candidates are reranked by Jev, and the Top-N passages are
given to DeepSeek together with the question. The system prompt forces
grounding: answers come from the passages only, cite their articles, and
admit when the regulation does not cover the question.
"""

from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import START, StateGraph

from agent.corpus import load_gdpr_chunks
from agent.embeddings import LocalEmbeddings
from agent.reranking import rerank_documents
from agent.retrieval import KeywordRetriever, hybrid_search
from agent.state import State

MODEL_NAME = "deepseek-flash"
TOP_K = 20
TOP_N = 4

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://privy:privy@localhost:5433/privy"
)

GENERATE_PROMPT = """\
You are Privy, an assistant that answers questions about the GDPR \
(the EU General Data Protection Regulation) from the official regulation text.

Answer ONLY from the passages below. For every statement, cite the article \
or recital number it comes from, e.g. "(Article 17)" or "(Recital 26)".
If the passages do not contain the answer, say exactly that you could not \
find it in the regulation.
If the question is outside the GDPR's scope or asks for legal advice, \
decline politely in one sentence.
Keep the answer clear and short.

Passages:
{passages}
"""


def _question(state: State) -> str:
    last = state["messages"][-1]
    return last.content if isinstance(last.content, str) else str(last.content)


def _format_passages(documents: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        if meta.get("kind") in ("article", "recital"):
            label = f"{meta['kind'].title()} {meta['number']}"
            if meta.get("point"):
                label = f"{label}({meta['point']})"
            elif meta.get("part"):
                label = f"{label} (part {meta['part']})"
        else:
            label = "Passage"
        blocks.append(f"[{i}] {label}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _vector_store():
    """Load the pgvector store lazily, so importing the graph never needs the DB."""
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=LocalEmbeddings(),
        collection_name="gdpr",
        connection=DATABASE_URL,
        use_jsonb=True,
        async_mode=True,
    )


async def retrieve(state: State) -> dict:
    """Hybrid retrieval: BM25 + dense in parallel, fused with RRF."""
    question = _question(state)
    keyword = KeywordRetriever(load_gdpr_chunks())
    documents = await hybrid_search(
        question,
        keyword,
        _vector_store(),
        keyword_k=TOP_K,
        vector_k=TOP_K,
        top_k=TOP_K,
    )
    return {"documents": documents}


async def rerank(state: State) -> dict:
    """Score the Top-K passages with Jev and keep the Top-N."""
    question = _question(state)
    return {
        "reranked": await rerank_documents(question, state["documents"] or [], TOP_N)
    }


def generate(state: State) -> dict:
    """Write the answer from the Top-N passages, with citations."""
    # stream_usage: report token counts while streaming, for the usage table.
    model = ChatDeepSeek(model=MODEL_NAME, stream_usage=True)
    passages = _format_passages(state["reranked"] or [])
    response = model.invoke(
        [
            SystemMessage(content=GENERATE_PROMPT.format(passages=passages)),
            *state["messages"],
        ]
    )
    return {"messages": [response]}


def build_graph(checkpointer=None):
    """Build the graph; pass a checkpointer for conversation memory.

    Without a checkpointer (the default, used by Studio) the dev server
    persists threads itself. The backend passes a Postgres checkpointer so
    conversations survive restarts.
    """
    workflow = StateGraph(State)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", rerank)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    return workflow.compile(checkpointer=checkpointer)


graph = build_graph()
