"""Privy graph: rewrite_query → retrieve → rerank → generate → verify.

The question is first rewritten into a standalone search query in the
regulation's own terms, so "who do I report it to?" after a question about
breaches is searched as breach notification to the supervisory authority,
and "can I sue?" as judicial remedy and compensation. The query goes through hybrid retrieval (BM25 + dense, fused with RRF) and Jev
reranking, which keeps only passages that are actually relevant. With none
left, the graph answers that the regulation does not cover the question
without calling the model. Otherwise DeepSeek answers from the passages, and
`verify` checks that every article it cites was among them.
"""

from __future__ import annotations

import os
from functools import cache

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from agent.citations import allowed_labels, invalid_citations
from agent.corpus import load_gdpr_chunks
from agent.embeddings import LocalEmbeddings
from agent.reranking import rerank_documents
from agent.retrieval import KeywordRetriever, hybrid_search
from agent.state import State

MODEL_NAME = "deepseek-flash"
TOP_K = 20
TOP_N = 4
# Jev's probability that a passage helps answer the question. Tuned in the
# evaluation (roadmap step 13).
MIN_RELEVANCE = 0.5

HISTORY_MESSAGES = 6
HISTORY_CHARS = 600

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://privy:privy@localhost:5433/privy"
)

REWRITE_PROMPT = """\
You turn the user's last message into a search query over the text of the \
GDPR (Regulation (EU) 2016/679).
- Resolve words such as "it", "that" or "they" from the conversation, so the \
query stands on its own.
- Use the regulation's own terms for everyday words, for example "delete my \
data" becomes "right to erasure", and "hackers stole our customer list" \
becomes "personal data breach notification".
- Keep it to one or two lines and do not answer the question.
- If the message has nothing to do with data protection, return it unchanged.
Return only the query."""

GENERATE_PROMPT = """\
You are Privy, an assistant that answers questions about the GDPR \
(the EU General Data Protection Regulation) from the official regulation text.

Rules:
- Answer ONLY from the passages below, never from other knowledge.
- After each statement, cite the passage it comes from, e.g. "(Article 17)", \
"(Article 4(1))" or "(Recital 26)". Cite only articles and recitals that \
appear in the passages.
- If the passages do not answer the question, say that you could not find it \
in the regulation and offer to help if the user rephrases.
- If the user asks for a judgment about their own situation (for example \
whether they can sue, or whether a company broke the law), do not give one: \
explain what the passages say, then suggest a lawyer or their national data \
protection authority.
- If the question has nothing to do with data protection, decline politely in \
one sentence.
- Keep it short and in plain language: the answer first, then the details.

Passages:
{passages}
"""

RETRY_NOTE = """
Your previous answer cited {invalid}, which are not among the passages. \
Write the answer again and cite only these: {allowed}."""

NOT_FOUND_ANSWER = (
    "I could not find an answer to that in the GDPR. I answer questions about "
    "the EU General Data Protection Regulation from its official text; could "
    "you rephrase, or ask about a specific right or obligation?"
)

UNVERIFIED_ANSWER = (
    "I could not write an answer that I can back with the regulation's text. "
    "Could you rephrase the question, or ask about a specific article?"
)

_ROLES = {"human": "User", "ai": "Assistant"}


def _model(temperature: float | None = None) -> ChatDeepSeek:
    # stream_usage: report token counts while streaming, for the usage table.
    return ChatDeepSeek(model=MODEL_NAME, stream_usage=True, temperature=temperature)


@cache
def _embeddings() -> LocalEmbeddings:
    # Loading the ONNX model takes seconds; do it once per process.
    return LocalEmbeddings()


@cache
def _keyword_retriever() -> KeywordRetriever:
    return KeywordRetriever(load_gdpr_chunks())


def _vector_store():
    """Open the pgvector store per call, so importing the graph never needs the DB."""
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=_embeddings(),
        collection_name="gdpr",
        connection=DATABASE_URL,
        use_jsonb=True,
        async_mode=True,
    )


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


def _history(messages: list[BaseMessage]) -> str:
    lines = [
        f"{_ROLES[m.type]}: {m.text[:HISTORY_CHARS]}"
        for m in messages[-HISTORY_MESSAGES:]
        if m.type in _ROLES
    ]
    return "\n".join(lines)


def rewrite_query(state: State) -> dict:
    """Turn the last message into a standalone query in the regulation's terms.

    Users ask in everyday words ("can I sue?") while the GDPR says "judicial
    remedy" and "compensation"; neither BM25 nor the small embedding model
    bridges that gap, so the model rewrites the question before the search.
    """
    messages = state["messages"]
    question = messages[-1].text
    history = _history(messages[:-1])
    context = f"Conversation:\n{history}\n\n" if history else ""
    response = _model(temperature=0).invoke(
        [
            SystemMessage(content=REWRITE_PROMPT),
            HumanMessage(content=f"{context}Last message: {question}"),
        ]
    )
    return {"search_query": response.text.strip() or question}


async def retrieve(state: State) -> dict:
    """Hybrid retrieval: BM25 + dense in parallel, fused with RRF."""
    documents = await hybrid_search(
        state["search_query"],
        _keyword_retriever(),
        _vector_store(),
        keyword_k=TOP_K,
        vector_k=TOP_K,
        top_k=TOP_K,
    )
    return {"documents": documents}


async def rerank(state: State) -> dict:
    """Score the Top-K passages with Jev and keep the relevant Top-N."""
    reranked = await rerank_documents(
        state["search_query"], state["documents"] or [], TOP_N, MIN_RELEVANCE
    )
    return {"reranked": reranked}


def route_after_rerank(state: State) -> str:
    """Answer only when some passage is relevant enough."""
    return "generate" if state["reranked"] else "not_found"


def not_found(state: State) -> dict:
    """Say the regulation does not cover the question, without the model."""
    get_stream_writer()({"type": "token", "content": NOT_FOUND_ANSWER})
    return {"messages": [AIMessage(content=NOT_FOUND_ANSWER)], "reranked": []}


def generate(state: State) -> dict:
    """Write the answer from the Top-N passages, with citations."""
    passages = _format_passages(state["reranked"])
    response = _model().invoke(
        [
            SystemMessage(content=GENERATE_PROMPT.format(passages=passages)),
            *state["messages"],
        ]
    )
    return {"messages": [response]}


def verify(state: State) -> dict:
    """Make sure every citation in the answer comes from the passages.

    An answer citing anything else is written again once with the allowed
    citations spelled out; if it still cites something else, it is replaced
    by an honest refusal. The replacement keeps the answer's message id, so
    it overwrites the answer in the conversation, and is also sent on the
    custom stream because the client has already seen the first version.
    """
    answer = state["messages"][-1]
    passages = state["reranked"]
    invalid = invalid_citations(answer.text, passages)
    if not invalid:
        return {}

    prompt = GENERATE_PROMPT.format(
        passages=_format_passages(passages)
    ) + RETRY_NOTE.format(
        invalid=", ".join(sorted(str(c) for c in invalid)),
        allowed=", ".join(allowed_labels(passages)),
    )
    retry = _model().invoke([SystemMessage(content=prompt), *state["messages"][:-1]])
    content = (
        retry.text if not invalid_citations(retry.text, passages) else UNVERIFIED_ANSWER
    )

    get_stream_writer()({"type": "replace", "content": content})
    return {"messages": [AIMessage(content=content, id=answer.id)]}


def build_graph(checkpointer=None):
    """Build the graph; pass a checkpointer for conversation memory.

    Without a checkpointer (the default, used by Studio) the dev server
    persists threads itself. The backend passes a Postgres checkpointer so
    conversations survive restarts.
    """
    workflow = StateGraph(State)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", rerank)
    workflow.add_node("not_found", not_found)
    workflow.add_node("generate", generate)
    workflow.add_node("verify", verify)
    workflow.add_edge(START, "rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_conditional_edges(
        "rerank", route_after_rerank, ["generate", "not_found"]
    )
    workflow.add_edge("generate", "verify")
    workflow.add_edge("verify", END)
    workflow.add_edge("not_found", END)
    return workflow.compile(checkpointer=checkpointer)


graph = build_graph()
