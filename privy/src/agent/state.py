"""Shared graph state."""

from __future__ import annotations

from langchain_core.documents import Document
from langgraph.graph import MessagesState


class State(MessagesState):
    """State for the Privy graph.

    `messages` comes from `MessagesState` (with the `add_messages` reducer).
    `documents` holds the Top-K passages from retrieval, `reranked` the Top-N
    passages that survived reranking and are shown to the model.
    """

    documents: list[Document]
    reranked: list[Document]
