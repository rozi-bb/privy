"""Shared graph state."""

from __future__ import annotations

from langchain_core.documents import Document
from langgraph.graph import MessagesState


class State(MessagesState):
    """State for the Privy graph.

    `messages` comes from `MessagesState` (with the `add_messages` reducer).
    `is_small_talk_message` says whether Jev judged the last message to be
    small talk, which is answered without retrieval.
    `search_query` is the last question rewritten to stand on its own, in the
    regulation's terms; retrieval and reranking use it.
    `documents` holds the Top-K passages from retrieval, `reranked` the Top-N
    passages that survived reranking and are shown to the model.
    """

    is_small_talk_message: bool
    search_query: str
    documents: list[Document]
    reranked: list[Document]
