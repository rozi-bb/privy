"""Privy baseline graph.

A single-node graph that answers with a chat model and no tools yet.
Each feature from step 9 onward replaces or extends this file.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import MessagesState, START, StateGraph

MODEL_NAME = "deepseek-flash"

SYSTEM_PROMPT = (
    "You are Privy, an assistant that answers questions about the GDPR "
    "(the EU General Data Protection Regulation). "
    "Keep answers clear and short, and mention the article numbers you rely on. "
    "If you do not know, say so instead of guessing."
)


def call_model(state: MessagesState) -> dict:
    """Reply to the conversation with the chat model."""
    model = ChatDeepSeek(model=MODEL_NAME)
    response = model.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
    return {"messages": [response]}


graph = (
    StateGraph(MessagesState)
    .add_node("call_model", call_model)
    .add_edge(START, "call_model")
    .compile(name="Privy")
)
