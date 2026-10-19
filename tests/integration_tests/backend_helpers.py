"""Shared helpers for the backend tests: fake graphs, sign-up, SSE parsing."""

import json
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.graph import START, MessagesState, StateGraph

FAKE_USAGE = {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}


def build_fake_graph(checkpointer: Any) -> Any:
    """A one-node graph whose 'generate' node echoes the question."""

    def generate(state: MessagesState) -> dict:
        question = state["messages"][-1].text
        return {
            "messages": [
                AIMessage(content=f"Answer to: {question}", usage_metadata=FAKE_USAGE)
            ]
        }

    workflow = StateGraph(MessagesState)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "generate")
    return workflow.compile(checkpointer=checkpointer)


def failing_graph_builder(error: Exception) -> Callable[[Any], Any]:
    """A graph whose 'generate' node raises the given error."""

    def build(checkpointer: Any) -> Any:
        def generate(state: MessagesState) -> dict:
            raise error

        workflow = StateGraph(MessagesState)
        workflow.add_node("generate", generate)
        workflow.add_edge(START, "generate")
        return workflow.compile(checkpointer=checkpointer)

    return build


def sign_up(client: TestClient, email: str, password: str = "correct-horse") -> dict:
    """Register and sign in; return the Authorization header."""
    assert (
        client.post(
            "/auth/register", json={"email": email, "password": password}
        ).status_code
        == 201
    )
    response = client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def sse_events(text: str) -> list[dict]:
    """Parse a Server-Sent Events body into its JSON payloads."""
    return [
        json.loads(block.removeprefix("data: "))
        for block in text.split("\n\n")
        if block.startswith("data: ")
    ]
