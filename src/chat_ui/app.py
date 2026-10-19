"""Privy chat UI (Chainlit), a client of the backend.

The UI never runs the agent and never calls DeepSeek or Jev: it signs in
through the backend, keeps the backend's login token in the Chainlit
session, and relays each question to the backend's streaming chat endpoint.

    uv run chainlit run src/chat_ui/app.py --port 8001
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import chainlit as cl
import httpx
from chainlit.data.base import BaseDataLayer

logger = logging.getLogger(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Retrieval and reranking run before the first token, so allow a slow start.
ANSWER_TIMEOUT = httpx.Timeout(10.0, read=120.0)
TITLE_CHARS = 80

SESSION_EXPIRED = "Your session has expired. Please sign out and sign in again."
BACKEND_DOWN = "Privy's server cannot be reached right now. Please try again."


class SessionExpired(Exception):
    """The backend refused the login token."""


@cl.data_layer
def no_data_layer() -> BaseDataLayer | None:
    """Use no Chainlit data layer: chat history lives in the backend."""
    # Without this, Chainlit builds its own tables in DATABASE_URL when that
    # variable is set. The backend owns all data; the UI stores nothing.
    return None


async def parse_events(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
    """Turn the backend's Server-Sent Events into their JSON payloads."""
    async for line in lines:
        if line.startswith("data: "):
            yield json.loads(line.removeprefix("data: "))


@cl.password_auth_callback
async def sign_in(email: str, password: str) -> cl.User | None:
    """Check the credentials with the backend and keep its login token."""
    try:
        async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=10.0) as client:
            response = await client.post(
                "/auth/login", json={"email": email, "password": password}
            )
    except httpx.HTTPError:
        logger.warning("Backend unreachable at %s during sign-in", BACKEND_URL)
        return None
    if response.status_code != 200:
        return None
    return cl.User(
        identifier=email.strip().lower(),
        metadata={"token": response.json()["access_token"]},
    )


def _auth_headers() -> dict[str, str]:
    user: cl.User = cl.user_session.get("user")
    return {"Authorization": f"Bearer {user.metadata['token']}"}


async def _thread_id(client: httpx.AsyncClient, first_question: str) -> str:
    """Return this chat's backend thread, creating it on the first question.

    Created lazily so that opening the page does not leave empty threads.
    """
    thread_id = cl.user_session.get("thread_id")
    if thread_id:
        return thread_id
    response = await client.post(
        "/threads",
        json={"title": first_question[:TITLE_CHARS]},
        headers=_auth_headers(),
    )
    if response.status_code == 401:
        raise SessionExpired
    response.raise_for_status()
    thread_id = response.json()["id"]
    cl.user_session.set("thread_id", thread_id)
    return thread_id


@cl.on_message
async def answer(message: cl.Message) -> None:
    """Send the question to the backend and stream the answer back."""
    reply = cl.Message(content="")
    try:
        async with httpx.AsyncClient(
            base_url=BACKEND_URL, timeout=ANSWER_TIMEOUT
        ) as client:
            thread_id = await _thread_id(client, message.content)
            async with client.stream(
                "POST",
                f"/threads/{thread_id}/chat",
                json={"question": message.content},
                headers=_auth_headers(),
            ) as response:
                if response.status_code == 401:
                    raise SessionExpired
                response.raise_for_status()
                async for event in parse_events(response.aiter_lines()):
                    if event["type"] == "token":
                        await reply.stream_token(event["content"])
                    elif event["type"] == "error":
                        await cl.ErrorMessage(content=event["message"]).send()
                        return
    except SessionExpired:
        await cl.ErrorMessage(content=SESSION_EXPIRED).send()
        return
    except httpx.HTTPError:
        logger.exception("Request to the backend failed")
        await cl.ErrorMessage(content=BACKEND_DOWN).send()
        return
    await reply.send()
