"""Chainlit data layer that reads chat history from the backend.

Chainlit shows earlier conversations in its sidebar only when it has a data
layer. Ours stores nothing: the backend already owns threads and messages,
so the sidebar list comes from `GET /threads` and opening a thread from
`GET /threads/{id}/messages`. Writes Chainlit makes while a chat runs
(steps, thread names) are ignored, because the backend records the
conversation itself.

Data-layer methods receive a user or thread id, not the request, so the layer
remembers which backend token belongs to which user (learned at sign-in, or
from the signed Chainlit session after a UI restart) and which user owns
which thread (learned when that user's threads are listed or created).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from chainlit.data.base import BaseDataLayer
from chainlit.types import (
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User

ASSISTANT_NAME = "Privy"


class ThreadDeletionUnsupported(Exception):
    """The backend keeps every thread (usage records refer to them)."""


class BackendDataLayer(BaseDataLayer):
    """Sidebar history served by the backend API."""

    def __init__(
        self, backend_url: str, transport: httpx.AsyncBaseTransport | None = None
    ):
        """Point the layer at the backend; tests pass a fake transport."""
        self._backend_url = backend_url
        self._transport = transport
        self._tokens: dict[str, str] = {}
        self._owners: dict[str, str] = {}

    def remember_token(self, identifier: str, token: str) -> None:
        """Record the backend token of a user who just signed in."""
        self._tokens[identifier] = token

    def remember_owner(self, thread_id: str, identifier: str) -> None:
        """Record that a user owns a thread (the UI just created it)."""
        self._owners[thread_id] = identifier

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._backend_url, transport=self._transport, timeout=10.0
        )

    async def _get(self, identifier: str, path: str) -> Any | None:
        token = self._tokens.get(identifier)
        if token is None:
            return None
        async with self._client() as client:
            response = await client.get(
                path, headers={"Authorization": f"Bearer {token}"}
            )
        if response.status_code != 200:
            return None
        return response.json()

    def _persisted(self, identifier: str) -> PersistedUser:
        return PersistedUser(
            id=identifier,
            identifier=identifier,
            createdAt=datetime.now(UTC).isoformat(),
            metadata={"token": self._tokens[identifier]},
        )

    async def get_user(self, identifier: str) -> PersistedUser | None:
        """Return the user if their backend token is known.

        Returning None makes Chainlit call create_user with the signed
        session's user, whose metadata carries the token, so a UI restart
        recovers the token without asking the user to sign in again.
        """
        return self._persisted(identifier) if identifier in self._tokens else None

    async def create_user(self, user: User) -> PersistedUser | None:
        """Learn the token from the signed-in user's metadata."""
        token = (user.metadata or {}).get("token")
        if not token:
            return None
        self.remember_token(user.identifier, token)
        return self._persisted(user.identifier)

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        """List the user's threads, newest first, for the sidebar."""
        identifier = filters.userId or ""
        threads = await self._get(identifier, "/threads") or []
        if filters.search:
            threads = [
                t for t in threads if filters.search.lower() in t["title"].lower()
            ]
        for thread in threads:
            self.remember_owner(thread["id"], identifier)

        start = 0
        if pagination.cursor:
            ids = [t["id"] for t in threads]
            start = (
                ids.index(pagination.cursor) + 1
                if pagination.cursor in ids
                else len(ids)
            )
        page = threads[start : start + pagination.first]
        return PaginatedResponse(
            pageInfo=PageInfo(
                hasNextPage=start + pagination.first < len(threads),
                startCursor=page[0]["id"] if page else None,
                endCursor=page[-1]["id"] if page else None,
            ),
            data=[self._thread_dict(t, identifier, steps=[]) for t in page],
        )

    async def get_thread(self, thread_id: str) -> ThreadDict | None:
        """Return a thread with its messages, for resuming it from the sidebar."""
        identifier = self._owners.get(thread_id)
        if identifier is None:
            return None
        thread = next(
            (
                t
                for t in await self._get(identifier, "/threads") or []
                if t["id"] == thread_id
            ),
            None,
        )
        messages = await self._get(identifier, f"/threads/{thread_id}/messages")
        if thread is None or messages is None:
            return None
        return self._thread_dict(
            thread, identifier, steps=self._steps(thread, messages)
        )

    async def get_thread_author(self, thread_id: str) -> str:
        """Return the owner's identifier, or "" when the thread is unknown."""
        return self._owners.get(thread_id, "")

    @staticmethod
    def _thread_dict(thread: dict, identifier: str, steps: list) -> ThreadDict:
        return ThreadDict(
            id=thread["id"],
            createdAt=thread["created_at"],
            name=thread["title"],
            userId=identifier,
            userIdentifier=identifier,
            tags=None,
            metadata={},
            steps=steps,
            elements=[],
        )

    @staticmethod
    def _steps(thread: dict, messages: list[dict]) -> list[dict]:
        # The backend stores no per-message time; spacing the messages from
        # the thread's start keeps Chainlit's ordering by createdAt correct.
        start = datetime.fromisoformat(thread["created_at"])
        steps = []
        for i, message in enumerate(messages):
            is_user = message["role"] == "user"
            created = (start + timedelta(milliseconds=i)).isoformat()
            steps.append(
                {
                    "id": f"{thread['id']}-{i}",
                    "threadId": thread["id"],
                    "parentId": None,
                    "name": "User" if is_user else ASSISTANT_NAME,
                    "type": "user_message" if is_user else "assistant_message",
                    "output": message["content"],
                    "createdAt": created,
                    "start": created,
                    "end": created,
                    "metadata": {},
                }
            )
        return steps

    async def delete_thread(self, thread_id: str) -> None:
        """Refuse: deleting would also drop the usage records of the thread."""
        raise ThreadDeletionUnsupported(thread_id)

    # The backend records the conversation itself; Chainlit's own writes
    # during a chat have nowhere to go and are dropped.

    async def update_thread(
        self,
        thread_id: str,
        name: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Ignore: thread titles are set by the backend on creation."""

    async def create_step(self, step_dict: Any) -> None:
        """Ignore: messages are stored by the backend's checkpointer."""

    async def update_step(self, step_dict: Any) -> None:
        """Ignore: messages are stored by the backend's checkpointer."""

    async def delete_step(self, step_id: str) -> None:
        """Ignore: messages are stored by the backend's checkpointer."""

    async def upsert_feedback(self, feedback: Any) -> str:
        """Ignore feedback; there is nowhere to store it yet."""
        return ""

    async def delete_feedback(self, feedback_id: str) -> bool:
        """Ignore feedback; there is nowhere to store it yet."""
        return True

    async def create_element(self, element: Any) -> None:
        """Ignore: file upload is off."""

    async def get_element(self, thread_id: str, element_id: str) -> None:
        """Return nothing: file upload is off."""
        return None

    async def delete_element(
        self, element_id: str, thread_id: str | None = None
    ) -> None:
        """Ignore: file upload is off."""

    async def get_favorite_steps(self, user_id: str) -> list:
        """Return nothing: favorites are not used."""
        return []

    async def build_debug_url(self) -> str:
        """Return no debug URL."""
        return ""

    async def close(self) -> None:
        """Nothing to close: each call opens its own HTTP client."""
