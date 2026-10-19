"""FastAPI backend for Privy.

The graph runs inside this server (architecture.md §4, option A). Users
register and sign in, own their threads, and chat in them; answers stream
back as Server-Sent Events and every answered question is recorded in the
usage table. Conversations are stored by the Postgres checkpointer under the
thread id, so they survive a restart.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field, field_validator

from agent.graph import MODEL_NAME, build_graph
from backend import config, db
from backend.auth import CurrentUser, create_token, hash_password, verify_password

logger = logging.getLogger(__name__)

GraphBuilder = Callable[..., Any]

router = APIRouter()


class Credentials(BaseModel):
    """Email and password, for registering and signing in."""

    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, email: str) -> str:
        email = email.strip().lower()
        if "@" not in email:
            raise ValueError("not an email address")
        return email


class UserOut(BaseModel):
    """A user as shown to the client (never the password hash)."""

    id: UUID
    email: str


class Token(BaseModel):
    """A login token for the Authorization header."""

    access_token: str
    token_type: str = "bearer"


class NewThread(BaseModel):
    """Options for a new thread.

    `id` lets a client use its own id for the thread (the chat UI passes its
    session's thread id so its sidebar and the backend agree on it).
    """

    title: str = Field(default="New chat", min_length=1, max_length=200)
    id: UUID | None = None


class ThreadOut(BaseModel):
    """A thread the user owns."""

    id: UUID
    title: str
    created_at: datetime


class MessageOut(BaseModel):
    """One turn of a conversation."""

    role: str
    content: str


class Question(BaseModel):
    """A question asked in a thread."""

    question: str = Field(min_length=1, max_length=4000)


@router.get("/health")
def health() -> dict[str, str]:
    """Server is up; the database is checked by the graph when it runs."""
    return {"status": "ok"}


@router.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(body: Credentials, request: Request) -> UserOut:
    """Create an account."""
    try:
        user = await db.create_user(
            request.app.state.pool, body.email, hash_password(body.password)
        )
    except db.EmailTaken as error:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Email already registered"
        ) from error
    return UserOut(**user)


@router.post("/auth/login")
async def login(body: Credentials, request: Request) -> Token:
    """Exchange email and password for a login token."""
    user = await db.get_user_by_email(request.app.state.pool, body.email)
    password_ok = verify_password(
        body.password, user["password_hash"] if user else None
    )
    if user is None or not password_ok:
        # One message for both cases, so the response does not reveal which
        # emails have an account.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong email or password")
    return Token(access_token=create_token(user["id"]))


@router.get("/me")
async def me(user: CurrentUser) -> UserOut:
    """Return the signed-in user."""
    return UserOut(**user)


@router.post("/threads", status_code=status.HTTP_201_CREATED)
async def create_thread(
    body: NewThread, request: Request, user: CurrentUser
) -> ThreadOut:
    """Start a new thread."""
    try:
        thread = await db.create_thread(
            request.app.state.pool, user["id"], body.title, body.id
        )
    except db.ThreadIdTaken as error:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Thread id already in use"
        ) from error
    return ThreadOut(**thread)


@router.get("/threads")
async def list_threads(request: Request, user: CurrentUser) -> list[ThreadOut]:
    """List the user's threads, newest first."""
    threads = await db.list_threads(request.app.state.pool, user["id"])
    return [ThreadOut(**thread) for thread in threads]


async def _owned_thread(
    request: Request, thread_id: UUID, user: dict[str, Any]
) -> None:
    # 404 rather than 403: another user's thread must be indistinguishable
    # from one that does not exist.
    if await db.get_thread(request.app.state.pool, thread_id, user["id"]) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Thread not found")


def _graph_config(thread_id: UUID) -> dict[str, Any]:
    return {"configurable": {"thread_id": str(thread_id)}}


_ROLES = {"human": "user", "ai": "assistant"}


@router.get("/threads/{thread_id}/messages")
async def thread_messages(
    thread_id: UUID, request: Request, user: CurrentUser
) -> list[MessageOut]:
    """Return the conversation stored in a thread."""
    await _owned_thread(request, thread_id, user)
    state = await request.app.state.graph.aget_state(_graph_config(thread_id))
    return [
        MessageOut(role=_ROLES[message.type], content=message.text)
        for message in state.values.get("messages", [])
        if message.type in _ROLES
    ]


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _answer_events(
    request: Request, user_id: UUID, thread_id: UUID, question: str
) -> AsyncIterator[str]:
    """Run the graph and turn its output into Server-Sent Events.

    Events: {"type": "token", "content": ...} for each piece of the answer,
    {"type": "replace", "content": ...} when the graph corrects an answer it
    already streamed, then {"type": "done"}, or {"type": "error", ...} if the
    run failed. Answer tokens come from the `generate` node's model stream;
    answers written without the model (and replacements) arrive on the
    graph's custom stream already in event form.
    """
    input_tokens = output_tokens = 0
    try:
        async for mode, data in request.app.state.graph.astream(
            {"messages": [{"role": "user", "content": question}]},
            config=_graph_config(thread_id),
            stream_mode=["messages", "custom"],
        ):
            if mode == "custom":
                yield _sse(data)
                continue
            message, metadata = data
            if metadata.get("langgraph_node") == "generate" and message.text:
                yield _sse({"type": "token", "content": message.text})
            # Every model call counts (the follow-up rewrite and a citation
            # retry too). With streaming, DeepSeek reports token counts on the
            # last chunk of each call.
            usage = getattr(message, "usage_metadata", None)
            if usage:
                input_tokens += usage["input_tokens"]
                output_tokens += usage["output_tokens"]
    except Exception:
        logger.exception("Answer failed in thread %s", thread_id)
        yield _sse(
            {"type": "error", "message": "Privy could not answer. Please try again."}
        )
        return

    await db.record_usage(
        request.app.state.pool,
        user_id,
        thread_id,
        MODEL_NAME,
        input_tokens,
        output_tokens,
    )
    yield _sse({"type": "done"})


@router.post("/threads/{thread_id}/chat")
async def chat(
    thread_id: UUID, body: Question, request: Request, user: CurrentUser
) -> StreamingResponse:
    """Ask a question in a thread and stream the answer."""
    # Checked before streaming starts, so a refusal is a normal HTTP error.
    await _owned_thread(request, thread_id, user)
    return StreamingResponse(
        _answer_events(request, user["id"], thread_id, body.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def create_app(
    database_url: str = config.DATABASE_URL, build: GraphBuilder = build_graph
) -> FastAPI:
    """Build the app; tests pass their own database and graph."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not config.JWT_SECRET:
            raise RuntimeError("JWT_SECRET is not set; add it to .env")
        # Connection settings the Postgres checkpointer requires.
        pool = AsyncConnectionPool(
            database_url,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,
        )
        async with pool:
            await db.apply_schema(pool)
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            app.state.pool = pool
            app.state.graph = build(checkpointer=checkpointer)
            yield

    app = FastAPI(title="Privy", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
