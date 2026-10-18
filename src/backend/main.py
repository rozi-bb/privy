"""FastAPI backend for Privy.

A thin server around the graph: a health check and a chat endpoint that runs
the graph with the caller's thread id, so conversation memory is per thread.
The graph is compiled with a Postgres checkpointer, so conversations survive
a restart. Sign-in and thread ownership arrive in later steps (12-13).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from agent.graph import build_graph
from backend.config import DATABASE_URL

app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the pool, set up the checkpointer, and build the graph."""
    # The checkpointer needs autocommit and no prepared statements.
    pool = AsyncConnectionPool(
        DATABASE_URL,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,
    )
    async with pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        app_state["graph"] = build_graph(checkpointer=checkpointer)
        yield


app = FastAPI(title="Privy", lifespan=lifespan)


class ChatRequest(BaseModel):
    """A question and the thread it belongs to."""

    question: str
    thread_id: str = "default"


@app.get("/health")
def health() -> dict:
    """Server is up; the database is checked by the graph when it runs."""
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest, http: Request) -> dict:
    """Run the graph for the question and return the answer."""
    graph = app_state["graph"]
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": request.question}]},
        config={"configurable": {"thread_id": request.thread_id}},
    )
    return {"answer": result["messages"][-1].content}
