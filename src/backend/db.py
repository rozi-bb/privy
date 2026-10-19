"""App data in Postgres: users, threads, and usage.

Plain SQL through psycopg. Every function takes the shared connection pool,
which runs in autocommit mode (the checkpointer needs it), so each statement
is its own transaction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

SCHEMA = Path(__file__).with_name("schema.sql")

Row = dict[str, Any]


class EmailTaken(Exception):
    """Raised when registering an email that already has an account."""


async def apply_schema(pool: AsyncConnectionPool) -> None:
    """Create the app tables if they do not exist yet."""
    async with pool.connection() as conn:
        # A multi-statement script cannot be a prepared statement, and the
        # pool prepares everything (prepare_threshold=0).
        await conn.execute(SCHEMA.read_text(), prepare=False)


async def _fetch_one(pool: AsyncConnectionPool, sql: str, params: tuple) -> Row | None:
    async with pool.connection() as conn:
        cur = await conn.cursor(row_factory=dict_row).execute(sql, params)
        return await cur.fetchone()


async def create_user(pool: AsyncConnectionPool, email: str, password_hash: str) -> Row:
    """Insert a user; raise EmailTaken if the email is already registered."""
    try:
        row = await _fetch_one(
            pool,
            "INSERT INTO users (email, password_hash) VALUES (%s, %s)"
            " RETURNING id, email, created_at",
            (email, password_hash),
        )
    except UniqueViolation as error:
        raise EmailTaken(email) from error
    assert row is not None
    return row


async def get_user_by_email(pool: AsyncConnectionPool, email: str) -> Row | None:
    """Return the user with this email, including the password hash."""
    return await _fetch_one(
        pool, "SELECT id, email, password_hash FROM users WHERE email = %s", (email,)
    )


async def get_user(pool: AsyncConnectionPool, user_id: UUID) -> Row | None:
    """Return the user with this id."""
    return await _fetch_one(
        pool, "SELECT id, email, created_at FROM users WHERE id = %s", (user_id,)
    )


async def create_thread(pool: AsyncConnectionPool, user_id: UUID, title: str) -> Row:
    """Start a new thread owned by the user."""
    row = await _fetch_one(
        pool,
        "INSERT INTO threads (user_id, title) VALUES (%s, %s)"
        " RETURNING id, title, created_at",
        (user_id, title),
    )
    assert row is not None
    return row


async def get_thread(
    pool: AsyncConnectionPool, thread_id: UUID, user_id: UUID
) -> Row | None:
    """Return the thread only if this user owns it.

    Filtering by owner in the query itself means another user's thread looks
    exactly like a thread that does not exist.
    """
    return await _fetch_one(
        pool,
        "SELECT id, title, created_at FROM threads WHERE id = %s AND user_id = %s",
        (thread_id, user_id),
    )


async def list_threads(pool: AsyncConnectionPool, user_id: UUID) -> list[Row]:
    """Return the user's threads, newest first."""
    async with pool.connection() as conn:
        cur = await conn.cursor(row_factory=dict_row).execute(
            "SELECT id, title, created_at FROM threads"
            " WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,),
        )
        return await cur.fetchall()


async def record_usage(
    pool: AsyncConnectionPool,
    user_id: UUID,
    thread_id: UUID,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Store the token counts of one answered question."""
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO usage (user_id, thread_id, model, input_tokens, output_tokens)"
            " VALUES (%s, %s, %s, %s, %s)",
            (user_id, thread_id, model, input_tokens, output_tokens),
        )
