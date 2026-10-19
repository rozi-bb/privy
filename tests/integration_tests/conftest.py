"""Fixtures for the backend tests.

The backend tests need the Postgres container (docker compose up -d) but no
API keys: the real graph is replaced by a fake one that answers instantly,
so they cost nothing to run. They use their own database, privy_test.
"""

import os
from collections.abc import Callable, Iterator
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from backend import config
from backend.main import create_app

from .backend_helpers import build_fake_graph

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://privy:privy@localhost:5433/privy_test"
)
ADMIN_DATABASE_URL = TEST_DATABASE_URL.rsplit("/", 1)[0] + "/postgres"


@pytest.fixture(scope="session")
def test_database() -> str:
    try:
        with psycopg.connect(ADMIN_DATABASE_URL, autocommit=True) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = 'privy_test'"
            ).fetchone()
            if not exists:
                conn.execute("CREATE DATABASE privy_test")
    except psycopg.OperationalError:
        pytest.skip("Postgres is not running (docker compose up -d)")
    return TEST_DATABASE_URL


@pytest.fixture
def make_client(
    test_database: str, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., TestClient]]:
    """Start the app against the test database; tables are emptied first."""
    monkeypatch.setattr(config, "JWT_SECRET", "test-secret-" + "x" * 32)
    clients: list[TestClient] = []
    emptied = False

    def make(build: Callable[[Any], Any] = build_fake_graph) -> TestClient:
        nonlocal emptied
        client = TestClient(create_app(database_url=test_database, build=build))
        client.__enter__()
        clients.append(client)
        if not emptied:
            with psycopg.connect(test_database, autocommit=True) as conn:
                conn.execute("TRUNCATE users, threads, usage CASCADE")
            emptied = True
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


@pytest.fixture
def db_rows(test_database: str) -> Callable[[str], list[tuple]]:
    """Run a query against the test database and return the rows."""

    def query(sql: str) -> list[tuple]:
        with psycopg.connect(test_database) as conn:
            return conn.execute(sql).fetchall()

    return query
