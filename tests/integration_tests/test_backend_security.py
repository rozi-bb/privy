"""Security tests (roadmap step 22).

Thread isolation, no endpoint without sign-in, and no secret in any response.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import jwt
from fastapi.routing import APIRoute

from backend import config
from backend.main import router

from .backend_helpers import failing_graph_builder, sign_up, sse_events

PUBLIC_ROUTES = {"/health", "/auth/register", "/auth/login"}


def test_every_other_endpoint_needs_sign_in(client) -> None:
    # The app's own route list nests included routers, so read the router.
    routes = [
        route
        for route in router.routes
        if isinstance(route, APIRoute) and route.path not in PUBLIC_ROUTES
    ]
    assert routes, "no protected routes found"
    for route in routes:
        path = route.path.replace("{thread_id}", str(uuid4()))
        for method in route.methods:
            response = client.request(method, path, json={"question": "Hi?"})
            assert response.status_code == 401, f"{method} {route.path}"


def _token(secret: str, sub: str, expires_in: timedelta) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {"sub": sub, "iat": now, "exp": now + expires_in}, secret, "HS256"
    )


def test_bad_tokens_are_refused(client, db_rows) -> None:
    sign_up(client, "ana@example.com")
    (user_id,) = db_rows("SELECT id::text FROM users")[0]
    bad_tokens = {
        "garbage": "not-a-jwt",
        "wrong secret": _token(
            "another-secret-" + "y" * 32, user_id, timedelta(hours=1)
        ),
        "expired": _token(config.JWT_SECRET, user_id, timedelta(seconds=-1)),
        "not a user id": _token(config.JWT_SECRET, "admin", timedelta(hours=1)),
    }
    for name, token in bad_tokens.items():
        response = client.get("/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 401, name


def test_token_of_a_deleted_user_stops_working(client, db_rows, test_database) -> None:
    import psycopg

    headers = sign_up(client, "ana@example.com")
    with psycopg.connect(test_database, autocommit=True) as conn:
        conn.execute("DELETE FROM users")
    assert client.get("/me", headers=headers).status_code == 401


def test_users_cannot_open_each_others_threads(client) -> None:
    ana = sign_up(client, "ana@example.com")
    bob = sign_up(client, "bob@example.com")
    thread = client.post("/threads", json={}, headers=ana).json()
    client.post(
        f"/threads/{thread['id']}/chat", json={"question": "Secret?"}, headers=ana
    )

    read = client.get(f"/threads/{thread['id']}/messages", headers=bob)
    write = client.post(
        f"/threads/{thread['id']}/chat", json={"question": "Hi?"}, headers=bob
    )
    missing = client.get(f"/threads/{uuid4()}/messages", headers=bob)

    assert read.status_code == write.status_code == 404
    # Someone else's thread looks exactly like one that does not exist.
    assert read.json() == missing.json()
    assert client.get("/threads", headers=bob).json() == []


def test_bob_cannot_write_into_anas_conversation(client) -> None:
    ana = sign_up(client, "ana@example.com")
    bob = sign_up(client, "bob@example.com")
    thread = client.post("/threads", json={}, headers=ana).json()
    client.post(
        f"/threads/{thread['id']}/chat", json={"question": "Mine?"}, headers=bob
    )

    messages = client.get(f"/threads/{thread['id']}/messages", headers=ana).json()
    assert messages == []


def test_password_hash_is_never_returned(client) -> None:
    response = client.post(
        "/auth/register", json={"email": "ana@example.com", "password": "correct-horse"}
    )
    headers = sign_up(client, "bob@example.com")
    assert set(response.json()) == {"id", "email"}
    assert set(client.get("/me", headers=headers).json()) == {"id", "email"}


def test_api_keys_never_reach_the_client(make_client, monkeypatch) -> None:
    secret = "sk-test-" + uuid4().hex
    for name in ("DEEPSEEK_API_KEY", "TYPESAFE_API_KEY", "LANGSMITH_API_KEY"):
        monkeypatch.setenv(name, secret)
    # Provider errors can quote the key; the client must only see a generic message.
    client = make_client(failing_graph_builder(RuntimeError(f"401 bad key {secret}")))
    headers = sign_up(client, "ana@example.com")
    thread = client.post("/threads", json={}, headers=headers).json()

    responses = [
        client.post(
            f"/threads/{thread['id']}/chat", json={"question": "Hi?"}, headers=headers
        ),
        client.get(f"/threads/{thread['id']}/messages", headers=headers),
        client.get("/threads", headers=headers),
        client.get("/me", headers=headers),
        client.get("/openapi.json"),
    ]
    assert sse_events(responses[0].text)[-1]["type"] == "error"
    for response in responses:
        assert secret not in response.text
        assert secret not in str(response.headers)


def test_choosing_someone_elses_thread_id_is_refused(client) -> None:
    ana = sign_up(client, "ana@example.com")
    bob = sign_up(client, "bob@example.com")
    thread = client.post("/threads", json={}, headers=ana).json()
    client.post(
        f"/threads/{thread['id']}/chat", json={"question": "Secret?"}, headers=ana
    )

    taken = client.post("/threads", json={"id": thread["id"]}, headers=bob)
    assert taken.status_code == 409
    read = client.get(f"/threads/{thread['id']}/messages", headers=bob)
    assert read.status_code == 404
