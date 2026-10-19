"""Business-flow tests (roadmap step 20): sign-in, chat, usage recorded."""

from agent.graph import MODEL_NAME

from .backend_helpers import FAKE_USAGE, failing_graph_builder, sign_up, sse_events


def test_register_then_sign_in(client) -> None:
    headers = sign_up(client, "ana@example.com")
    response = client.get("/me", headers=headers)
    assert response.status_code == 200
    assert response.json()["email"] == "ana@example.com"


def test_email_is_case_insensitive(client) -> None:
    sign_up(client, "Ana@Example.com")
    response = client.post(
        "/auth/login", json={"email": "ANA@example.COM", "password": "correct-horse"}
    )
    assert response.status_code == 200


def test_same_email_cannot_register_twice(client) -> None:
    sign_up(client, "ana@example.com")
    response = client.post(
        "/auth/register", json={"email": "ana@example.com", "password": "another-one"}
    )
    assert response.status_code == 409


def test_wrong_password_is_refused(client) -> None:
    sign_up(client, "ana@example.com")
    response = client.post(
        "/auth/login", json={"email": "ana@example.com", "password": "wrong-horse"}
    )
    assert response.status_code == 401


def test_unknown_email_gets_the_same_answer_as_a_wrong_password(client) -> None:
    sign_up(client, "ana@example.com")
    wrong_password = client.post(
        "/auth/login", json={"email": "ana@example.com", "password": "wrong-horse"}
    )
    unknown_email = client.post(
        "/auth/login", json={"email": "bob@example.com", "password": "wrong-horse"}
    )
    assert unknown_email.status_code == wrong_password.status_code == 401
    assert unknown_email.json() == wrong_password.json()


def test_short_password_is_rejected(client) -> None:
    response = client.post(
        "/auth/register", json={"email": "ana@example.com", "password": "short"}
    )
    assert response.status_code == 422


def test_chat_streams_the_answer(client) -> None:
    headers = sign_up(client, "ana@example.com")
    thread = client.post("/threads", json={}, headers=headers).json()

    response = client.post(
        f"/threads/{thread['id']}/chat",
        json={"question": "What is personal data?"},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = sse_events(response.text)
    answer = "".join(e["content"] for e in events if e["type"] == "token")
    assert answer == "Answer to: What is personal data?"
    assert events[-1] == {"type": "done"}


def test_usage_is_recorded_per_question(client, db_rows) -> None:
    headers = sign_up(client, "ana@example.com")
    thread = client.post("/threads", json={}, headers=headers).json()
    for question in ("First?", "Second?"):
        client.post(
            f"/threads/{thread['id']}/chat",
            json={"question": question},
            headers=headers,
        )

    rows = db_rows(
        "SELECT u.email, usage.model, usage.input_tokens, usage.output_tokens"
        " FROM usage JOIN users u ON u.id = usage.user_id"
    )
    expected = (
        "ana@example.com",
        MODEL_NAME,
        FAKE_USAGE["input_tokens"],
        FAKE_USAGE["output_tokens"],
    )
    assert rows == [expected, expected]


def test_thread_remembers_earlier_turns(client) -> None:
    headers = sign_up(client, "ana@example.com")
    thread = client.post("/threads", json={"title": "Breaches"}, headers=headers).json()
    for question in ("First?", "Second?"):
        client.post(
            f"/threads/{thread['id']}/chat",
            json={"question": question},
            headers=headers,
        )

    messages = client.get(f"/threads/{thread['id']}/messages", headers=headers).json()
    assert messages == [
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "Answer to: First?"},
        {"role": "user", "content": "Second?"},
        {"role": "assistant", "content": "Answer to: Second?"},
    ]


def test_conversation_survives_a_restart(make_client) -> None:
    before = make_client()
    headers = sign_up(before, "ana@example.com")
    thread = before.post("/threads", json={}, headers=headers).json()
    before.post(
        f"/threads/{thread['id']}/chat", json={"question": "Hi?"}, headers=headers
    )

    after = make_client()
    messages = after.get(f"/threads/{thread['id']}/messages", headers=headers).json()
    assert [m["content"] for m in messages] == ["Hi?", "Answer to: Hi?"]


def test_threads_are_listed_newest_first(client) -> None:
    headers = sign_up(client, "ana@example.com")
    for title in ("Old", "New"):
        client.post("/threads", json={"title": title}, headers=headers)
    titles = [t["title"] for t in client.get("/threads", headers=headers).json()]
    assert titles == ["New", "Old"]


def test_failed_answer_sends_an_error_and_records_no_usage(
    make_client, db_rows
) -> None:
    client = make_client(failing_graph_builder(RuntimeError("model is down")))
    headers = sign_up(client, "ana@example.com")
    thread = client.post("/threads", json={}, headers=headers).json()

    response = client.post(
        f"/threads/{thread['id']}/chat", json={"question": "Hi?"}, headers=headers
    )
    events = sse_events(response.text)
    assert events[-1]["type"] == "error"
    assert db_rows("SELECT count(*) FROM usage") == [(0,)]
