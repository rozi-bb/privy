"""The sidebar history layer, against a fake backend (no network)."""

import httpx
import pytest
from chainlit.types import Pagination, ThreadFilter
from chainlit.user import User

from chat_ui.data_layer import BackendDataLayer, ThreadDeletionUnsupported

pytestmark = pytest.mark.anyio

THREADS = {
    "ana-token": [
        {"id": "t2", "title": "Fines", "created_at": "2026-10-19T10:00:00+00:00"},
        {"id": "t1", "title": "Breaches", "created_at": "2026-10-18T10:00:00+00:00"},
    ],
    "bob-token": [],
}
MESSAGES = {
    "t1": [
        {"role": "user", "content": "How fast must a breach be reported?"},
        {"role": "assistant", "content": "Within 72 hours (Article 33)."},
    ]
}


def _backend(request: httpx.Request) -> httpx.Response:
    token = request.headers.get("Authorization", "").removeprefix("Bearer ")
    if token not in THREADS:
        return httpx.Response(401)
    if request.url.path == "/threads":
        return httpx.Response(200, json=THREADS[token])
    thread_id = request.url.path.split("/")[2]
    if thread_id not in {t["id"] for t in THREADS[token]}:
        return httpx.Response(404)
    return httpx.Response(200, json=MESSAGES.get(thread_id, []))


def _layer() -> BackendDataLayer:
    layer = BackendDataLayer("http://backend", transport=httpx.MockTransport(_backend))
    layer.remember_token("ana@example.com", "ana-token")
    layer.remember_token("bob@example.com", "bob-token")
    return layer


async def test_sidebar_lists_the_users_threads() -> None:
    page = await _layer().list_threads(
        Pagination(first=20), ThreadFilter(userId="ana@example.com")
    )
    assert [(t["id"], t["name"]) for t in page.data] == [
        ("t2", "Fines"),
        ("t1", "Breaches"),
    ]
    assert not page.pageInfo.hasNextPage


async def test_sidebar_pages_and_searches() -> None:
    layer = _layer()
    first = await layer.list_threads(
        Pagination(first=1), ThreadFilter(userId="ana@example.com")
    )
    assert [t["id"] for t in first.data] == ["t2"] and first.pageInfo.hasNextPage
    second = await layer.list_threads(
        Pagination(first=1, cursor="t2"), ThreadFilter(userId="ana@example.com")
    )
    assert [t["id"] for t in second.data] == ["t1"]
    found = await layer.list_threads(
        Pagination(first=20), ThreadFilter(userId="ana@example.com", search="fine")
    )
    assert [t["id"] for t in found.data] == ["t2"]


async def test_opened_thread_shows_its_messages_in_order() -> None:
    layer = _layer()
    await layer.list_threads(
        Pagination(first=20), ThreadFilter(userId="ana@example.com")
    )
    thread = await layer.get_thread("t1")
    assert thread["userIdentifier"] == "ana@example.com"
    assert [(s["type"], s["output"]) for s in thread["steps"]] == [
        ("user_message", "How fast must a breach be reported?"),
        ("assistant_message", "Within 72 hours (Article 33)."),
    ]
    assert thread["steps"][0]["createdAt"] < thread["steps"][1]["createdAt"]


async def test_threads_of_others_are_unknown() -> None:
    layer = _layer()
    await layer.list_threads(
        Pagination(first=20), ThreadFilter(userId="bob@example.com")
    )
    assert await layer.get_thread_author("t1") == ""
    assert await layer.get_thread("t1") is None


async def test_unknown_user_gets_an_empty_sidebar() -> None:
    page = await _layer().list_threads(
        Pagination(first=20), ThreadFilter(userId="eve@example.com")
    )
    assert page.data == []


async def test_token_is_recovered_from_the_signed_session_after_a_restart() -> None:
    layer = BackendDataLayer("http://backend", transport=httpx.MockTransport(_backend))
    assert await layer.get_user("ana@example.com") is None
    user = await layer.create_user(
        User(identifier="ana@example.com", metadata={"token": "ana-token"})
    )
    assert user.metadata["token"] == "ana-token"
    assert await layer.get_user("ana@example.com") is not None


async def test_deleting_a_thread_is_refused() -> None:
    with pytest.raises(ThreadDeletionUnsupported):
        await _layer().delete_thread("t1")
