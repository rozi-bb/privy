import ast
from pathlib import Path

import pytest

from chat_ui import app

pytestmark = pytest.mark.anyio


async def _lines(*lines: str):
    for line in lines:
        yield line


async def test_parse_events_reads_data_lines_only() -> None:
    lines = _lines(
        'data: {"type": "token", "content": "Article"}',
        "",
        ": keep-alive comment",
        'data: {"type": "done"}',
    )
    events = [event async for event in app.parse_events(lines)]
    assert events == [{"type": "token", "content": "Article"}, {"type": "done"}]


def test_ui_does_not_touch_the_agent_or_provider_keys() -> None:
    source = Path(app.__file__).read_text()
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "agent" not in imported
    for key in ("DEEPSEEK_API_KEY", "TYPESAFE_API_KEY", "LANGSMITH_API_KEY"):
        assert key not in source
