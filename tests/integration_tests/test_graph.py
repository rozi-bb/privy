import pytest

from agent import graph

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
async def test_agent_simple_passthrough() -> None:
    inputs = {"messages": [{"role": "user", "content": "Hi"}]}
    res = await graph.ainvoke(inputs)
    assert res is not None
    assert len(res["messages"]) == 2
