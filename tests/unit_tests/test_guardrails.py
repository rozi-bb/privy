"""Graph guardrails, tested with a fake chat model (no API calls)."""

import importlib

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START, StateGraph

from agent.reranking import select_relevant
from agent.state import State

# `agent.graph` the attribute is the compiled graph (exported by the package),
# which shadows the module of the same name.
privy = importlib.import_module("agent.graph")

pytestmark = pytest.mark.anyio

BREACH = Document(page_content="72 hours", metadata={"kind": "article", "number": 33})


def _fake_model(*replies: str):
    def model(temperature=None):
        return GenericFakeChatModel(
            messages=iter([AIMessage(content=r) for r in replies])
        )

    return model


def _one_node_graph(name, node):
    workflow = StateGraph(State)
    workflow.add_node(name, node)
    workflow.add_edge(START, name)
    return workflow.compile()


async def _run(graph, state) -> tuple[dict, list[dict]]:
    events = [data async for data in graph.astream(state, stream_mode="custom")]
    return await graph.ainvoke(state), events


def _recording_model(reply: str, prompts: list):
    class Recorder(GenericFakeChatModel):
        def _generate(self, messages, *args, **kwargs):
            prompts.append(messages)
            return super()._generate(messages, *args, **kwargs)

    def model(temperature=None):
        return Recorder(messages=iter([AIMessage(content=reply)]))

    return model


def test_question_is_rewritten_in_the_regulations_terms(monkeypatch) -> None:
    prompts: list = []
    monkeypatch.setattr(
        privy, "_model", _recording_model("right to erasure of personal data", prompts)
    )
    state = {"messages": [HumanMessage(content="Can I make them delete my data?")]}
    assert privy.rewrite_query(state) == {
        "search_query": "right to erasure of personal data"
    }
    # A first question has no conversation to resolve references from.
    assert "Conversation:" not in prompts[0][-1].content


def test_follow_up_is_rewritten_into_a_standalone_query(monkeypatch) -> None:
    prompts: list = []
    monkeypatch.setattr(
        privy,
        "_model",
        _recording_model(
            "personal data breach notification to supervisory authority", prompts
        ),
    )
    state = {
        "messages": [
            HumanMessage(content="How fast must a breach be reported?"),
            AIMessage(content="Within 72 hours (Article 33)."),
            HumanMessage(content="And who do I report it to?"),
        ]
    }
    assert privy.rewrite_query(state) == {
        "search_query": "personal data breach notification to supervisory authority"
    }
    assert "How fast must a breach be reported?" in prompts[0][-1].content


def test_empty_rewrite_falls_back_to_the_question(monkeypatch) -> None:
    monkeypatch.setattr(privy, "_model", _fake_model("  "))
    state = {"messages": [HumanMessage(content="What is a DPO?")]}
    assert privy.rewrite_query(state) == {"search_query": "What is a DPO?"}


def test_low_relevance_passages_are_dropped() -> None:
    a, b, c = (Document(page_content=x) for x in "abc")
    assert select_relevant(
        [(0.9, a), (0.2, b), (0.7, c)], top_n=4, min_relevance=0.5
    ) == [a, c]


def test_nothing_relevant_means_nothing_kept() -> None:
    a = Document(page_content="a")
    assert select_relevant([(0.1, a)], top_n=4, min_relevance=0.5) == []


def test_jev_down_keeps_the_fusion_order() -> None:
    a, b, c = (Document(page_content=x) for x in "abc")
    assert select_relevant(
        [(None, a), (None, b), (None, c)], top_n=2, min_relevance=0.5
    ) == [
        a,
        b,
    ]


def test_route_skips_the_model_without_relevant_passages() -> None:
    assert privy.route_after_rerank({"reranked": []}) == "not_found"
    assert privy.route_after_rerank({"reranked": [BREACH]}) == "generate"


async def test_not_found_answers_without_the_model() -> None:
    result, events = await _run(
        _one_node_graph("not_found", privy.not_found),
        {"messages": [HumanMessage(content="What is the capital of France?")]},
    )
    assert result["messages"][-1].content == privy.NOT_FOUND_ANSWER
    assert events == [{"type": "token", "content": privy.NOT_FOUND_ANSWER}]


async def test_answer_with_valid_citations_is_kept(monkeypatch) -> None:
    monkeypatch.setattr(privy, "_model", _fake_model())
    answer = AIMessage(content="Within 72 hours (Article 33).", id="answer")
    result, events = await _run(
        _one_node_graph("verify", privy.verify),
        {"messages": [HumanMessage(content="q"), answer], "reranked": [BREACH]},
    )
    assert result["messages"][-1].content == answer.content
    assert events == []


async def test_invented_citation_is_rewritten(monkeypatch) -> None:
    monkeypatch.setattr(privy, "_model", _fake_model("Within 72 hours (Article 33)."))
    state = {
        "messages": [
            HumanMessage(content="q"),
            AIMessage(content="Within 72 hours (Article 99).", id="answer"),
        ],
        "reranked": [BREACH],
    }
    result, events = await _run(_one_node_graph("verify", privy.verify), state)
    assert [m.content for m in result["messages"]] == [
        "q",
        "Within 72 hours (Article 33).",
    ]
    assert events == [{"type": "replace", "content": "Within 72 hours (Article 33)."}]


async def test_answer_that_stays_invented_is_replaced_by_a_refusal(monkeypatch) -> None:
    monkeypatch.setattr(privy, "_model", _fake_model("Still wrong (Article 99)."))
    state = {
        "messages": [
            HumanMessage(content="q"),
            AIMessage(content="Within 72 hours (Article 99).", id="answer"),
        ],
        "reranked": [BREACH],
    }
    result, events = await _run(_one_node_graph("verify", privy.verify), state)
    assert result["messages"][-1].content == privy.UNVERIFIED_ANSWER
    assert events == [{"type": "replace", "content": privy.UNVERIFIED_ANSWER}]
