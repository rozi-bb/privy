"""Evaluate Privy against the success criteria in prd.md §12 (roadmap step 13).

Uploads `gdpr_dataset.json` to LangSmith (dataset "privy-gdpr", re-synced
when the file changes), runs the real graph on every example, scores the
answers, and prints the PRD table. The run costs API credit: one DeepSeek
answer plus 20 Jev calls per example, an extra DeepSeek call per follow-up
or citation retry, and one DeepSeek judge call per in-scope example.

    uv run python evals/run.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from langchain_deepseek import ChatDeepSeek  # noqa: E402
from langsmith import Client, aevaluate  # noqa: E402

from agent.citations import find_citations, invalid_citations  # noqa: E402
from agent.graph import MIN_RELEVANCE, MODEL_NAME, TOP_N, graph  # noqa: E402

DATASET_NAME = "privy-gdpr"
DATASET_FILE = Path(__file__).with_name("gdpr_dataset.json")

# prd.md §12. Invented citations are pass/fail: one failure blocks a release.
TARGETS = {
    "right_article": 0.90,
    "no_invented_citations": 1.00,
    "honest_dont_know": 0.90,
    "relevance": 0.90,
}

REFUSAL_RE = re.compile(
    # Models write both ' and ’ in contractions.
    r"could not find|couldn['’]t find|cannot help|can['’]t help|only answer|outside (the )?scope|not covered",
    re.IGNORECASE,
)

JUDGE_PROMPT = """\
You grade the answers of Privy, an assistant that answers questions about the \
GDPR from the regulation text.

Conversation (the last user message is the question):
{conversation}

Reference notes (what a good answer covers):
{reference}

Privy's answer:
{answer}

Is the answer relevant: does it address the question that was asked, \
consistent with the reference notes? Missing details are fine. An answer that \
only says it could not find the information is not relevant. When the user \
asks for a judgment about their own case, the answer is relevant only if it \
explains the provisions and points to a lawyer or a data protection authority \
instead of judging.

Reply in JSON with the keys "reasoning" (one or two sentences) and \
"relevant" (true or false)."""


def _example_rows() -> list[dict[str, Any]]:
    return [
        {
            "inputs": {"messages": item["messages"]},
            "outputs": {
                "category": item["category"],
                "expected_citations": item["expected_citations"],
                "reference": item["reference"],
            },
        }
        for item in json.loads(DATASET_FILE.read_text())
    ]


def sync_dataset(client: Client) -> None:
    """Make the LangSmith dataset match the JSON file (LangSmith keeps old versions)."""
    rows = _example_rows()
    if not client.has_dataset(dataset_name=DATASET_NAME):
        client.create_dataset(
            DATASET_NAME,
            description="GDPR questions with expected articles (prd.md §15).",
        )
    else:
        existing = list(client.list_examples(dataset_name=DATASET_NAME))
        current = sorted(
            json.dumps([e.inputs, e.outputs], sort_keys=True) for e in existing
        )
        wanted = sorted(
            json.dumps([r["inputs"], r["outputs"]], sort_keys=True) for r in rows
        )
        if current == wanted:
            return
        client.delete_examples([e.id for e in existing])
    client.create_examples(dataset_name=DATASET_NAME, examples=rows)


async def target(inputs: dict) -> dict:
    """Run the graph on one example and expose what the evaluators need."""
    state = await graph.ainvoke({"messages": inputs["messages"]})
    answer = state["messages"][-1].text
    passages = state.get("reranked") or []
    return {
        "answer": answer,
        "search_query": state["search_query"],
        "citations": sorted(str(c) for c in find_citations(answer)),
        "invalid_citations": sorted(
            str(c) for c in invalid_citations(answer, passages)
        ),
        "passages": [_label(doc.metadata) for doc in passages],
    }


def _label(meta: dict) -> str:
    label = f"{meta['kind'].title()} {meta['number']}"
    return f"{label}({meta['point']})" if meta.get("point") else label


def _outputs(obj: Any) -> dict:
    return (obj.outputs if hasattr(obj, "outputs") else obj.get("outputs")) or {}


def right_article(run: Any, example: Any) -> dict:
    """Score 1 if the answer cites at least one article the question is about."""
    expected = set(_outputs(example)["expected_citations"])
    if not expected:
        return {"score": None, "comment": "no article expected"}
    # "Article 4(1)" counts as citing Article 4.
    cited = {re.sub(r"\(\d+\)$", "", c) for c in _outputs(run)["citations"]}
    return {
        "score": int(bool(cited & expected)),
        "comment": f"expected one of {sorted(expected)}; cited {sorted(cited)}",
    }


def no_invented_citations(run: Any, example: Any) -> dict:
    """Score 1 if every cited article or recital was among the passages."""
    invalid = _outputs(run)["invalid_citations"]
    return {
        "score": int(not invalid),
        "comment": f"not in passages: {invalid}" if invalid else "",
    }


def honest_dont_know(run: Any, example: Any) -> dict:
    """Score 1 if an uncovered question is declined up front.

    The refusal must be the first sentence. Related, correctly cited context
    after it is allowed (decided 2026-10-19 after the Schrems II answer: an
    honest "not in the regulation" plus what the regulation does say).
    """
    if _outputs(example)["category"] != "unsupported":
        return {"score": None, "comment": "question is covered"}
    outputs = _outputs(run)
    first_sentence = re.split(r"(?<=[.!?])\s", outputs["answer"].strip(), maxsplit=1)[0]
    declined = (
        bool(REFUSAL_RE.search(first_sentence)) and not outputs["invalid_citations"]
    )
    return {"score": int(declined), "comment": outputs["answer"][:200]}


_judge = ChatDeepSeek(model=MODEL_NAME, temperature=0).with_structured_output(
    None, method="json_mode"
)


async def relevance(run: Any, example: Any) -> dict:
    """Score 1 if an LLM judge finds the answer addresses the question."""
    reference = _outputs(example)
    if reference["category"] == "unsupported":
        return {"score": None, "comment": "question is not covered"}
    messages = (
        example.inputs["messages"]
        if hasattr(example, "inputs")
        else example["inputs"]["messages"]
    )
    conversation = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    grade = await _judge.ainvoke(
        JUDGE_PROMPT.format(
            conversation=conversation,
            reference=reference["reference"],
            answer=_outputs(run)["answer"],
        )
    )
    return {
        "score": int(bool(grade.get("relevant"))),
        "comment": grade.get("reasoning", ""),
    }


async def main() -> int:
    """Sync the dataset, run the experiment, and print the PRD table."""
    client = Client()
    sync_dataset(client)
    results = await aevaluate(
        target,
        data=DATASET_NAME,
        evaluators=[right_article, no_invented_citations, honest_dont_know, relevance],
        experiment_prefix="privy",
        metadata={"model": MODEL_NAME, "top_n": TOP_N, "min_relevance": MIN_RELEVANCE},
        max_concurrency=4,
        client=client,
    )

    scores: dict[str, list[int]] = {name: [] for name in TARGETS}
    failures: list[str] = []
    async for row in results:
        question = row["example"].inputs["messages"][-1]["content"]
        for result in row["evaluation_results"]["results"]:
            if result.score is None:
                continue
            scores[result.key].append(int(result.score))
            if not result.score:
                failures.append(f"  {result.key}: {question}\n    {result.comment}")

    print(f"\nExperiment: {results.experiment_name}\n")
    print(f"{'criterion':<24}{'score':>8}{'target':>8}  result")
    passed = True
    for name, target_score in TARGETS.items():
        values = scores[name]
        score = sum(values) / len(values) if values else 0.0
        ok = score >= target_score
        passed &= ok
        print(
            f"{name:<24}{score:>8.0%}{target_score:>8.0%}  {'pass' if ok else 'FAIL'} ({sum(values)}/{len(values)})"
        )
    if failures:
        print("\nFailures:\n" + "\n".join(failures))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
