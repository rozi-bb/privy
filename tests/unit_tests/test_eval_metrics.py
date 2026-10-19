"""The evaluation's code-based metrics, checked on hand-made runs (no API calls)."""

from evals.run import honest_dont_know, no_invented_citations, right_article


def _run(answer: str = "", citations=(), invalid=()) -> dict:
    return {
        "outputs": {
            "answer": answer,
            "citations": list(citations),
            "invalid_citations": list(invalid),
        }
    }


def _example(category: str, expected=()) -> dict:
    return {"outputs": {"category": category, "expected_citations": list(expected)}}


def test_right_article_accepts_any_expected_article() -> None:
    example = _example("in_scope", ["Article 44", "Article 46"])
    assert right_article(_run(citations=["Article 46"]), example)["score"] == 1
    assert right_article(_run(citations=["Article 83"]), example)["score"] == 0


def test_right_article_counts_a_definition_as_its_article() -> None:
    example = _example("in_scope", ["Article 4"])
    assert right_article(_run(citations=["Article 4(1)"]), example)["score"] == 1


def test_right_article_skips_unsupported_questions() -> None:
    assert right_article(_run(), _example("unsupported"))["score"] is None


def test_invented_citation_fails() -> None:
    example = _example("in_scope", ["Article 33"])
    assert no_invented_citations(_run(invalid=["Article 99"]), example)["score"] == 0
    assert no_invented_citations(_run(), example)["score"] == 1


def test_honest_dont_know_needs_the_refusal_first() -> None:
    example = _example("unsupported")
    refusal = "I could not find an answer to that in the GDPR."
    assert honest_dont_know(_run(refusal), example)["score"] == 1
    assert honest_dont_know(_run("Paris is the capital."), example)["score"] == 0
    late = "Transfers need safeguards (Article 46). I could not find the ruling."
    assert honest_dont_know(_run(late, citations=["Article 46"]), example)["score"] == 0


def test_honest_dont_know_allows_correctly_cited_context() -> None:
    example = _example("unsupported")
    answer = "I couldn't find that in the regulation. Transfers need safeguards (Article 46)."
    cited = _run(answer, citations=["Article 46"])
    invented = _run(answer, citations=["Article 46"], invalid=["Article 46"])
    assert honest_dont_know(cited, example)["score"] == 1
    assert honest_dont_know(invented, example)["score"] == 0


def test_honest_dont_know_accepts_a_curly_apostrophe() -> None:
    answer = "I couldn’t find anything about that in the GDPR."
    assert honest_dont_know(_run(answer), _example("unsupported"))["score"] == 1
