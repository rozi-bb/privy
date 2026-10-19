from langchain_core.documents import Document

from agent.citations import Citation, allowed_labels, find_citations, invalid_citations


def _article(number: int, **meta) -> Document:
    return Document(
        page_content="text", metadata={"kind": "article", "number": number, **meta}
    )


def _recital(number: int) -> Document:
    return Document(page_content="text", metadata={"kind": "recital", "number": number})


def test_finds_articles_recitals_and_paragraphs() -> None:
    text = (
        "Notify within 72 hours (Article 33(1); Recital 85), see also Article 34(2)(a)."
    )
    assert find_citations(text) == {
        Citation("article", 33),
        Citation("article", 34),
        Citation("recital", 85),
    }


def test_finds_every_number_in_a_list() -> None:
    assert find_citations("Articles 15, 16 and 17 give rights.") == {
        Citation("article", 15),
        Citation("article", 16),
        Citation("article", 17),
    }


def test_keeps_the_point_only_for_the_definitions_article() -> None:
    assert find_citations("Personal data (Article 4(1)).") == {
        Citation("article", 4, 1)
    }
    assert find_citations("Within a month (Article 12(3)).") == {
        Citation("article", 12)
    }


def test_ignores_articles_of_other_legal_acts() -> None:
    text = "Article 8(1) of the Charter and Article 16(1) of the Treaty protect data."
    assert find_citations(text) == set()


def test_answer_citing_only_its_passages_is_valid() -> None:
    passages = [_article(33, part=1), _recital(85)]
    assert invalid_citations("72 hours (Article 33(1); Recital 85).", passages) == set()


def test_citation_outside_the_passages_is_invalid() -> None:
    passages = [_article(33, part=1)]
    assert invalid_citations("Fines apply (Article 83).", passages) == {
        Citation("article", 83)
    }


def test_definition_point_must_match_the_passage() -> None:
    passages = [_article(4, point=5)]
    assert invalid_citations("Pseudonymisation (Article 4(5)).", passages) == set()
    assert invalid_citations("Personal data (Article 4(1)).", passages) == {
        Citation("article", 4, 1)
    }
    assert invalid_citations("See the definitions in Article 4.", passages) == set()


def test_allowed_labels_list_definitions_by_point() -> None:
    passages = [_article(4, point=1), _article(17), _recital(26)]
    assert allowed_labels(passages) == ["Article 17", "Article 4(1)", "Recital 26"]


def test_a_range_mentions_every_article_in_it() -> None:
    assert find_citations("Articles 25 to 27 and Article 30") == {
        Citation("article", n) for n in (25, 26, 27, 30)
    }
    assert find_citations("Articles 12–14") == {
        Citation("article", n) for n in (12, 13, 14)
    }


def test_articles_referred_to_in_a_passage_may_be_repeated() -> None:
    fines = Document(
        page_content="infringements of the obligations pursuant to Articles 8, 11, 25 to 39",
        metadata={"kind": "article", "number": 83, "part": 2},
    )
    answer = (
        "Up to EUR 10 million for breaches of Articles 8, 11 and 25–39 (Article 83(4))."
    )
    assert invalid_citations(answer, [fines]) == set()
    assert invalid_citations("See Article 40 (Article 83).", [fines]) == {
        Citation("article", 40)
    }


def test_retry_prompt_lists_only_passage_labels() -> None:
    fines = Document(
        page_content="Articles 8, 11, 25 to 39",
        metadata={"kind": "article", "number": 83},
    )
    assert allowed_labels([fines]) == ["Article 83"]
