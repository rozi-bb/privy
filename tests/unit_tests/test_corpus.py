from collections import Counter

from agent.corpus import chunk_gdpr, load_gdpr_chunks

SAMPLE = """THE EUROPEAN PARLIAMENT AND THE COUNCIL OF THE EUROPEAN UNION,

(1)

The protection of natural persons in relation to the processing of personal data is a fundamental right.

(2)

This Regulation respects all fundamental rights.

HAVE ADOPTED THIS REGULATION:

Article 1

Subject matter

1. This Regulation lays down rules relating to the protection of natural persons.

Article 2

Material scope

1. This Regulation applies to the processing of personal data.

This Regulation shall be binding in its entirety and directly applicable in all Member States.

Done at Brussels, 27 April 2016.

(1) OJ C 229, 31.7.2012, p. 90.
"""


def test_chunk_gdpr_splits_recitals_and_articles() -> None:
    chunks = chunk_gdpr(SAMPLE)
    kinds = {(c.metadata["kind"], c.metadata["number"]) for c in chunks}
    assert ("recital", 1) in kinds
    assert ("recital", 2) in kinds
    assert ("article", 1) in kinds
    assert ("article", 2) in kinds


def test_chunk_gdpr_skips_footnotes_after_the_binding_clause() -> None:
    chunks = chunk_gdpr(SAMPLE)
    assert all("OJ C 229" not in c.page_content for c in chunks)


def test_long_article_is_split_into_parts() -> None:
    long_body = "\n\n".join(["This is a long paragraph about personal data."] * 120)
    text = f"HAVE ADOPTED THIS REGULATION:\n\nArticle 5\n\n{long_body}"
    chunks = chunk_gdpr(text)
    parts = [c for c in chunks if c.metadata["kind"] == "article"]
    assert len(parts) > 1
    assert all("part" in c.metadata for c in parts)


DEFINITIONS_SAMPLE = """(1)

A recital.

HAVE ADOPTED THIS REGULATION:

CHAPTER I

General provisions

Article 4

Definitions

For the purposes of this Regulation:

(1)

\t

'personal data' means any information relating to a natural person;

(2)

\t

'processing' means any operation performed on personal data;

CHAPTER II

Principles

Article 5

Principles relating to processing of personal data
"""


def test_numbered_points_inside_an_article_are_not_recitals() -> None:
    chunks = chunk_gdpr(DEFINITIONS_SAMPLE)
    recitals = [c for c in chunks if c.metadata["kind"] == "recital"]
    assert [c.page_content for c in recitals] == ["A recital."]


def test_article_with_numbered_points_gets_one_chunk_per_point() -> None:
    chunks = chunk_gdpr(DEFINITIONS_SAMPLE)
    points = [c for c in chunks if c.metadata.get("number") == 4]
    assert [c.metadata["point"] for c in points] == [1, 2]
    assert points[0].page_content.startswith("'personal data' means")


def test_chapter_headings_do_not_stick_to_articles() -> None:
    chunks = chunk_gdpr(DEFINITIONS_SAMPLE)
    assert all("CHAPTER" not in c.page_content for c in chunks)
    assert all("Principles\n" not in c.page_content for c in chunks)


def test_real_corpus_has_every_unit_once() -> None:
    chunks = load_gdpr_chunks()
    keys = Counter(
        (
            c.metadata["kind"],
            c.metadata["number"],
            c.metadata.get("part"),
            c.metadata.get("point"),
        )
        for c in chunks
    )
    assert max(keys.values()) == 1
    recitals = {c.metadata["number"] for c in chunks if c.metadata["kind"] == "recital"}
    articles = {c.metadata["number"] for c in chunks if c.metadata["kind"] == "article"}
    assert recitals == set(range(1, 174))
    assert articles == set(range(1, 100))
    definitions = [c for c in chunks if c.metadata.get("point")]
    assert {c.metadata["number"] for c in definitions} == {4}
    assert len(definitions) == 26
