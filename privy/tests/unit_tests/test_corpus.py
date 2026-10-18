from agent.corpus import chunk_gdpr

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
