from langchain_core.documents import Document

from agent.retrieval import KeywordRetriever, doc_key, reciprocal_rank_fusion, tokenize


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("Hello  WORLD") == ["hello", "world"]


def test_doc_key_uses_kind_number_and_part() -> None:
    doc = Document(
        page_content="text", metadata={"kind": "article", "number": 17, "part": 2}
    )
    assert doc_key(doc) == "article:17:2:0"


def test_doc_key_without_part() -> None:
    doc = Document(page_content="text", metadata={"kind": "recital", "number": 26})
    assert doc_key(doc) == "recital:26:0:0"


def test_doc_key_tells_definition_points_apart() -> None:
    first = Document(
        page_content="a", metadata={"kind": "article", "number": 4, "point": 1}
    )
    second = Document(
        page_content="b", metadata={"kind": "article", "number": 4, "point": 2}
    )
    assert doc_key(first) != doc_key(second)


def test_doc_key_uses_doc_id_when_present() -> None:
    doc = Document(page_content="text", metadata={"doc_id": "abc123"})
    assert doc_key(doc) == "doc_id:abc123"


def test_rrf_raises_docs_ranked_high_in_both_lists() -> None:
    a = Document(page_content="shared", metadata={"kind": "article", "number": 1})
    b = Document(page_content="only bm25", metadata={"kind": "article", "number": 2})
    c = Document(page_content="only dense", metadata={"kind": "article", "number": 3})

    fused = reciprocal_rank_fusion([[a, b], [a, c]])
    assert fused[0].page_content == "shared"


def test_keyword_retriever_returns_top_docs() -> None:
    docs = [
        Document(
            page_content="alpha beta gamma", metadata={"kind": "article", "number": i}
        )
        for i in range(5)
    ]
    retriever = KeywordRetriever(docs)
    results = retriever.search("gamma", 2)
    assert len(results) == 2
    assert results[0].metadata["number"] == 0
