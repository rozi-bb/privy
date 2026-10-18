"""Local embeddings via fastembed (no API calls)."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class LocalEmbeddings(Embeddings):
    """Embeddings computed on this machine with fastembed."""

    def __init__(self, cache_dir: str | None = None) -> None:
        """Load the embedding model, downloading it on first use."""
        from fastembed import TextEmbedding

        self._model = TextEmbedding(EMBEDDING_MODEL, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages."""
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        return [float(x) for x in next(iter(self._model.query_embed(text)))]
