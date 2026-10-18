"""Build the GDPR index: chunk the corpus, embed it, store it in pgvector.

Run once (and after any change to chunking or embeddings):

    uv run python scripts/build_index.py
"""

from __future__ import annotations

import asyncio
import os

from langchain_postgres import PGVector

from agent.corpus import load_gdpr_chunks
from agent.embeddings import LocalEmbeddings

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://privy:privy@localhost:5433/privy"
)


async def main() -> None:
    chunks = load_gdpr_chunks()
    print(f"Chunks: {len(chunks)}")

    store = PGVector(
        embeddings=LocalEmbeddings(),
        collection_name="gdpr",
        connection=DATABASE_URL,
        use_jsonb=True,
        async_mode=True,
    )
    await store.adelete_collection()
    await store.acreate_collection()
    await store.aadd_documents(chunks)

    print("Index built.")


if __name__ == "__main__":
    asyncio.run(main())
