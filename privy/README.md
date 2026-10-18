# Privy

An assistant that answers questions about the GDPR (EU General Data Protection Regulation) from the official regulation text, with citations. Built with LangGraph and hybrid retrieval (BM25 + dense embeddings, fused with Reciprocal Rank Fusion, reranked by Jev).

- What it does: `docs/prd.md`
- How it is built: `docs/architecture.md`
- Order of work: `docs/roadmap.md`

## Stack

LangGraph (Python 3.13, uv) · DeepSeek · local `bge-small-en-v1.5` embeddings · Postgres + pgvector · Jev (TypeSafe) reranking · LangSmith for tracing and evaluation.

## Run

```bash
cp .env.example .env   # then fill in the keys
uv sync
uv run langgraph dev   # opens LangGraph Studio
```

## Test

```bash
uv run pytest -q
```
