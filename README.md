# Privy

An assistant that answers questions about the GDPR (EU General Data Protection Regulation) from the official regulation text, with citations. Built with LangGraph and hybrid retrieval (BM25 + dense embeddings, fused with Reciprocal Rank Fusion, reranked by Jev).

- What it does: `docs/prd.md`
- How it is built: `docs/architecture.md`
- Order of work: `docs/roadmap.md`

## Demo

One conversation in the chat UI (answers shortened):

> **You:** How fast must a data breach be reported?
>
> **Privy:** A controller must report a personal data breach to the supervisory authority without undue delay and, where feasible, within 72 hours of becoming aware of it, unless the breach is unlikely to risk people's rights and freedoms (Article 33(1); Recital 85). A processor must notify the controller without undue delay (Article 33(2)).
>
> **You:** And who do I report it to?
>
> **Privy:** Report it to the competent supervisory authority (Article 33(1)). If you are a processor, report it to the controller without undue delay (Article 33(2)).
>
> **You:** What is the capital of France?
>
> **Privy:** I could not find an answer to that in the GDPR. I answer questions about the EU General Data Protection Regulation from its official text; could you rephrase, or ask about a specific right or obligation?

How each answer is made: Jev first checks whether the message is small talk, such as a greeting; if it is sure, Privy replies briefly without searching the regulation. Otherwise the question is rewritten into the regulation's terms (so the follow-up becomes a question about breach notification), searched with BM25 and embeddings, reranked by Jev, answered by DeepSeek from the top passages only, and checked so that every cited article appears in those passages. When no passage is relevant, Privy says so without calling the model.

## Evaluation

23 questions in LangSmith (`evals/`), against the targets in `docs/prd.md` §12:

| Criterion | Target | v0.1.0 |
|---|---|---|
| Right article cited | 90% | 100% (19/19) |
| No invented citations | 100% | 100% (23/23) |
| Honest "I don't know" when unsupported | 90% | 100% (4/4) |
| Relevance | 90% | 100% (19/19) |

## Stack

LangGraph (Python 3.13, uv) · DeepSeek · local `bge-small-en-v1.5` embeddings · Postgres + pgvector · Jev (TypeSafe) reranking · FastAPI backend with accounts · Chainlit chat UI · LangSmith for tracing and evaluation.

## Run

```bash
cp .env.example .env                    # fill in the keys and secrets
uv sync
docker compose up -d                    # Postgres with pgvector
uv run python scripts/build_index.py    # index the GDPR text (local embeddings, no API calls)
uv run uvicorn backend.main:app --port 8000
uv run python scripts/create_user.py you@example.com   # in another terminal
uv run chainlit run src/chat_ui/app.py --port 8001     # open http://localhost:8001
```

For development, `uv run langgraph dev` opens the graph in LangGraph Studio.

## Test

```bash
uv run pytest -q tests/unit_tests                         # free
uv run pytest -q tests/integration_tests/test_backend_*.py   # needs Docker, no API calls
uv run python evals/run.py                                # LangSmith evaluation (costs API credit)
```
