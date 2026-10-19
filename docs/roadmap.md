# Privy — Roadmap

| | |
|---|---|
| Last updated | 2026-10-19 |

The order of work. What each feature must do is in [prd.md](prd.md); how it is built is in [architecture.md](architecture.md). Steps are numbered in the order they are done.

Legend: ✅ done · 🔄 in progress · ⬜ not started

## Tahap 1 — Buktikan agent berguna

### Setup (once)

| # | Step | Status |
|---|---|---|
| 1 | Scaffold from the LangGraph template | ✅ |
| 2 | Git, `.gitignore`, `.env` | ✅ |
| 3 | `AGENTS.md` | ✅ |
| 4 | PRD | ✅ |
| 5 | Architecture | ✅ |
| 6 | Roadmap | ✅ |
| 7 | Turn on tracing and confirm traces reach LangSmith | ✅ |
| 8 | Run the baseline graph in Studio | ✅ |

### Features (one at a time, each evaluated before the next)

Each feature is built, then evaluated, before the next one starts. If the evaluation fails, fix before moving on.

| # | Feature | PRD | Done when | Status |
|---|---|---|---|---|
| 9 | Conversation memory | #8 | The assistant answers from earlier turns, and still does after the server restarts (Postgres checkpointer in the backend) | ✅ |
| 10 | Retrieval practice on FiQA | — | BM25 + dense + RRF + reranker pipeline runs on the FiQA benchmark; NDCG@10 is measured for each stage | ✅ |
| 11 | Hybrid retrieval on the GDPR corpus | #7, #9 | Index built from `corpus/` into pgvector + BM25; questions are answered from retrieved passages with correct article citations | ✅ |
| 12 | Guardrails and grounding | #10, #11 | "I don't know" when unsupported; no invented citations; out-of-scope questions declined | ✅ |

FiQA result (200 queries, NDCG@10): `bm25 0.136 · dense 0.363 · hybrid 0.287 · hybrid+rerank 0.412`. The reranker is the biggest single win; naive 50:50 RRF pulls the fused ranking below dense on this paraphrase-heavy corpus, so hybrid's value is corpus-dependent.

### Release

| # | Step | Status |
|---|---|---|
| 13 | Full evaluation in LangSmith (groundedness, correctness, relevance) against prd.md #12 targets | ⬜ |
| 14 | Update README with a short demo; tag `v0.1.0` | ⬜ |

## Tahap 2 — Versi tipis, jalan ujung ke ujung

| # | Step | Status |
|---|---|---|
| 15 | Backend skeleton: FastAPI with a health check, calls the agent, reads config from the environment | ✅ |
| 16 | Postgres in Docker: app tables (accounts, usage) plus pgvector chunks | ✅ |
| 17 | Login with accounts; users open only their own threads | ✅ |
| 18 | Chat endpoint: check sign-in, run the agent, stream the answer, record usage | ✅ |
| 19 | Chat UI with Chainlit, as a client of the backend | ✅ |
| 20 | Business-flow tests: sign-in, wrong password, usage recorded | ✅ |
| 21 | Security tests: thread isolation, no endpoint without sign-in, API key never reaches the browser | ✅ |

Step 18 was checked end to end through the chat UI with the real DeepSeek model: the answer streamed and its token counts were recorded. The chat UI does not yet list earlier threads in its sidebar; the backend already stores and lists them (`GET /threads`).

## Tahap 3 — Kembangkan bergantian

| # | Step | Status |
|---|---|---|
| 22 | Packages and usage limits | ⬜ |
| 23 | Payments | ⬜ |
| 24–27 | VPS, domain, HTTPS, backups, release procedure | ⬜ |
| 28–30 | Quality monitoring, cost per user, error monitoring | ⬜ |
