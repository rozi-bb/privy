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
| 9 | Conversation memory | #8 | The assistant answers from earlier turns (follow-up questions are rewritten before retrieval), and still does after the server restarts (Postgres checkpointer in the backend) | ✅ |
| 10 | Retrieval practice on FiQA | — | BM25 + dense + RRF + reranker pipeline runs on the FiQA benchmark; NDCG@10 is measured for each stage | ✅ |
| 11 | Hybrid retrieval on the GDPR corpus | #7, #9 | Index built from `corpus/` into pgvector + BM25; questions are answered from retrieved passages with correct article citations | ✅ |
| 12 | Guardrails and grounding | #10, #11 | "I don't know" when unsupported; no invented citations; out-of-scope questions declined | ✅ |

Steps 9 and 12 were marked done before any evaluation; the first evaluations showed follow-ups failing and guardrails existing only as prompt text. Both were rebuilt and pass in step 13.

FiQA result (200 queries, NDCG@10): `bm25 0.136 · dense 0.363 · hybrid 0.287 · hybrid+rerank 0.412`. The reranker is the biggest single win; naive 50:50 RRF pulls the fused ranking below dense on this paraphrase-heavy corpus, so hybrid's value is corpus-dependent.

### Release

| # | Step | Status |
|---|---|---|
| 13 | Full evaluation in LangSmith (groundedness, correctness, relevance) against prd.md #12 targets | ✅ |
| 14 | Update README with a short demo; tag `v0.1.0` | ✅ |

Evaluation runs (dataset `privy-gdpr`, 23 examples):

| Experiment | right article | no invented citations | honest "I don't know" | relevance | Notes |
|---|---|---|---|---|---|
| `privy-e9d6c417` | 89% | 100% | 100% | 95% | Everyday words ("sue", "legal grounds") did not retrieve the right articles |
| `privy-0f87fd71` | 84% | 100% | 75% | 84% | Query rewrite fixed retrieval, but the citation check rejected articles the passages themselves refer to (fixed since) |
| `privy-5ea45026` | 100% | 100% | 100%* | 100% | All targets met |

\* LangSmith shows 75%: the metric's refusal pattern missed the curly apostrophe in "I couldn’t find". Rescored from the stored outputs after the fix: 4/4. Before this run, `honest_dont_know` was also changed (with the owner's approval) to accept correctly cited context after an up-front refusal.

## Tahap 2 — Versi tipis, jalan ujung ke ujung

| # | Step | Status |
|---|---|---|
| 15 | Backend skeleton: FastAPI with a health check, calls the agent, reads config from the environment | ✅ |
| 16 | Postgres in Docker: app tables (accounts, usage) plus pgvector chunks | ✅ |
| 17 | Login with accounts; users open only their own threads | ✅ |
| 18 | Chat endpoint: check sign-in, run the agent, stream the answer, record usage | ✅ |
| 19 | Chat UI with Chainlit, as a client of the backend | ✅ |
| 20 | Chat history in the UI: the sidebar lists the user's earlier threads; opening one shows its messages and continues the conversation (backend: `GET /threads`, `GET /threads/{id}/messages`) | ✅ |
| 21 | Business-flow tests: sign-in, wrong password, usage recorded | ✅ |
| 22 | Security tests: thread isolation, no endpoint without sign-in, API key never reaches the browser | ✅ |

Step 18 was checked end to end through the chat UI with the real DeepSeek model: the answer streamed and its token counts were recorded.

## Tahap 3 — Kembangkan bergantian

| # | Step | Status |
|---|---|---|
| 23 | Tune retrieval: compare `TOP_K`, `TOP_N`, and `MIN_RELEVANCE` combinations with the step 13 evaluation and keep the cheapest one that still meets the prd.md §12 targets (see architecture.md §10) | ⬜ |
| 24–26 | Quality monitoring, cost per user, error monitoring | ⬜ |
| 27 | Small-talk routing: Jev decides whether a message is small talk, which gets a short reply without retrieval (`classify_intent`, `small_talk`; see architecture.md §2). Also moves the blocking loads in `retrieve` to a thread, which `langgraph dev` rejected | ✅ |
| 28 | Evaluate small-talk routing: add greetings, thanks, and greetings mixed with a real question to the step 13 dataset, then tune `MIN_SMALL_TALK` (see architecture.md §10) | ⬜ |

Step 27 is covered by unit tests with a fake Jev and model; Jev has not yet been run on real messages, which is what step 28 checks.

Privy is a personal project that runs on the local machine (Docker Postgres, backend, chat UI). It is not deployed to a server and has no paying users, so there are no steps for a VPS, domain, HTTPS, backups, a release procedure, packages and usage limits, or payments.
