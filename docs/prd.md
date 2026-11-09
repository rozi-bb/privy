# Privy — Product Requirements

| | |
|---|---|
| Status | Draft |
| Last updated | 2026-10-19 |

## 1. Summary

Privy answers questions about the GDPR (EU Regulation 2016/679) from the official regulation text, with citations to the articles it relies on.

## 2. Background

The GDPR is dense and scattered across 99 articles and 173 recitals. Individuals want to know their rights; small companies want to know their obligations. Search engines return opinions and summaries of unclear quality. Privy answers from the regulation itself.

## 3. Target users

- Individuals asking about their data rights ("Can I ask a company to delete my data?").
- Small companies and developers asking about obligations ("Do I need to appoint a DPO?").
- Not a substitute for a lawyer; Privy answers from the regulation, not legal advice.

## 4. Scope

- Answer questions about the GDPR from the official text (articles and recitals).
- Cite the articles the answer is grounded in.
- Admit when the regulation does not cover the question.
- Remember earlier messages in a conversation.

## 5. Out of scope

- Legal advice and case-specific judgments.
- Other laws (ePrivacy, national implementation laws) — out of scope in release 1.
- Questions unrelated to privacy or data protection are declined politely.
- Writing documents, generating policies, or acting on behalf of the user.

## 6. Persona and tone

Privy is a precise, plain-language assistant. Short answers first, then the article numbers. Never sounds like a lawyer and never gives legal advice. This section becomes the system prompt.

## 7. Knowledge source

- The official GDPR text: `corpus/gdpr.pdf` (EUR-Lex, Regulation (EU) 2016/679, OJ L 119, 4.5.2016).
- Retrieval is hybrid: BM25 (keyword) + dense embeddings (semantic), fused with Reciprocal Rank Fusion, reranked by Jev.

> **FiQA is not a knowledge source.** The FiQA dataset (`practice/fiqa/`, a financial question-answering benchmark) is used only once as labeled practice data (roadmap step 10) to measure retrieval quality. It is never indexed, never retrieved, and never reaches an answer. See `architecture.md` §9.

## 8. Memory policy

- Conversation memory per thread, through LangGraph's checkpointer and a `thread_id`.
- Memory must survive a restart of the server.

## 9. Tools and actions

- Retrieval only, in release 1. No write actions.

## 10. Guardrails

- No legal advice. If asked for a judgment ("can I sue?"), Privy explains the relevant article and suggests consulting a lawyer.
- No fabricated answers. If no retrieved passage supports an answer, Privy says it does not know.
- No fabricated citations. Every article number mentioned must appear in a retrieved passage.
- Out-of-scope questions are declined with one polite sentence.
- Small talk (greetings, thanks, "what can you do?") gets a short, friendly reply without citations. Anything that might be a data-protection question goes through retrieval, even when it starts with a greeting.

## 11. Fallback and escalation

- When no passage is relevant enough, Privy says it could not find an answer in the regulation and offers to rephrase.
- When the question needs a lawyer or a data protection authority, Privy says so and points to the user's national authority.

## 12. Success criteria

Measured in LangSmith (see roadmap step 13):

| Category | Target |
|---|---|
| Groundedness — right article cited | 90% |
| Groundedness — no invented citations | 100% |
| Honest "I don't know" when unsupported | 90% |
| Relevance — answers the question asked | 90% |

Invented citations are pass/fail: a single failure blocks a release.

## 13. Constraints

- Corpus: one document, the official GDPR text (~88 pages).
- Latency target: answer within ~10 seconds on the local machine.
- Budget: DeepSeek API costs must stay trivial for development (see LangSmith traces).

## 14. Assumptions and risks

- The GDPR text does not change; if it does, the corpus and index are rebuilt.
- The model may know the GDPR from training. The prompt must force grounding in retrieved passages, and evaluation must catch off-corpus answers.

## 15. Initial evaluation dataset

5–10 question–answer examples stored in LangSmith before the first feature is evaluated, e.g.:

1. "What is the right to erasure?" → Article 17.
2. "How much can a company be fined?" → Article 83 (up to €20 million or 4%).
3. "How fast must a breach be reported?" → 72 hours (Article 33).
4. "Can a company keep my data forever?" → storage limitation, Article 5.
5. "What is a DPO?" → Article 37.
6. "Do children get special protection?" → Article 8.
7. "Can I get a copy of my data?" → Article 15.
8. "What is personal data?" → Article 4(1).

## Product side

- **Login** — accounts, so conversations are private per user (roadmap step 17).
- No packages, usage limits, or payments: Privy is a personal project that runs locally, not a public service.
