# Aegis Hybrid RAG Harness: Build Plan

Written for: the hackathon team building and judging the harness.

## Goal

Ship `aegis_hybrid_rag_harness.py`: a Python agent harness that runs hybrid retrieval (BM25 + dense vectors), reranks candidates with a cross-encoder, and governs execution with the five control-plane steps.

## Scope

In scope:
- Pydantic tool contracts for BM25, vector, hybrid, and rerank calls, plus Tier 3 order payloads.
- Hybrid index: BM25 and a hashed TF-IDF dense vector, fused with reciprocal rank fusion (RRF).
- Cross-encoder rerank. Uses `sentence-transformers` when installed; otherwise an offline lexical scorer.
- Dual memory: a RAM buffer for ReAct steps and a SQLite store for patient mandates, dosages, trial protocols, and an order audit log.
- Deterministic router: greeting, medical query, or Tier 3 order.
- HITL gate: Tier 1 read-only auto-applies; Tier 3 freezes until a reviewer approves or rejects.
- Loop circuit breaker: trips on identical low-confidence searches.
- OpenRouter for answer generation, with an extractive fallback when no key is set.
- CLI (primary) and a Streamlit UI (stretch).

Out of scope: real PHI, real clinical guidance, production auth, and a real embedding API.

## Phases

| Phase | Time | Deliverable | Done when |
|---|---|---|---|
| 1. Environment and hybrid index | 00:00-10:00 | Pydantic contracts, BM25 index, vector index, RRF, seed corpus | Clean ingestion of chunks; BM25 ranks the expected doc first |
| 2. Memory and tools | 10:00-25:00 | Tool dispatch with validation; SQLite memory | Malformed params rejected; mandates recalled after restart |
| 3. Router and HITL gate | 25:00-45:00 | Deterministic router; Tier 3 freeze and decision | Greetings make zero tool calls; rejected order writes nothing |
| 4. Circuit breaker and testing | 45:00-60:00 | Loop breaker; unit tests; CLI validation prompts | Repetitive query trips; full CLI run completes |

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Dense vectors | Hashed TF-IDF (unigrams and bigrams), cosine | Deterministic, offline, no extra API cost. Swap for an embedding model later. |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` if it loads; lexical fallback otherwise | Keeps the harness runnable without torch or a model download. The trace shows which reranker ran. |
| LLM | OpenRouter chat completions, temperature 0 | Brief requirement. Model set by `OPENROUTER_MODEL`. |
| Breaker threshold | Allow 2 identical low-confidence searches; trip on the 3rd | The brief says "> 2 times" in Step 5 and "after 2 identical calls" in the rubric. We read the rubric as "after 2 repeats". Easy to change via `BREAKER_MAX_IDENTICAL`. |
| Tier 3 execution | Freeze, then a separate decision call | Keeps the gate testable. The CLI blocks on `input()`; the Streamlit UI uses buttons. |
| Order parsing | Regex on fixed templates, validated by Pydantic | Deterministic, and every Tier 3 payload is validated before a reviewer sees it. |

## Rubric mapping

| Criterion | Points | Where it is covered | Verified by |
|---|---|---|---|
| Step 1: Pydantic RAG contracts | 20 | `RetrievalInput`, `HybridSearchInput`, `RerankInput`, `TrialUpdateInput`, `DosageUpdateInput` | `tests/test_harness.py` malformed-input tests |
| Step 2: Memory and cross-encoder | 20 | `HybridRetriever`, `Reranker`, `LongTermMemory` | Score-ordering test; mandate recall after reopening the DB |
| Step 3: Deterministic router | 20 | `route_query` | Greeting tests assert `tool_calls == 0` |
| Step 4: HITL clinical gate | 20 | `AegisHarness.handle` / `decide`, `cli_approver` | Freeze, approve, and reject tests |
| Step 5: Circuit breaker | 20 | `LoopCircuitBreaker` | Repeated nonsense query trips on the 3rd identical search |
