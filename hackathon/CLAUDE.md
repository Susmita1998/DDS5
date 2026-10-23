# CLAUDE.md: Aegis Hybrid RAG Harness

Guidance for Claude Code working in this folder. Read `PLAN.md` for scope and decisions, and `WORKFLOW.md` for the product flow.

## What this is

A hackathon build of a clinical-intelligence agent harness for the fictional Aegis HealthTech Engine. The main file is `aegis_hybrid_rag_harness.py`. It runs hybrid retrieval, reranks with a cross-encoder, and enforces the five control-plane steps.

All data in `data/seed_data.json` is synthetic. Do not add real patient data, PHI, or real clinical guidance.

## Layout

```
hackathon/
  CLAUDE.md                     this file
  PLAN.md                       plan, decisions, rubric mapping
  WORKFLOW.md                   product workflow and CLI validation prompts
  aegis_hybrid_rag_harness.py   the harness (CLI entry point)
  app.py                        Streamlit UI (uses the same harness)
  data/seed_data.json           synthetic chunks, mandates, trial protocols
  tests/test_harness.py         unittest suite (no pytest required)
  requirements.txt              runtime deps
  .env.example                  OPENROUTER_API_KEY, OPENROUTER_MODEL, RERANKER_MODEL
```

`aegis_memory.db` is created at runtime next to the harness. It is gitignored.

## Commands

```bash
cd hackathon
pip install -r requirements.txt

# one-shot prompt
python aegis_hybrid_rag_harness.py --prompt "What is the warfarin and aspirin interaction risk?"

# interactive CLI (/reset clears session state, /exit quits)
python aegis_hybrid_rag_harness.py

# tests
python -m unittest discover -s tests -v

# Streamlit UI
streamlit run app.py
```

Set `OPENROUTER_API_KEY` to get LLM answers. Without it, the harness returns an extractive answer from the top reranked passage. Everything else works offline.

## Architecture rules

- **Route first.** `route_query` decides the path. Greetings return `DIRECT_ANSWER` with zero tool calls. Keep the router rule-based. Do not put an LLM in the router.
- **Validate every tool call.** Every retrieval call goes through `AegisHarness.call_tool`, which validates the payload against `TOOL_SCHEMAS` before dispatch. Models use `extra="forbid"`.
- **Tier 1 auto-applies, Tier 3 freezes.** Read-only retrieval never prompts. Dosage and trial updates produce a `PendingOrder` and write nothing until `decide()` is called with `approved=True`.
- **Every Tier 3 decision is audited.** Both APPROVED and REJECTED rows go to `clinical_orders`, written in the same transaction as the change.
- **Trial updates use a fixed column list.** `TrialUpdateInput.field` is a `Literal`, so the f-string SQL in `LongTermMemory._apply` is safe. Do not widen it to free text.
- **The breaker counts identical low-confidence searches across turns.** Within one question, query variants are distinct, so the agent never repeats itself. Allow `BREAKER_MAX_IDENTICAL` repeats of the same query in a session (default 2). The next one raises `CircuitOpen`. Confidence means top rerank score, and `LOW_CONFIDENCE` is 0.5. Do not move the count back inside a turn: that trips legitimate partial-evidence questions.
- **Patient IDs are stripped from retrieval queries.** `reformulations` removes them before search. Mandates are looked up separately.
- **Reranker fallback is explicit.** If `sentence-transformers` or the model is unavailable, `Reranker.name` is `lexical-fallback`. The trace prints it. Never present fallback scores as cross-encoder output.

## Conventions

- Python 3.11+. Type hints throughout. Pydantic v2 APIs (`model_validate`, `model_dump`, `field_validator`, `model_validator`).
- Match the existing file style: short docstrings, comments only where the logic is not obvious.
- Keep the harness to one module. Tests use `unittest` so they run without extra installs.
- Do not commit `aegis_memory.db` or `.env`.

## Testing expectations

Before saying a change works, run `python -m unittest discover -s tests -v` and report the actual result. Tests that cover the rubric:

- malformed query and rerank payloads are rejected (Step 1)
- BM25 and hybrid scores are in descending order (Step 2)
- mandates survive reopening the database (Step 2)
- greetings make zero tool calls (Step 3)
- Tier 3 freezes, rejects, and approves correctly (Step 4)
- a repeated nonsense query trips the breaker on the third identical search (Step 5)

## Known limits

- The dense index is a hashed TF-IDF vector, not a neural embedding. It is deterministic and offline.
- The default reranker is a lexical fallback unless `sentence-transformers` and the cross-encoder model are installed.
- Order parsing is template-based. Free-text orders are routed to the gate but rejected with a template hint if unrecognised.
