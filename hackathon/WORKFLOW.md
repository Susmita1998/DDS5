# Aegis Hybrid RAG: Product Workflow

Written for: hackathon participants and judges running the demo.

## 1. Request lifecycle

```
user input
   │
   ▼
[Step 3] route_query (deterministic)
   ├── greeting ───────────────► DIRECT_ANSWER        0 tool calls, no retrieval
   ├── medical question ───────► HYBRID_RAG_SEARCH    Tier 1, auto-applied
   └── imperative order ───────► TIER_3_EXECUTION     Tier 3, frozen for HITL
```

### HYBRID_RAG_SEARCH (Tier 1, read-only)

1. Remove patient IDs from the query (they are identifiers, not content).
2. Build up to three distinct query variants: original, keywords only, abbreviations expanded.
3. For each variant (Step 2 retrieval):
   - BM25 top-20
   - hashed TF-IDF cosine top-20
   - reciprocal rank fusion (k=60), candidate pool of 10
   - cross-encoder rerank of the pool (or the lexical fallback)
4. Stop at the first variant whose top rerank score is at least 0.5.
5. If none reach 0.5, record each low-confidence search with the Step 5 breaker and keep the best hits.
6. Attach mandates from SQLite (Step 2) when the question names a patient.
7. Generate an answer through OpenRouter with the top 5 hits as numbered sources. Without a key, return the top passage.

### TIER_3_EXECUTION (dosage or trial change, HITL)

1. `parse_order` matches one of two templates and validates the payload with Pydantic (Step 1).
2. The order is frozen. Nothing is written. Status is `PENDING_APPROVAL`.
3. A reviewer approves or rejects:
   - CLI: an interactive prompt blocks the session until `y` or `n`.
   - Streamlit: Approve and Reject buttons.
4. Approved: the change and the audit row are written in one transaction. Status is `APPLIED`.
5. Rejected: an audit row is written. Status is `REJECTED`. The record is unchanged.

## 2. Control-plane map

| Step | Component | Enforced by |
|---|---|---|
| 1. Type-safe tool interfaces | `*SearchInput`, `RerankInput`, `TrialUpdateInput`, `DosageUpdateInput` | `AegisHarness.call_tool` and `parse_order` |
| 2. Dual memory | `ShortTermBuffer` (RAM), `LongTermMemory` (SQLite) | Mandates are read on every patient-scoped question |
| 3. Deterministic router | `route_query` | Regex rules only |
| 4. HITL gate | `handle(..., approver)` and `decide()` | Writes happen only in `commit_order(approved=True)` |
| 5. Loop circuit breaker | `LoopCircuitBreaker` | Trips on the 3rd identical low-confidence search in a session |

## 3. Runbook

```bash
cd hackathon
pip install -r requirements.txt
export OPENROUTER_API_KEY=...        # optional; without it answers are extractive

python -m unittest discover -s tests -v    # expect 29 passing
python aegis_hybrid_rag_harness.py         # interactive CLI
streamlit run app.py                       # UI
```

## 4. CLI validation prompts

Run these in one interactive session, in order. Each row is one prompt.

| # | Prompt | Expected status | Expected evidence |
|---|---|---|---|
| 1 | `hi` | `OK`, route `DIRECT_ANSWER`, tool calls 0 | Greeting reply, no retrieval |
| 2 | `What is the warfarin and aspirin interaction risk?` | `OK`, tool calls 1 | Top hit `DOC-003` |
| 3 | `What is the metformin maximum dose for P001?` | `OK` | Answer cites `DOC-002`; mandate "Metformin dose must not exceed 2000 mg per day." listed |
| 4 | `update dosage for patient P001 metformin to 500 mg` | `PENDING_APPROVAL`, then answer `n` | Status `REJECTED`; `patient_dosages` unchanged |
| 5 | `update dosage for patient P001 metformin to 500 mg` | `PENDING_APPROVAL`, then answer `y` | Status `APPLIED`; `patient_dosages` has P001 / metformin / 500 |
| 6 | `update trial AEG-TR-001 max_enrollment to 150` | `PENDING_APPROVAL`, then `y` | `trial_protocols` shows 150 |
| 7 | `update dosage for patient P1 metformin to 500 mg` | `VALIDATION_ERROR` | Unrecognised order (malformed patient ID) |
| 8 | `update dosage for patient P001 metformin to 0 mg` | `VALIDATION_ERROR` | `dose_mg: Input should be greater than 0` |
| 9 | `quantum blockchain xylophone` (three times) | `NO_EVIDENCE`, `NO_EVIDENCE`, `CIRCUIT_OPEN` | Third identical low-confidence search trips the breaker |
| 10 | `/reset`, then `quantum blockchain xylophone` | `NO_EVIDENCE` | Breaker counts cleared |

Check the audit trail after prompts 4 to 6:

```bash
python -c "import sqlite3; c=sqlite3.connect('aegis_memory.db'); print(c.execute('select order_type, target, status from clinical_orders').fetchall())"
```

## 5. Known limits for the demo

- The default reranker is the lexical fallback. It scores plain term coverage, so near-synonyms such as "dose" and "dosing" do not match. Install `sentence-transformers` for the cross-encoder.
- Order parsing is template-based. Natural-language orders outside the two templates are rejected with a hint.
- The corpus is 12 synthetic chunks. Retrieval quality is only meaningful on this corpus.
- No authentication or per-user approval identity. The reviewer is whoever is at the terminal.
