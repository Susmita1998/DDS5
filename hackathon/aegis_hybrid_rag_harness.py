"""Aegis HealthTech hybrid RAG harness.

Pipeline: BM25 + dense cosine -> reciprocal rank fusion -> cross-encoder rerank.
Governed by a deterministic router, a Tier 3 HITL gate, and a loop circuit breaker.
All demo data is synthetic. This is not clinical guidance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import uuid
import zlib
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "seed_data.json"
DEFAULT_DB = HERE / "aegis_memory.db"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")
CROSS_ENCODER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

LOW_CONFIDENCE = 0.5        # top rerank score below this counts as a low-confidence search
MAX_REACT_STEPS = 5         # ceiling on search iterations per question
BREAKER_MAX_IDENTICAL = 2   # identical low-confidence searches allowed; the next one trips
RRF_K = 60
DIM = 4096                  # width of the hashed dense vector

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "should", "the", "to", "what", "when", "which", "with",
}
ABBREVIATIONS = {
    "htn": "hypertension", "mi": "myocardial infarction", "ckd": "chronic kidney disease",
    "t2dm": "type 2 diabetes", "egfr": "estimated glomerular filtration rate",
    "ae": "adverse event", "dvt": "deep vein thrombosis", "ddi": "drug interaction",
}

GREETING_RE = re.compile(
    r"^(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|ok|okay|bye|goodbye)"
    r"(\s+(there|team|aegis|doctor|dr))?[\s.!]*$",
    re.IGNORECASE,
)
ORDER_RE = re.compile(
    r"^(please\s+)?(update|change|set|increase|decrease|adjust|raise|lower|reduce|discontinue|amend)\b",
    re.IGNORECASE,
)
ORDER_TARGET_RE = re.compile(r"\b(dose|dosage|mg|trial|enrollment|status|endpoint|protocol)\b", re.IGNORECASE)
INFO_RE = re.compile(r"\b(me|what|which|show|list|explain|summari[sz]e|why|how)\b", re.IGNORECASE)
DOSAGE_ORDER_RE = re.compile(
    r"(?P<pid>P\d{3})\b.*?\b(?P<drug>[a-z][a-z-]*)\s+(?:dose\s+|dosage\s+)?(?:to|at)\s+(?P<dose>\d+(?:\.\d+)?)\s*mg\b",
    re.IGNORECASE,
)
TRIAL_ORDER_RE = re.compile(
    r"(?P<tid>AEG-TR-\d{3})\s+(?P<field>max_enrollment|status|primary_endpoint)\s+to\s+(?P<value>.+)$",
    re.IGNORECASE,
)
PATIENT_RE = re.compile(r"\bP\d{3}\b")

GREETING_REPLY = (
    "Hello. Ask a clinical research or patient-record question, "
    "or submit a Tier 3 order (dosage or trial update)."
)
SYSTEM_PROMPT = (
    "You are the Aegis clinical intelligence assistant. Answer only from the numbered sources. "
    "Cite source ids in square brackets, for example [DOC-003]. If the sources do not answer the "
    "question, say so. The data is synthetic demo data; do not present the answer as clinical advice."
)


# ---------------------------------------------------------------- Step 1: contracts

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RetrievalInput(StrictModel):
    query: str = Field(min_length=3, max_length=300)
    top_k: int = Field(default=5, ge=1, le=20)


class BM25SearchInput(RetrievalInput):
    pass


class VectorSearchInput(RetrievalInput):
    pass


class HybridSearchInput(RetrievalInput):
    candidate_pool: int = Field(default=10, ge=2, le=50)


class RerankCandidate(StrictModel):
    doc_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    text: str = Field(min_length=1)


class RerankInput(StrictModel):
    query: str = Field(min_length=3, max_length=300)
    candidates: list[RerankCandidate] = Field(min_length=1, max_length=50)
    top_n: int = Field(default=5, ge=1, le=20)


class TrialUpdateInput(StrictModel):
    trial_id: str = Field(pattern=r"^AEG-TR-\d{3}$")
    field: Literal["status", "max_enrollment", "primary_endpoint"]
    new_value: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def _check_value(self) -> TrialUpdateInput:
        if self.field == "max_enrollment" and not (self.new_value.isdigit() and 1 <= int(self.new_value) <= 10000):
            raise ValueError("max_enrollment must be an integer between 1 and 10000")
        if self.field == "status" and self.new_value not in {"RECRUITING", "ACTIVE", "PAUSED", "CLOSED"}:
            raise ValueError("status must be RECRUITING, ACTIVE, PAUSED or CLOSED")
        return self


class DosageUpdateInput(StrictModel):
    patient_id: str = Field(pattern=r"^P\d{3}$")
    drug: str = Field(pattern=r"^[a-z][a-z-]{1,40}$")
    dose_mg: float = Field(gt=0, le=5000)


TOOL_SCHEMAS: dict[str, type[StrictModel]] = {
    "bm25_search": BM25SearchInput,
    "vector_search": VectorSearchInput,
    "hybrid_search": HybridSearchInput,
    "cross_encoder_rerank": RerankInput,
}


class Chunk(BaseModel):
    doc_id: str
    source: str
    text: str


class SearchHit(BaseModel):
    doc_id: str
    source: str
    text: str
    bm25: float = 0.0
    vector: float = 0.0
    rrf: float = 0.0
    rerank: float = 0.0


class ReActStep(BaseModel):
    thought: str
    action: str
    observation: str


class PendingOrder(BaseModel):
    order_id: str
    order_type: Literal["dosage_update", "trial_update"]
    summary: str
    payload: dict


class HarnessResponse(BaseModel):
    route: str
    status: Literal["OK", "NO_EVIDENCE", "PENDING_APPROVAL", "APPLIED", "REJECTED", "CIRCUIT_OPEN", "VALIDATION_ERROR"]
    answer: str
    tool_calls: int
    hits: list[SearchHit] = []
    pending: PendingOrder | None = None
    mandates: list[str] = []
    trace: list[str] = []


# ---------------------------------------------------------------- router (Step 3)

Route = Literal["DIRECT_ANSWER", "HYBRID_RAG_SEARCH", "TIER_3_EXECUTION"]


class RouteDecision(BaseModel):
    route: Route
    reason: str


def route_query(text: str) -> RouteDecision:
    """Deterministic routing. No LLM involved."""
    cleaned = text.strip()
    if GREETING_RE.match(cleaned):
        return RouteDecision(route="DIRECT_ANSWER", reason="greeting")
    is_question = cleaned.endswith("?") or INFO_RE.search(cleaned) is not None
    if ORDER_RE.match(cleaned) and ORDER_TARGET_RE.search(cleaned) and not is_question:
        return RouteDecision(route="TIER_3_EXECUTION", reason="imperative clinical order")
    return RouteDecision(route="HYBRID_RAG_SEARCH", reason="medical query")


def parse_order(text: str) -> PendingOrder:
    """Template-based Tier 3 parser. Payloads are validated by Pydantic before a reviewer sees them."""
    trial = TRIAL_ORDER_RE.search(text)
    if trial:
        field = trial["field"].lower()
        value = trial["value"].strip().rstrip(".")
        if field == "status":
            value = value.upper()
        payload = TrialUpdateInput(trial_id=trial["tid"].upper(), field=field, new_value=value).model_dump()
        return PendingOrder(
            order_id=uuid.uuid4().hex[:8],
            order_type="trial_update",
            summary=f"Trial {payload['trial_id']}: {field} -> {value}",
            payload=payload,
        )
    dosage = DOSAGE_ORDER_RE.search(text)
    if dosage:
        payload = DosageUpdateInput(
            patient_id=dosage["pid"].upper(),
            drug=dosage["drug"].lower(),
            dose_mg=float(dosage["dose"]),
        ).model_dump()
        return PendingOrder(
            order_id=uuid.uuid4().hex[:8],
            order_type="dosage_update",
            summary=f"Patient {payload['patient_id']}: {payload['drug']} dose -> {payload['dose_mg']:g} mg",
            payload=payload,
        )
    raise ValueError(
        "unrecognised Tier 3 order. Try 'update dosage for patient P001 metformin to 500 mg' "
        "or 'update trial AEG-TR-001 max_enrollment to 120'"
    )


# ---------------------------------------------------------------- hybrid index (Step 1-2)

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def content_terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def _features(text: str) -> list[str]:
    terms = content_terms(text)
    return terms + [f"{a}_{b}" for a, b in zip(terms, terms[1:])]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf: dict[str, Counter] = {}
        self.doc_len: dict[str, int] = {}
        df: Counter = Counter()
        for c in chunks:
            tokens = tokenize(c.text)
            self.tf[c.doc_id] = Counter(tokens)
            self.doc_len[c.doc_id] = len(tokens)
            df.update(set(tokens))
        n = max(len(chunks), 1)
        self.avgdl = sum(self.doc_len.values()) / n
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, n: int) -> list[tuple[str, float]]:
        terms = set(content_terms(query))
        scores: dict[str, float] = {}
        for doc_id, tf in self.tf.items():
            score = 0.0
            for t in terms:
                if t not in tf:
                    continue
                freq = tf[t]
                norm = freq + self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl)
                score += self.idf[t] * freq * (self.k1 + 1) / norm
            if score > 0:
                scores[doc_id] = score
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]


class VectorIndex:
    """Hashed TF-IDF vectors with cosine similarity. Deterministic and offline."""

    def __init__(self, chunks: list[Chunk]):
        counts: dict[str, Counter] = {}
        df: Counter = Counter()
        for c in chunks:
            feats = Counter(_features(c.text))
            counts[c.doc_id] = feats
            df.update(feats.keys())
        n = max(len(chunks), 1)
        self.idf = {f: math.log((1 + n) / (1 + d)) + 1 for f, d in df.items()}
        self.vectors = {doc_id: self._vectorize(feats) for doc_id, feats in counts.items()}

    def _vectorize(self, feats: Counter) -> dict[int, float]:
        vec: dict[int, float] = defaultdict(float)
        for feat, tf in feats.items():
            if feat in self.idf:
                vec[zlib.crc32(feat.encode()) % DIM] += (1 + math.log(tf)) * self.idf[feat]
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    def search(self, query: str, n: int) -> list[tuple[str, float]]:
        qv = self._vectorize(Counter(_features(query)))
        scores = []
        for doc_id, dv in self.vectors.items():
            s = sum(w * dv.get(k, 0.0) for k, w in qv.items())
            if s > 0:
                scores.append((doc_id, s))
        return sorted(scores, key=lambda kv: kv[1], reverse=True)[:n]


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return dict(scores)


def lexical_score(query: str, text: str) -> float:
    """Offline fallback reranker: query-term coverage plus bigram overlap, in [0, 1]."""
    q_terms = content_terms(query)
    if not q_terms:
        return 0.0
    d_terms = content_terms(text)
    coverage = len(set(q_terms) & set(d_terms)) / len(set(q_terms))
    q_bigrams = set(zip(q_terms, q_terms[1:]))
    bigram = len(q_bigrams & set(zip(d_terms, d_terms[1:]))) / len(q_bigrams) if q_bigrams else 0.0
    return 0.8 * coverage + 0.2 * bigram


class Reranker:
    """Cross-encoder when sentence-transformers and the model load; otherwise the lexical fallback."""

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        self.model = None
        self.name = "lexical-fallback"
        self.load_error: str | None = None
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
            self.name = model_name
        except Exception as exc:  # package missing, or model download blocked
            self.load_error = str(exc)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self.model is not None:
            logits = self.model.predict([(query, t) for t in texts])
            return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]
        return [lexical_score(query, t) for t in texts]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], reranker: Reranker):
        self.chunks = {c.doc_id: c for c in chunks}
        self.bm25 = BM25Index(chunks)
        self.vec = VectorIndex(chunks)
        self.reranker = reranker

    def bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        return self.bm25.search(query, top_k)

    def vector_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        return self.vec.search(query, top_k)

    def rerank(self, query: str, candidates: list[RerankCandidate], top_n: int) -> list[tuple[str, float]]:
        scores = self.reranker.score(query, [c.text for c in candidates])
        ranked = sorted(zip((c.doc_id for c in candidates), scores), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_n]

    def hybrid_search(self, query: str, top_k: int, candidate_pool: int) -> list[SearchHit]:
        bm = self.bm25.search(query, candidate_pool)
        vc = self.vec.search(query, candidate_pool)
        fused = rrf_fuse([[d for d, _ in bm], [d for d, _ in vc]])
        pool_ids = sorted(fused, key=fused.get, reverse=True)[:candidate_pool]
        if not pool_ids:
            return []
        scores = self.reranker.score(query, [self.chunks[d].text for d in pool_ids])
        bm_scores, vc_scores = dict(bm), dict(vc)
        hits = [
            SearchHit(
                doc_id=d,
                source=self.chunks[d].source,
                text=self.chunks[d].text,
                bm25=bm_scores.get(d, 0.0),
                vector=vc_scores.get(d, 0.0),
                rrf=fused[d],
                rerank=s,
            )
            for d, s in zip(pool_ids, scores)
        ]
        hits.sort(key=lambda h: h.rerank, reverse=True)
        return hits[:top_k]


# ---------------------------------------------------------------- memory (Step 2)

SCHEMA = """
CREATE TABLE IF NOT EXISTS patient_mandates (
    patient_id TEXT NOT NULL,
    mandate TEXT NOT NULL,
    PRIMARY KEY (patient_id, mandate)
);
CREATE TABLE IF NOT EXISTS patient_dosages (
    patient_id TEXT NOT NULL,
    drug TEXT NOT NULL,
    dose_mg REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (patient_id, drug)
);
CREATE TABLE IF NOT EXISTS trial_protocols (
    trial_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    max_enrollment INTEGER NOT NULL,
    primary_endpoint TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS clinical_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    order_type TEXT NOT NULL,
    target TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL
);
"""


class LongTermMemory:
    """SQLite store for patient mandates, dosages, trial protocols, and the Tier 3 audit log."""

    def __init__(self, db_path: Path | str, seed: dict | None = None):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        if seed:
            self._seed(seed)

    def _seed(self, seed: dict) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO patient_mandates VALUES (?, ?)",
                [(m["patient_id"], m["mandate"]) for m in seed["mandates"]],
            )
            self.conn.executemany(
                "INSERT OR IGNORE INTO trial_protocols VALUES (?, ?, ?, ?, ?)",
                [(t["trial_id"], t["title"], t["status"], t["max_enrollment"], t["primary_endpoint"]) for t in seed["trials"]],
            )

    def mandates_for(self, patient_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT mandate FROM patient_mandates WHERE patient_id = ? ORDER BY mandate", (patient_id,)
        ).fetchall()
        return [r["mandate"] for r in rows]

    def dosage_for(self, patient_id: str, drug: str) -> float | None:
        row = self.conn.execute(
            "SELECT dose_mg FROM patient_dosages WHERE patient_id = ? AND drug = ?", (patient_id, drug)
        ).fetchone()
        return row["dose_mg"] if row else None

    def trial(self, trial_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM trial_protocols WHERE trial_id = ?", (trial_id,)).fetchone()
        return dict(row) if row else None

    def commit_order(self, order: PendingOrder, approved: bool) -> None:
        """Apply (if approved) and audit in one transaction. A failed apply rolls back the audit row too."""
        with self.conn:
            if approved:
                self._apply(order)
            self.conn.execute(
                "INSERT INTO clinical_orders (created_at, order_type, target, payload, status) VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    order.order_type,
                    order.summary,
                    json.dumps(order.payload),
                    "APPROVED" if approved else "REJECTED",
                ),
            )

    def _apply(self, order: PendingOrder) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if order.order_type == "dosage_update":
            d = DosageUpdateInput.model_validate(order.payload)
            self.conn.execute(
                "INSERT INTO patient_dosages (patient_id, drug, dose_mg, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(patient_id, drug) DO UPDATE SET dose_mg = excluded.dose_mg, updated_at = excluded.updated_at",
                (d.patient_id, d.drug, d.dose_mg, now),
            )
            return
        t = TrialUpdateInput.model_validate(order.payload)
        value: str | int = int(t.new_value) if t.field == "max_enrollment" else t.new_value
        # t.field is a validated Literal, so interpolating it into the column name is safe.
        cur = self.conn.execute(f"UPDATE trial_protocols SET {t.field} = ? WHERE trial_id = ?", (value, t.trial_id))
        if cur.rowcount == 0:
            raise ValueError(f"unknown trial {t.trial_id}")


class ShortTermBuffer:
    """RAM buffer of ReAct steps for the current session."""

    def __init__(self, max_steps: int = 20):
        self.steps: deque[ReActStep] = deque(maxlen=max_steps)

    def add(self, step: ReActStep) -> None:
        self.steps.append(step)

    def clear(self) -> None:
        self.steps.clear()


# ---------------------------------------------------------------- circuit breaker (Step 5)

class CircuitOpen(RuntimeError):
    pass


class LoopCircuitBreaker:
    """Session-wide. Trips when an identical low-confidence search is repeated more than BREAKER_MAX_IDENTICAL times."""

    def __init__(self, max_identical: int = BREAKER_MAX_IDENTICAL):
        self.max_identical = max_identical
        self.low_conf_counts: Counter = Counter()

    def record(self, query: str, confidence: float) -> None:
        if confidence >= LOW_CONFIDENCE:
            return
        key = " ".join(tokenize(query))
        self.low_conf_counts[key] += 1
        if self.low_conf_counts[key] > self.max_identical:
            raise CircuitOpen(
                f"'{query}' was searched {self.low_conf_counts[key]} times with low confidence "
                f"(limit {self.max_identical})."
            )

    def reset(self) -> None:
        self.low_conf_counts.clear()


def reformulations(question: str) -> list[str]:
    """Query variants tried in order: original, keywords only, abbreviations expanded. Duplicates dropped.

    Patient IDs are identifiers, not content, so they are removed before retrieval.
    """
    base = " ".join(PATIENT_RE.sub(" ", question).split())
    terms = content_terms(base)
    variants = [base, " ".join(terms), " ".join(ABBREVIATIONS.get(t, t) for t in terms)]
    unique: list[str] = []
    for v in variants:
        if len(v) >= 3 and v not in unique:
            unique.append(v)
    return unique


# ---------------------------------------------------------------- generation

class OpenRouterClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model

    @classmethod
    def from_env(cls) -> OpenRouterClient | None:
        key = os.getenv("OPENROUTER_API_KEY")
        return cls(key) if key else None

    def answer(self, question: str, hits: list[SearchHit], mandates: list[str]) -> str:
        sources = "\n".join(f"[{h.doc_id}] ({h.source}) {h.text}" for h in hits)
        mandate_block = "\n".join(f"- {m}" for m in mandates) or "none on file"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nSources:\n{sources}\n\nPatient mandates:\n{mandate_block}"},
        ]
        response = httpx.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {self.api_key}", "X-Title": "Aegis Hybrid RAG"},
            json={"model": self.model, "messages": messages, "temperature": 0},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


def _describe(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in exc.errors())
    return str(exc)


# ---------------------------------------------------------------- harness

class AegisHarness:
    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB,
        data_path: Path | str = DEFAULT_DATA,
        llm: OpenRouterClient | None = None,
        reranker: Reranker | None = None,
    ):
        seed = json.loads(Path(data_path).read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in seed["chunks"]]
        self.retriever = HybridRetriever(chunks, reranker or Reranker())
        self.memory = LongTermMemory(db_path, seed)
        self.short_term = ShortTermBuffer()
        self.breaker = LoopCircuitBreaker()
        self.llm = llm
        self.pending: dict[str, PendingOrder] = {}
        self.tool_calls = 0

    def reset(self) -> None:
        self.short_term.clear()
        self.breaker.reset()
        self.pending.clear()

    def call_tool(self, name: str, payload: dict):
        """Validate against the tool's Pydantic schema, then dispatch. Raises ValidationError on bad input."""
        self.tool_calls += 1
        args = TOOL_SCHEMAS[name].model_validate(payload)
        handlers = {
            "bm25_search": lambda a: self.retriever.bm25_search(a.query, a.top_k),
            "vector_search": lambda a: self.retriever.vector_search(a.query, a.top_k),
            "hybrid_search": lambda a: self.retriever.hybrid_search(a.query, a.top_k, a.candidate_pool),
            "cross_encoder_rerank": lambda a: self.retriever.rerank(a.query, a.candidates, a.top_n),
        }
        return handlers[name](args)

    def handle(self, text: str, approver: Callable[[PendingOrder], bool] | None = None) -> HarnessResponse:
        self.tool_calls = 0
        decision = route_query(text)
        trace = [f"route={decision.route} ({decision.reason})"]
        self.short_term.add(ReActStep(thought=f"user: {text}", action=decision.route, observation=""))
        try:
            if decision.route == "DIRECT_ANSWER":
                return self._respond(decision.route, "OK", GREETING_REPLY, trace)
            if decision.route == "TIER_3_EXECUTION":
                return self._tier3(text, trace, approver)
            return self._rag(text, trace)
        except CircuitOpen as exc:
            trace.append(f"circuit breaker OPEN: {exc}")
            return self._respond(
                decision.route, "CIRCUIT_OPEN",
                f"Search halted by the loop circuit breaker. {exc} Please rephrase the question.", trace,
            )
        except ValueError as exc:  # includes pydantic.ValidationError
            return self._respond(decision.route, "VALIDATION_ERROR", f"Rejected: {_describe(exc)}", trace)

    def decide(self, order_id: str, approved: bool) -> HarnessResponse:
        return self._decide(order_id, approved, [])

    def _respond(self, route, status, answer, trace, hits=(), pending=None, mandates=()) -> HarnessResponse:
        return HarnessResponse(
            route=route,
            status=status,
            answer=answer,
            tool_calls=self.tool_calls,
            hits=list(hits),
            pending=pending,
            mandates=list(mandates),
            trace=trace,
        )

    def _rag(self, question: str, trace: list[str]) -> HarnessResponse:
        trace.append("Tier 1 read-only retrieval: auto-applied, no approval needed")
        hits = self._react_search(question, trace)
        patient = PATIENT_RE.search(question)
        mandates = self.memory.mandates_for(patient.group(0)) if patient else []
        if patient:
            trace.append(f"long-term memory: {len(mandates)} mandate(s) for {patient.group(0)}")
        answer = self._answer(question, hits, mandates, trace)
        status = "OK" if hits else "NO_EVIDENCE"
        return self._respond("HYBRID_RAG_SEARCH", status, answer, trace, hits=hits, mandates=mandates)

    def _react_search(self, question: str, trace: list[str]) -> list[SearchHit]:
        best: list[SearchHit] = []
        # Each variant is distinct, so one turn never repeats itself. Repeats across turns are what the breaker counts.
        for step, query in enumerate(reformulations(question)[:MAX_REACT_STEPS]):
            hits = self.call_tool("hybrid_search", {"query": query, "top_k": 5})
            confidence = hits[0].rerank if hits else 0.0
            self.short_term.add(ReActStep(
                thought=f"search {step + 1}", action=f"hybrid_search({query!r})",
                observation=f"n={len(hits)} top_rerank={confidence:.3f}",
            ))
            trace.append(
                f"step {step + 1}: hybrid_search({query!r}) -> n={len(hits)}, "
                f"top_rerank={confidence:.3f} [{self.retriever.reranker.name}]"
            )
            if confidence >= LOW_CONFIDENCE:
                return hits
            self.breaker.record(query, confidence)
            if hits and (not best or hits[0].rerank > best[0].rerank):
                best = hits
        return best

    def _answer(self, question: str, hits: list[SearchHit], mandates: list[str], trace: list[str]) -> str:
        mandate_text = f"\nMandates on file: {'; '.join(mandates)}" if mandates else ""
        if not hits:
            return "No supporting evidence found in the corpus. Try clinical terms or abbreviations." + mandate_text
        if self.llm is not None:
            try:
                answer = self.llm.answer(question, hits, mandates)
                trace.append(f"generation: OpenRouter model {self.llm.model}")
                return answer
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                trace.append(f"OpenRouter failed ({exc.__class__.__name__}); using extractive answer")
        top = hits[0]
        return f"Most relevant evidence [{top.doc_id}, {top.source}]: {top.text}{mandate_text}"

    def _tier3(self, text: str, trace: list[str], approver) -> HarnessResponse:
        order = parse_order(text)
        trace.append(f"Tier 3 gate: {order.summary} -> frozen pending HITL approval, nothing written")
        self.pending[order.order_id] = order
        if approver is None:
            return self._respond(
                "TIER_3_EXECUTION", "PENDING_APPROVAL",
                f"Order {order.order_id} is frozen until a reviewer approves or rejects it.", trace, pending=order,
            )
        approved = approver(order)
        trace.append(f"reviewer decision: {'approve' if approved else 'reject'}")
        return self._decide(order.order_id, approved, trace)

    def _decide(self, order_id: str, approved: bool, trace: list[str]) -> HarnessResponse:
        order = self.pending.pop(order_id, None)
        if order is None:
            raise KeyError(f"no pending order {order_id}")
        try:
            self.memory.commit_order(order, approved)
        except ValueError as exc:
            return self._respond("TIER_3_EXECUTION", "VALIDATION_ERROR", f"Order not applied: {_describe(exc)}", trace)
        if approved:
            return self._respond("TIER_3_EXECUTION", "APPLIED", f"Order {order_id} applied and logged.", trace)
        return self._respond("TIER_3_EXECUTION", "REJECTED", f"Order {order_id} rejected. No records changed.", trace)


# ---------------------------------------------------------------- CLI

def cli_approver(order: PendingOrder) -> bool:
    print(f"\n[HITL] Tier 3 order {order.order_id}: {order.summary}")
    print("[HITL] Execution frozen until a reviewer decides.")
    return input("Approve? [y/N]: ").strip().lower() in {"y", "yes"}


def render(resp: HarnessResponse) -> str:
    lines = [f"[{resp.status}] route={resp.route} tool_calls={resp.tool_calls}", resp.answer]
    lines += [f"  - {t}" for t in resp.trace]
    lines += [f"  hit {h.doc_id} rerank={h.rerank:.3f} bm25={h.bm25:.3f} vector={h.vector:.3f}" for h in resp.hits]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aegis hybrid RAG harness")
    parser.add_argument("--prompt", help="run one prompt and exit")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite memory file")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="seed data JSON")
    args = parser.parse_args(argv)

    harness = AegisHarness(args.db, args.data, llm=OpenRouterClient.from_env())
    if args.prompt:
        print(render(harness.handle(args.prompt, approver=cli_approver)))
        return 0

    print("Aegis CLI. Ask a question, submit a Tier 3 order, /reset, or /exit.")
    while True:
        try:
            line = input("aegis> ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if line in {"/exit", "/quit"}:
            break
        if line == "/reset":
            harness.reset()
            print("session reset")
            continue
        print(render(harness.handle(line, approver=cli_approver)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
