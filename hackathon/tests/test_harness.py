"""Rubric-aligned tests. Run: python -m unittest discover -s tests -v"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

HACKATHON = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HACKATHON))

from aegis_hybrid_rag_harness import (  # noqa: E402
    AegisHarness,
    CircuitOpen,
    DosageUpdateInput,
    HybridSearchInput,
    LoopCircuitBreaker,
    RerankCandidate,
    RerankInput,
    Reranker,
    TrialUpdateInput,
    route_query,
)

SEED = HACKATHON / "data" / "seed_data.json"


def make_harness(db_path: Path) -> AegisHarness:
    # llm=None keeps tests offline and deterministic; Reranker falls back if the cross-encoder is absent.
    return AegisHarness(db_path=db_path, data_path=SEED, llm=None)


class HarnessTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "memory.db"
        self.harness = make_harness(self.db)

    def tearDown(self):
        self.harness.memory.conn.close()
        self._tmp.cleanup()

    def dosage_rows(self):
        return self.harness.memory.conn.execute("SELECT * FROM patient_dosages").fetchall()

    def order_log(self):
        return self.harness.memory.conn.execute("SELECT status, target FROM clinical_orders").fetchall()


class Step1ContractTests(unittest.TestCase):
    def test_empty_or_short_query_rejected(self):
        with self.assertRaises(ValidationError):
            HybridSearchInput(query="ab")
        with self.assertRaises(ValidationError):
            HybridSearchInput(query="   ")

    def test_out_of_range_top_k_rejected(self):
        with self.assertRaises(ValidationError):
            HybridSearchInput(query="warfarin aspirin", top_k=0)
        with self.assertRaises(ValidationError):
            HybridSearchInput(query="warfarin aspirin", top_k=21)

    def test_unknown_fields_rejected(self):
        with self.assertRaises(ValidationError):
            HybridSearchInput(query="warfarin aspirin", drop_table=True)

    def test_rerank_requires_valid_candidates(self):
        with self.assertRaises(ValidationError):
            RerankInput(query="warfarin aspirin", candidates=[])
        with self.assertRaises(ValidationError):
            RerankCandidate(doc_id="bad id with spaces!", text="x")

    def test_trial_update_value_checked_per_field(self):
        with self.assertRaises(ValidationError):
            TrialUpdateInput(trial_id="AEG-TR-001", field="max_enrollment", new_value="abc")
        with self.assertRaises(ValidationError):
            TrialUpdateInput(trial_id="AEG-TR-001", field="status", new_value="DELETED")
        with self.assertRaises(ValidationError):
            TrialUpdateInput(trial_id="AEG-TR-001", field="primary_endpoint; DROP", new_value="x")

    def test_dosage_bounds(self):
        with self.assertRaises(ValidationError):
            DosageUpdateInput(patient_id="P001", drug="metformin", dose_mg=0)
        with self.assertRaises(ValidationError):
            DosageUpdateInput(patient_id="P1", drug="metformin", dose_mg=500)

    def test_call_tool_rejects_malformed_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = make_harness(Path(tmp) / "m.db")
            with self.assertRaises(ValidationError):
                h.call_tool("hybrid_search", {"query": "warfarin", "top_k": "lots"})
            with self.assertRaises(ValidationError):
                h.call_tool("bm25_search", {"query": "warfarin", "unexpected": 1})
            h.memory.conn.close()


class Step2RetrievalTests(HarnessTestCase):
    def test_bm25_ranks_warfarin_interaction_first_with_descending_scores(self):
        ranked = self.harness.call_tool("bm25_search", {"query": "warfarin aspirin bleeding", "top_k": 5})
        self.assertEqual(ranked[0][0], "DOC-003")
        scores = [s for _, s in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_hybrid_results_are_sorted_by_rerank_score(self):
        hits = self.harness.call_tool("hybrid_search", {"query": "metformin maximum dose", "top_k": 5})
        self.assertEqual(hits[0].doc_id, "DOC-002")
        scores = [h.rerank for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_cross_encoder_rerank_orders_candidates(self):
        candidates = [
            {"doc_id": "DOC-010", "text": "Deep vein thrombosis presents with leg swelling."},
            {"doc_id": "DOC-002", "text": "Metformin maximum dose is 2000 mg per day."},
        ]
        ranked = self.harness.call_tool(
            "cross_encoder_rerank", {"query": "metformin maximum dose", "candidates": candidates, "top_n": 2}
        )
        self.assertEqual(ranked[0][0], "DOC-002")

    def test_mandates_survive_reopening_the_database(self):
        reopened = make_harness(self.db)  # a fresh harness on the same file, as after a restart
        try:
            mandates = reopened.memory.mandates_for("P001")
            self.assertIn("Avoid NSAIDs: chronic kidney disease stage 3.", mandates)
        finally:
            reopened.memory.conn.close()

    def test_mandate_is_included_in_rag_response(self):
        resp = self.harness.handle("What is the metformin maximum dose for P001?")
        self.assertIn("Metformin dose must not exceed 2000 mg per day.", resp.mandates)


class Step3RouterTests(HarnessTestCase):
    def test_greetings_make_zero_tool_calls(self):
        for greeting in ["hi", "Hello there", "Good morning", "thanks!"]:
            with self.subTest(greeting=greeting):
                resp = self.harness.handle(greeting)
                self.assertEqual(resp.route, "DIRECT_ANSWER")
                self.assertEqual(resp.tool_calls, 0)
                self.assertEqual(resp.hits, [])

    def test_medical_questions_route_to_rag(self):
        for question in [
            "What is the warfarin and aspirin interaction risk?",
            "metformin maximum dose",
            "Update me on trial AEG-TR-001 status",
        ]:
            with self.subTest(question=question):
                self.assertEqual(route_query(question).route, "HYBRID_RAG_SEARCH")

    def test_clinical_orders_route_to_tier3(self):
        self.assertEqual(
            route_query("update dosage for patient P001 metformin to 500 mg").route, "TIER_3_EXECUTION"
        )

    def test_rag_question_uses_retrieval_tools(self):
        resp = self.harness.handle("What is the warfarin and aspirin interaction risk?")
        self.assertEqual(resp.status, "OK")
        self.assertGreaterEqual(resp.tool_calls, 1)
        self.assertEqual(resp.hits[0].doc_id, "DOC-003")


class Step4HITLTests(HarnessTestCase):
    ORDER = "update dosage for patient P001 metformin to 500 mg"

    def test_tier3_freezes_without_writing(self):
        resp = self.harness.handle(self.ORDER)
        self.assertEqual(resp.status, "PENDING_APPROVAL")
        self.assertIsNotNone(resp.pending)
        self.assertEqual(self.dosage_rows(), [])
        self.assertEqual(self.order_log(), [])

    def test_rejection_writes_nothing_and_is_audited(self):
        resp = self.harness.handle(self.ORDER, approver=lambda order: False)
        self.assertEqual(resp.status, "REJECTED")
        self.assertEqual(self.dosage_rows(), [])
        self.assertEqual([r["status"] for r in self.order_log()], ["REJECTED"])

    def test_approval_applies_and_audits(self):
        resp = self.harness.handle(self.ORDER, approver=lambda order: True)
        self.assertEqual(resp.status, "APPLIED")
        self.assertEqual(self.harness.memory.dosage_for("P001", "metformin"), 500.0)
        self.assertEqual([r["status"] for r in self.order_log()], ["APPROVED"])

    def test_pending_order_can_be_decided_later(self):
        resp = self.harness.handle("update trial AEG-TR-001 max_enrollment to 150")
        self.assertEqual(resp.status, "PENDING_APPROVAL")
        self.assertEqual(self.harness.memory.trial("AEG-TR-001")["max_enrollment"], 200)
        decided = self.harness.decide(resp.pending.order_id, approved=True)
        self.assertEqual(decided.status, "APPLIED")
        self.assertEqual(self.harness.memory.trial("AEG-TR-001")["max_enrollment"], 150)

    def test_tier1_retrieval_never_prompts(self):
        def must_not_be_called(order):
            raise AssertionError("Tier 1 retrieval must not request approval")

        resp = self.harness.handle("What is the warfarin and aspirin interaction risk?", approver=must_not_be_called)
        self.assertEqual(resp.status, "OK")

    def test_unrecognised_or_invalid_orders_are_rejected(self):
        bad_orders = [
            "update dosage for patient P1 metformin to 500 mg",
            "update dosage for patient P001 metformin to 0 mg",
            "update trial AEG-TR-001 max_enrollment to lots",
        ]
        for text in bad_orders:
            with self.subTest(text=text):
                resp = self.harness.handle(text, approver=lambda order: True)
                self.assertEqual(resp.status, "VALIDATION_ERROR")
        self.assertEqual(self.dosage_rows(), [])


class Step5CircuitBreakerTests(HarnessTestCase):
    def test_breaker_allows_two_identical_low_confidence_searches(self):
        breaker = LoopCircuitBreaker()
        breaker.record("quantum blockchain", 0.0)
        breaker.record("quantum blockchain", 0.0)
        with self.assertRaises(CircuitOpen):
            breaker.record("quantum blockchain", 0.0)

    def test_breaker_ignores_confident_searches(self):
        breaker = LoopCircuitBreaker()
        for _ in range(10):
            breaker.record("warfarin aspirin", 0.9)

    NONSENSE = "quantum blockchain xylophone"

    def test_repetitive_nonsense_query_trips_on_third_identical_search(self):
        self.assertEqual(self.harness.handle(self.NONSENSE).status, "NO_EVIDENCE")
        self.assertEqual(self.harness.handle(self.NONSENSE).status, "NO_EVIDENCE")
        resp = self.harness.handle(self.NONSENSE)
        self.assertEqual(resp.status, "CIRCUIT_OPEN")
        self.assertEqual(resp.tool_calls, 1)
        self.assertIn("circuit breaker OPEN", "\n".join(resp.trace))

    def test_breaker_only_counts_identical_queries(self):
        for _ in range(3):
            self.harness.handle(self.NONSENSE)
        self.assertEqual(self.harness.handle("hypertension first line treatment").status, "OK")

    def test_legitimate_low_confidence_question_is_not_tripped_within_a_turn(self):
        resp = self.harness.handle("What is the metformin maximum dose for P001?")
        self.assertNotEqual(resp.status, "CIRCUIT_OPEN")

    def test_reset_clears_breaker_counts(self):
        for _ in range(3):
            self.harness.handle(self.NONSENSE)
        self.harness.reset()
        self.assertEqual(self.harness.handle(self.NONSENSE).status, "NO_EVIDENCE")


class RerankerFallbackTests(unittest.TestCase):
    def test_fallback_is_labelled_when_cross_encoder_missing(self):
        reranker = Reranker()
        if reranker.model is None:
            self.assertEqual(reranker.name, "lexical-fallback")
            self.assertIsNotNone(reranker.load_error)


if __name__ == "__main__":
    unittest.main()
