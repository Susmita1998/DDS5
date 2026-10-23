"""Streamlit UI for the Aegis hybrid RAG harness. Run: streamlit run app.py"""

import streamlit as st

from aegis_hybrid_rag_harness import AegisHarness, HarnessResponse, OpenRouterClient


@st.cache_resource
def get_harness() -> AegisHarness:
    return AegisHarness(llm=OpenRouterClient.from_env())


def ask(question: str) -> None:
    st.session_state.history.append((question, harness.handle(question)))


def decide(order_id: str, approved: bool) -> None:
    verdict = "approved" if approved else "rejected"
    st.session_state.history.append((f"Reviewer {verdict} order {order_id}", harness.decide(order_id, approved)))


def render(resp: HarnessResponse) -> None:
    st.markdown(f"**Status:** `{resp.status}` · route `{resp.route}` · tool calls {resp.tool_calls}")
    st.write(resp.answer)
    if resp.pending and resp.pending.order_id in harness.pending:
        st.warning("Tier 3 order frozen. Nothing is written until a reviewer decides.")
        left, right = st.columns(2)
        left.button("Approve", key=f"approve-{resp.pending.order_id}", type="primary",
                    on_click=decide, args=(resp.pending.order_id, True))
        right.button("Reject", key=f"reject-{resp.pending.order_id}",
                     on_click=decide, args=(resp.pending.order_id, False))
    if resp.hits:
        st.dataframe(
            [{"doc": h.doc_id, "source": h.source, "rerank": round(h.rerank, 3),
              "bm25": round(h.bm25, 3), "vector": round(h.vector, 3)} for h in resp.hits],
            hide_index=True,
        )
    with st.expander("Trace"):
        st.code("\n".join(resp.trace) or "(no trace)")


harness = get_harness()
if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="Aegis Clinical Intelligence", layout="wide")
st.title("Aegis Clinical Intelligence")
st.caption("Synthetic demo data. Hybrid BM25 + dense retrieval, cross-encoder rerank, HITL gate for Tier 3 orders.")

with st.sidebar:
    st.write(f"Reranker: `{harness.retriever.reranker.name}`")
    st.write("LLM: " + ("OpenRouter" if harness.llm else "extractive (no OPENROUTER_API_KEY)"))
    if st.button("Reset session"):
        harness.reset()
        st.session_state.history = []

prompt = st.chat_input("Ask a question, or submit e.g. 'update dosage for patient P001 metformin to 500 mg'")
if prompt:
    ask(prompt)

for label, resp in st.session_state.history:
    if label:
        st.chat_message("user").write(label)
    with st.chat_message("assistant"):
        render(resp)
