"""Streamlit frontend for RAGLens.

This provides a chat-style interface for interacting with the RAG backend.
"""

from __future__ import annotations

import os
import re
import sys

# ── Path setup ──────────────────────────────────────────────────────────
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

BACKEND_URL = os.environ.get("RAGLENS_BACKEND_URL", "http://localhost:8000")


def sanitize_math_syntax(text: str) -> str:
    """Sanitize raw LaTeX bracket delimiters into standard Markdown math syntax.

    - Converts display math \\[ ... \\] to $$ ... $$
    - Converts inline math \\( ... \\) to $ ... $
    """
    if not text:
        return text

    # Convert display brackets \[ ... \] to $$ ... $$
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    # Convert inline brackets \( ... \) to $ ... $
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)

    return text


def render_response(response: dict) -> None:
    """Render answer text, sources, evidence, and retrieval details."""
    answer_text = sanitize_math_syntax(response.get("answer", ""))
    st.markdown(answer_text)

    # Sources
    if response.get("sources"):
        with st.expander("📄 Sources", expanded=False):
            for src in response["sources"]:
                st.markdown(
                    f"[{src['citation_id']}] **{src.get('title', 'Untitled')}** "
                    f"— {src.get('section', '—')} — Page {src.get('page', '?')}"
                )
                if src.get("evidence"):
                    st.markdown(f"> {src['evidence'][:300]}...")

    # Evidence
    if response.get("evidence"):
        with st.expander("🔬 Retrieved Evidence", expanded=False):
            for ev in response["evidence"]:
                st.markdown(f"---\n**{ev.get('paper_id', '')}**")
                st.markdown(sanitize_math_syntax(ev.get("text", "")))
                st.caption(f"Score: {ev.get('score', 0):.4f}")

    # Retrieval details
    if response.get("retrieval"):
        with st.expander("🔍 Retrieval Details", expanded=False):
            st.json(response["retrieval"])


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="RAGLens",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Header ──────────────────────────────────────────────────────────
    st.title("RAGLens")
    st.caption("Research Paper Intelligence — Citation-aware RAG over research papers")

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        mode = st.radio(
            "Search Mode",
            ["Q&A", "Paper Summary", "Comparison", "Literature Review", "Evidence Search"],
            index=0,
        )
        st.divider()
        st.markdown("### Retrieval Settings")
        st.slider("Dense top-k", 1, 50, 20, key="dense_top_k")
        st.slider("Sparse top-k (BM25)", 1, 50, 20, key="sparse_top_k")
        st.slider("Hybrid top-k", 1, 50, 20, key="hybrid_top_k")
        st.slider("Rerank top-k", 1, 20, 5, key="rerank_top_k")
        st.divider()
        st.markdown("### Filters")
        st.text_input("Year range (e.g. 2020-2025)")
        st.text_input("Source (arxiv, acl, etc.)")
        st.divider()
        if st.button("Refresh Health", use_container_width=True):
            check_health()

    # ── Main Chat Interface ──────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                if isinstance(msg["content"], dict):
                    render_response(msg["content"])
                else:
                    st.markdown(sanitize_math_syntax(msg["content"]))

    # Chat input
    prompt = st.chat_input("Ask a question about the research papers...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = query_backend(prompt, mode)
            render_response(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Health status at the bottom
    check_health()


def check_health() -> None:
    """Check backend health and display status."""
    import httpx

    try:
        resp = httpx.get(f"{BACKEND_URL}/health/", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status", "unknown")
            st.sidebar.success(f"Backend: {status}")
        else:
            st.sidebar.error("Backend: error")
    except Exception:
        st.sidebar.error("Backend: disconnected")


def query_backend(query: str, mode: str) -> dict:
    """Send a query to the FastAPI backend and return the response."""
    import httpx

    endpoint_map = {
        "Q&A": "/query",
        "Paper Summary": "/summarize",
        "Comparison": "/compare",
        "Literature Review": "/literature-review",
        "Evidence Search": "/search",
    }
    endpoint = endpoint_map.get(mode, "/query")

    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{BACKEND_URL}{endpoint}",
                json={"query": query, "mode": mode},
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                if "answer" in data:
                    data["answer"] = sanitize_math_syntax(data["answer"])
                return data
            return {
                "answer": f"Error: Backend returned {resp.status_code}",
                "sources": [],
                "evidence": [],
                "retrieval": {},
            }
    except Exception as e:
        return {
            "answer": f"Error connecting to backend: {e}",
            "sources": [],
            "evidence": [],
            "retrieval": {},
        }


if __name__ == "__main__":
    main()