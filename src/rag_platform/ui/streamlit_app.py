"""Streamlit chat UI — connects to FastAPI WebSocket and streams tokens."""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import streamlit as st
import websockets

WS_URL = os.getenv("WS_URL", "ws://localhost:8000/ws/chat")

st.set_page_config(
    page_title="RAG Analytics Platform",
    page_icon="📊",
    layout="wide",
)

# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- Sidebar ---
with st.sidebar:
    st.title("📊 RAG Analytics")
    st.caption("Ask questions about SEC filings")
    st.divider()
    st.write(f"**Session:** `{st.session_state.session_id[:8]}...`")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    st.divider()
    st.markdown("**Supported tickers**")
    st.write("AAPL · MSFT · GOOGL · TSLA · JPM")
    st.divider()
    st.caption("Powered by LangGraph + Qdrant + Ollama")

# --- Main area ---
st.title("SEC Filings Q&A")
st.caption(
    "Ask anything about Apple, Microsoft, Google, Tesla, or JPMorgan 10-K/10-Q filings."
)

# Replay existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("Sources", expanded=False):
                for i, cite in enumerate(msg["citations"], 1):
                    st.write(f"[{i}] `{cite}`")


# --- WebSocket streaming ---
async def _stream_tokens(query: str, session_id: str) -> tuple[str, list[str]]:
    """Connect to WebSocket, collect tokens and citations, return both."""
    tokens: list[str] = []
    citations: list[str] = []
    try:
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps({"query": query, "session_id": session_id}))
            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                if msg["type"] == "token":
                    tokens.append(msg["content"])
                elif msg["type"] == "done":
                    citations = msg.get("citations", [])
                    break
                elif msg["type"] == "error":
                    tokens.append(f"\n\n⚠️ Error: {msg['message']}")
                    break
    except Exception as exc:
        tokens.append(f"\n\n⚠️ Could not reach the API: {exc}")
    return "".join(tokens), citations


def _query(query: str, session_id: str) -> tuple[str, list[str]]:
    return asyncio.run(_stream_tokens(query, session_id))


# --- Handle new input ---
if prompt := st.chat_input("Ask about SEC filings…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, citations = _query(prompt, st.session_state.session_id)
        st.markdown(answer)
        if citations:
            with st.expander("Sources", expanded=False):
                for i, cite in enumerate(citations, 1):
                    st.write(f"[{i}] `{cite}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
    })
