"""Streamlit chat UI — connects to FastAPI WebSocket and streams tokens."""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import streamlit as st
import websockets

_WS_BASE = os.getenv("WS_URL", "ws://localhost:8000/ws/chat")
_API_KEY = os.getenv("API_KEY", "change-me")
WS_URL = f"{_WS_BASE}?api_key={_API_KEY}"

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
    st.caption("Powered by LangGraph + Qdrant + Claude")

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
async def _stream(
    query: str,
    session_id: str,
    text_ph,
    status_ph,
) -> tuple[str, list[str]]:
    """Connect to the WebSocket and stream tokens + status updates into placeholders.

    Retries up to 3 times with a 30-second gap to handle Render free-tier cold starts
    (the backend spins down after 15 min of inactivity; Render returns HTTP 502 while it
    wakes up, which takes ~30-60 s).
    """
    citations: list[str] = []
    last_exc: Exception | None = None

    for attempt in range(3):
        tokens: list[str] = []
        if attempt > 0:
            status_ph.caption(
                f"⏳ Backend is warming up — retrying in 30 s… (attempt {attempt + 1}/3)"
            )
            await asyncio.sleep(30)

        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=300,
                close_timeout=10,
            ) as ws:
                await ws.send(json.dumps({"query": query, "session_id": session_id}))
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=300)
                    msg = json.loads(raw)
                    if msg["type"] == "status":
                        status_ph.caption(f"⏳ {msg['content']}")
                    elif msg["type"] == "token":
                        tokens.append(msg["content"])
                        text_ph.markdown("".join(tokens) + " ▌")
                    elif msg["type"] == "done":
                        citations = msg.get("citations", [])
                        return "".join(tokens), citations
                    elif msg["type"] == "error":
                        tokens.append(f"\n\n⚠️ {msg['message']}")
                        return "".join(tokens), citations

        except asyncio.TimeoutError:
            return (
                "\n\n⚠️ Request timed out — the model is still loading. Try again.",
                [],
            )
        except Exception as exc:
            last_exc = exc
            err = str(exc)
            # Retry only on connection-level failures (cold-start 502/503, refused).
            if attempt < 2 and any(
                code in err for code in ("502", "503", "Connection", "refused")
            ):
                continue
            return f"\n\n⚠️ Could not reach the API: {exc}", []

    return (
        f"\n\n⚠️ Backend still warming up after 3 attempts. "
        "Wait 30 s and try again. "
        f"(Last error: {last_exc})",
        [],
    )


# --- Handle new input ---
if prompt := st.chat_input("Ask about SEC filings…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_ph = st.empty()
        text_ph = st.empty()

        answer, citations = asyncio.run(
            _stream(prompt, st.session_state.session_id, text_ph, status_ph)
        )

        status_ph.empty()
        text_ph.markdown(answer)

        if citations:
            with st.expander("Sources", expanded=False):
                for i, cite in enumerate(citations, 1):
                    st.write(f"[{i}] `{cite}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
    })
