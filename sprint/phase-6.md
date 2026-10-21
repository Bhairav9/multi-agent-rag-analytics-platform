# Phase 6 — Streamlit UI

> Goal: end this phase with a working chat UI at `http://localhost:8501` that connects to the FastAPI WebSocket, streams tokens word-by-word as the agent answers, shows citations, and remembers multi-turn conversation history within a session. The UI must work end-to-end with Docker Compose running.

---

## Task checklist

| # | Task | File(s) | Status |
|---|---|---|---|
| 1 | Streamlit chat app | `src/rag_platform/ui/streamlit_app.py` | ⬜ |
| 2 | Sidebar: ticker filter + session controls | inside `streamlit_app.py` | ⬜ |
| 3 | Token streaming from WebSocket | inside `streamlit_app.py` | ⬜ |
| 4 | Citations display | inside `streamlit_app.py` | ⬜ |
| 5 | Docker Compose: add Streamlit service | `docker-compose.yml` | ⬜ |
| 6 | Makefile: `ui` target | `Makefile` | ⬜ |

Phase 6 fills in `ui/streamlit_app.py` and wires Docker + Makefile. No new tests needed — Streamlit UIs are tested manually.

---

## Core concepts (read once before you type anything)

### 1. How Streamlit works

Streamlit re-runs your script top-to-bottom on every user interaction. State that must survive re-runs lives in `st.session_state` — a dict-like object that persists within a browser session.

```python
# Bad — resets on every re-run
messages = []

# Good — persists across re-runs
if "messages" not in st.session_state:
    st.session_state.messages = []
```

*Used in:* Task 1.

---

### 2. WebSocket connection from Streamlit

Streamlit runs in the browser but the Python backend runs on the server. To connect to the FastAPI WebSocket from Streamlit Python code, use the `websockets` library (sync wrapper via `asyncio.run`):

```python
import asyncio
import websockets

async def _query(ws_url, query, session_id):
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"query": query, "session_id": session_id}))
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "done":
                return msg
            yield msg["content"]   # token
```

*Used in:* Task 3.

---

### 3. Token streaming with `st.write_stream`

Streamlit 1.31+ has `st.write_stream(generator)` — it renders tokens as they arrive, word-by-word, exactly like ChatGPT:

```python
with st.chat_message("assistant"):
    response = st.write_stream(token_generator())
```

`write_stream` takes any generator that yields strings. Our WebSocket yields one token per message.

*Used in:* Task 3.

---

### 4. Session state for chat history

Each turn is stored as `{"role": "user"/"assistant", "content": "..."}` in `st.session_state.messages`. On re-run, we replay all messages to render the full conversation:

```python
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
```

*Used in:* Task 2.

---

### 5. Citations display

After the agent answers, the `done` message contains `citations` (list of accession numbers or URLs). We display them as an expandable section below the answer:

```python
with st.expander("Sources"):
    for i, citation in enumerate(citations, 1):
        st.write(f"[{i}] {citation}")
```

*Used in:* Task 4.

---

### 6. Docker Compose Streamlit service

Streamlit runs as a separate container on port 8501. It needs to reach the FastAPI app — use the Docker internal hostname `app` (the FastAPI service name):

```yaml
streamlit:
  build: .
  command: uv run streamlit run src/rag_platform/ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
  ports:
    - "8501:8501"
  environment:
    - WS_URL=ws://app:8000/ws/chat
  depends_on:
    - app
```

*Used in:* Task 5.

---

## Order to type things

1. `streamlit_app.py` — full chat UI with session state, WebSocket streaming, citations.
2. `docker-compose.yml` — add Streamlit service.
3. `Makefile` — add `ui` target.

---

## Task descriptions

### Task 1–4 — `src/rag_platform/ui/streamlit_app.py`

**What you're doing:** A single-file Streamlit app that:
- Shows a sidebar with the app title, session ID, and a "Clear chat" button
- Renders prior messages from `st.session_state` on each re-run
- Accepts new user queries via `st.chat_input`
- Connects to the FastAPI WebSocket and streams the answer token-by-token
- Shows citations in an expandable section after each answer

**Done when:** Typing a question in the browser shows a streaming answer with `[1]` citations.

```python
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
if "citations" not in st.session_state:
    st.session_state.citations = []

# --- Sidebar ---
with st.sidebar:
    st.title("📊 RAG Analytics")
    st.caption("Ask questions about SEC filings")
    st.divider()
    st.write(f"**Session:** `{st.session_state.session_id[:8]}...`")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.citations = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    st.divider()
    st.caption("Powered by LangGraph + Qdrant + Ollama")

# --- Main area ---
st.title("SEC Filings Q&A")
st.caption("Ask anything about Apple, Microsoft, Google, Tesla, or JPMorgan 10-K/10-Q filings.")

# Replay existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("Sources", expanded=False):
                for i, cite in enumerate(msg["citations"], 1):
                    st.write(f"[{i}] `{cite}`")

# --- WebSocket query ---
async def _stream_query(query: str, session_id: str):
    """Connect to WebSocket, yield tokens, return final citations."""
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"query": query, "session_id": session_id}))
        citations = []
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg["type"] == "token":
                yield msg["content"]
            elif msg["type"] == "done":
                citations = msg.get("citations", [])
                break
            elif msg["type"] == "error":
                yield f"\n\n⚠️ Error: {msg['message']}"
                break
        return citations


def _run_query(query: str, session_id: str):
    """Sync wrapper — run async generator, collect tokens and citations."""
    tokens = []
    citations = []

    async def _collect():
        gen = _stream_query(query, session_id)
        async for token in gen:
            tokens.append(token)
        try:
            nonlocal citations
            citations = gen.ag_frame.f_locals.get("citations", [])
        except Exception:
            pass

    asyncio.run(_collect())
    return "".join(tokens), citations


# --- Handle new input ---
if prompt := st.chat_input("Ask about SEC filings…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer, citations = _run_query(prompt, st.session_state.session_id)
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
```

---

### Task 5 — Docker Compose: add Streamlit service

**What you're doing:** Add a `streamlit` service to `docker-compose.yml` so `docker compose up` starts the UI alongside the API.

Add after the `app` service:

```yaml
  streamlit:
    build: .
    command: >
      uv run streamlit run src/rag_platform/ui/streamlit_app.py
      --server.port 8501
      --server.address 0.0.0.0
      --server.headless true
    ports:
      - "8501:8501"
    environment:
      - WS_URL=ws://app:8000/ws/chat
    depends_on:
      - app
    restart: unless-stopped
```

---

### Task 6 — Makefile `ui` target

```makefile
ui:
	uv run streamlit run src/rag_platform/ui/streamlit_app.py --server.port 8501
```

---

## Definition of done for Phase 6

- `make ui` opens `http://localhost:8501` (FastAPI running separately with `make run`).
- Typing a question returns a streamed answer with citations in an expander.
- "Clear chat" button resets the conversation.
- `docker compose up` starts both `app` (port 8000) and `streamlit` (port 8501).
- `ruff check .` passes.
