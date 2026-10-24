"""WebSocket chat endpoint — streams agent responses to the Streamlit UI."""

from __future__ import annotations

import json
import logging
import re
import time

from fastapi import APIRouter, Query
from fastapi.websockets import WebSocket, WebSocketDisconnect

from rag_platform.agents.state import AgentState
from rag_platform.api.metrics import CACHE_HITS, CACHE_MISSES
from rag_platform.cache.conversation import load_history, save_turn
from rag_platform.cache.llm_cache import get_cached, set_cached

logger = logging.getLogger(__name__)

ws_router = APIRouter()

_GREETINGS = frozenset({"hi", "hello", "hey", "howdy", "greetings", "sup", "yo"})
_INTRO = (
    "Hello! I'm the **SEC Filings Q&A** assistant. I can help you analyse "
    "**Apple, Microsoft, Google, Tesla,** and **JPMorgan** 10-K/10-Q filings.\n\n"
    "Try asking:\n"
    "- *What was Apple's total revenue in FY2024?*\n"
    "- *How much did Tesla spend on R&D in 2023?*\n"
    "- *What are JPMorgan's key risk factors?*"
)

# Keywords that indicate a genuine SEC/financial query worth running the pipeline on.
_SEC_KEYWORDS = frozenset({
    "revenue", "income", "profit", "loss", "earnings", "eps", "ebitda",
    "cash", "debt", "expense", "cost", "margin", "sales", "fiscal", "quarter",
    "annual", "guidance", "forecast", "balance", "sheet", "filing", "10-k",
    "10-q", "sec", "report", "risk", "factor", "equity", "shares", "dividend",
    "buyback", "acquisition", "r&d", "research", "capex", "operating", "gaap",
    "non-gaap", "segment", "growth", "outlook", "aapl", "msft", "googl",
    "tsla", "jpm", "apple", "microsoft", "google", "tesla", "jpmorgan",
})

_OFF_TOPIC_REPLY = (
    "I'm a **SEC Filings Q&A** assistant — I can only answer questions about "
    "the financial filings of **Apple, Microsoft, Google, Tesla,** and **JPMorgan**.\n\n"
    "Try asking something like:\n"
    "- *What was Apple's total revenue in FY2024?*\n"
    "- *How much cash does Microsoft hold?*\n"
    "- *What are Tesla's main risk factors?*"
)


def _is_sec_query(query: str) -> bool:
    """Return True if the query contains at least one SEC/financial keyword."""
    words = set(re.findall(r"[a-z0-9&]+", query.lower()))
    return bool(words & _SEC_KEYWORDS)


async def _send_status(ws: WebSocket, content: str) -> None:
    await ws.send_text(json.dumps({"type": "status", "content": content}))


@ws_router.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket, api_key: str = Query(...)) -> None:
    await websocket.accept()
    if api_key != websocket.app.state.settings.api_key:
        await websocket.send_text(json.dumps({"type": "error", "message": "Unauthorized"}))
        await websocket.close(code=1008)
        return
    graph = websocket.app.state.graph
    redis = websocket.app.state.redis

    logger.info("WebSocket connected", extra={"session_id": "pending"})
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            query: str = data.get("query", "").strip()
            session_id: str = data.get("session_id", "default")

            if not query:
                continue

            q_preview = query[:120]
            logger.info(
                "Query received",
                extra={"session_id": session_id, "query": q_preview},
            )
            t0 = time.perf_counter()

            # Fast path 1: greetings skip the LLM pipeline entirely.
            if query.lower().strip("!?.") in _GREETINGS:
                logger.info(
                    "Greeting shortcut",
                    extra={"session_id": session_id, "query": q_preview},
                )
                await websocket.send_text(json.dumps({"type": "token", "content": _INTRO}))
                await websocket.send_text(json.dumps({"type": "done", "final_answer": _INTRO, "citations": []}))
                continue

            # Fast path 2: conversational / off-topic queries skip the pipeline too.
            if not _is_sec_query(query):
                logger.info(
                    "Off-topic shortcut",
                    extra={"session_id": session_id, "query": q_preview},
                )
                await websocket.send_text(json.dumps({"type": "token", "content": _OFF_TOPIC_REPLY}))
                await websocket.send_text(json.dumps({"type": "done", "final_answer": _OFF_TOPIC_REPLY, "citations": []}))
                continue

            cached = await get_cached(query, redis)
            if cached is not None:
                CACHE_HITS.inc()
            else:
                CACHE_MISSES.inc()
            if cached is not None:
                logger.info(
                    "Cache hit",
                    extra={"session_id": session_id, "query": q_preview},
                )
                await _send_status(websocket, "Serving from cache...")
                await websocket.send_text(json.dumps({"type": "token", "content": cached["final_answer"]}))
                # Explicit fields — never let cached keys override the "type" field.
                await websocket.send_text(json.dumps({
                    "type": "done",
                    "final_answer": cached.get("final_answer", ""),
                    "citations": cached.get("citations", []),
                }))
                continue

            history = await load_history(session_id, redis)
            initial_state = AgentState(
                query=query,
                chat_history=history,
                session_id=session_id,
                needs_web=False,
                retrieved_chunks=[],
                web_results=[],
                draft_answer="",
                final_answer="",
                citations=[],
                critic_feedback="",
                revision_count=0,
            )

            await _send_status(websocket, "Searching SEC corpus...")
            logger.info("Graph invoked", extra={"session_id": session_id})

            prev_state: dict = {}
            sent_answer = ""
            result: dict = {}

            async for state in graph.astream(initial_state, stream_mode="values"):
                result = state

                if len(state.get("retrieved_chunks", [])) > len(prev_state.get("retrieved_chunks", [])):
                    logger.info(
                        "Retrieval complete",
                        extra={"session_id": session_id, "chunks": len(state["retrieved_chunks"])},
                    )
                    await _send_status(websocket, "Generating answer...")
                prev_state = state

                answer: str = state.get("final_answer", "")
                if answer and answer != sent_answer:
                    new_content = answer[len(sent_answer):]
                    await websocket.send_text(json.dumps({"type": "token", "content": new_content}))
                    sent_answer = answer

            final_answer: str = result.get("final_answer", "")
            citations: list[str] = result.get("citations", [])
            duration_ms = round((time.perf_counter() - t0) * 1000)
            logger.info(
                "Response sent",
                extra={
                    "session_id": session_id,
                    "duration_ms": duration_ms,
                    "answer_len": len(final_answer),
                    "citations": len(citations),
                },
            )
            await websocket.send_text(json.dumps({"type": "done", "final_answer": final_answer, "citations": citations}))

            if final_answer:  # don't persist empty answers from silent graph failures
                await set_cached(query, {"final_answer": final_answer, "citations": citations}, redis)
                await save_turn(session_id, query, final_answer, redis)
            else:
                logger.warning("Empty final_answer — skipping cache and history", extra={"session_id": session_id})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
