"""WebSocket endpoint — streams agent graph tokens to the client."""

from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from qdrant_client import QdrantClient

from rag_platform.agents.graph import build_graph
from rag_platform.agents.state import AgentState
from rag_platform.cache.conversation import load_history, save_turn
from rag_platform.cache.llm_cache import get_cached, set_cached
from rag_platform.cache.redis_client import get_redis_client
from rag_platform.config import get_settings
from rag_platform.llm.provider import get_llm_provider
from rag_platform.retrieval.bm25 import FastEmbedSparse
from rag_platform.retrieval.embeddings import FastEmbedDense
from rag_platform.retrieval.hybrid import HybridRetriever
from rag_platform.retrieval.qdrant_store import QdrantStore

ws_router = APIRouter()


@ws_router.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    settings = get_settings()

    store = QdrantStore(
        QdrantClient(url=settings.qdrant_url), settings.qdrant_collection
    )

    retriever = HybridRetriever(FastEmbedDense(), FastEmbedSparse(), store)
    llm = get_llm_provider()
    redis = get_redis_client()
    graph = await build_graph(llm, retriever)

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            query: str = data.get("query", "").strip()
            session_id: str = data.get("session_id", "default")

            if not query:
                continue

            cached = await get_cached(query, redis)
            if cached:
                await websocket.send_text(json.dumps({"type": "token", "content": cached["final_answer"]}))
                await websocket.send_text(json.dumps({"type": "done", **cached}))
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

            result = await graph.ainvoke(initial_state)
            final_answer = result.get("final_answer", "")
            citations = result.get("citations", [])

            await websocket.send_text(json.dumps({"type": "token", "content": final_answer}))
            await websocket.send_text(json.dumps({
                "type": "done",
                "final_answer": final_answer,
                "citations": citations,
            }))

            await set_cached(query, {"final_answer": final_answer, "citations": citations}, redis)
            await save_turn(session_id, query, final_answer, redis)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))

        