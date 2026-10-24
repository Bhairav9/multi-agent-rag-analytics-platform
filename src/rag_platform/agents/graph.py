"""LangGraph agent graph — assembles all nodes and edges."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rag_platform.agents.critic import make_critic_node
from rag_platform.agents.retrieval import make_retrieval_node
from rag_platform.agents.state import MAX_REVISIONS, AgentState
from rag_platform.agents.supervisor import make_supervisor_node
from rag_platform.agents.synthesizer import make_synthesizer_node
from rag_platform.agents.web_search import web_search_node
from rag_platform.llm.provider import LLMProvider
from rag_platform.retrieval.hybrid import HybridRetriever


def _route_supervisor(state: AgentState) -> str:
    return "web_search" if state.get("needs_web") else "retrieval"


def _route_critic(state: AgentState) -> str:
    if state.get("final_answer"):
        return END
    if state.get("revision_count", 0) < MAX_REVISIONS:
        return "synthesizer"
    return END


async def build_graph(llm: LLMProvider, retriever: HybridRetriever):
    """Build and compile the multi-agent LangGraph graph."""
    supervisor = await make_supervisor_node(llm)
    retrieval = make_retrieval_node(retriever)
    synthesizer = make_synthesizer_node(llm)
    critic = make_critic_node(llm)

    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor)
    graph.add_node("retrieval", retrieval)
    graph.add_node("web_search", web_search_node)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("critic", critic)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", _route_supervisor, ["retrieval", "web_search"])
    graph.add_edge("retrieval", "synthesizer")
    graph.add_edge("web_search", "synthesizer")
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges("critic", _route_critic, ["synthesizer", END])

    return graph.compile()
