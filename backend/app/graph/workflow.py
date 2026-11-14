
from langgraph.graph import StateGraph, START, END

from app.graph.state import GraphState
from app.graph.nodes import (
    security_guardrail,
    retrieve_pinecone,
    retrieve_fallback,
    generate_synthesis
)


def check_security(state: GraphState):
    return "retrieve" if state["is_safe"] else "reject"


def check_retrieval_success(state: GraphState):
    if state["active_source"] == "error" or not state["retrieved_chunks"]:
        return "fallback"

    return "generate"


def reject_query(state: GraphState):
    return {
        "synthesized_answer": (
            "Your query was blocked by the FlameGuard security "
            "system. Please rephrase your question."
        )
    }


workflow = StateGraph(GraphState)

workflow.add_node("security_gate", security_guardrail)
workflow.add_node("reject_query", reject_query)
workflow.add_node("pinecone_retriever", retrieve_pinecone)
workflow.add_node("offline_fallback", retrieve_fallback)
workflow.add_node("llm_generator", generate_synthesis)

workflow.add_edge(START, "security_gate")

workflow.add_conditional_edges(
    "security_gate",
    check_security,
    {
        "retrieve": "pinecone_retriever",
        "reject": "reject_query"
    }
)

workflow.add_edge("reject_query", END)

workflow.add_conditional_edges(
    "pinecone_retriever",
    check_retrieval_success,
    {
        "generate": "llm_generator",
        "fallback": "offline_fallback"
    }
)

workflow.add_edge("offline_fallback", "llm_generator")
workflow.add_edge("llm_generator", END)

flameguard_graph = workflow.compile()
