"""
Multi-agent LangGraph for the HAUI RAG chatbot.

Architecture: Supervisor-Worker pattern with 2 RAG workers + 1 general fallback.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.state import AgentState
from src.agent.tools import retrieve_curriculum, retrieve_regulations
from src.agent.supervisor import supervisor_node, route_to_agent
from src.agent.nodes import (
    make_generate_node,
    make_answer_node,
    make_rewrite_node,
    make_grade_edge,
    general_respond_node,
    CURRICULUM_SYSTEM,
    REGULATION_SYSTEM,
)


def _add_rag_worker(workflow: StateGraph, prefix: str, tool, system_prompt: str) -> None:
    """
    Add a full RAG worker pipeline:
    generate → retrieve → grade → (answer | rewrite loop)
    """

    generate = make_generate_node(tool, system_prompt, f"{prefix}_generate")
    retrieve = ToolNode([tool])
    grade = make_grade_edge(f"{prefix}_grade")
    rewrite = make_rewrite_node(f"{prefix}_rewrite")
    answer = make_answer_node(f"{prefix}_answer")

    # register nodes
    workflow.add_node(f"{prefix}_generate", generate)
    workflow.add_node(f"{prefix}_retrieve", retrieve)
    workflow.add_node(f"{prefix}_rewrite", rewrite)
    workflow.add_node(f"{prefix}_answer", answer)

    # ─────────────────────────────
    # generate → tool call OR END
    # ─────────────────────────────
    workflow.add_conditional_edges(
        f"{prefix}_generate",
        tools_condition,
        {
            "tools": f"{prefix}_retrieve",
            END: END,
        },
    )

    # ─────────────────────────────
    # retrieve → grade decision
    # ─────────────────────────────
    def grade_router(state):
        return grade(state)

    workflow.add_conditional_edges(
        f"{prefix}_retrieve",
        grade_router,
        {
            "generate_answer": f"{prefix}_answer",
            "rewrite_question": f"{prefix}_rewrite",
        },
    )

    # ─────────────────────────────
    # answer → END
    # rewrite → loop back
    # ─────────────────────────────
    workflow.add_edge(f"{prefix}_answer", END)
    workflow.add_edge(f"{prefix}_rewrite", f"{prefix}_generate")


def build_graph():
    """Compile full supervisor-worker LangGraph."""

    workflow = StateGraph(AgentState)

    # ── Supervisor ──
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges("supervisor", route_to_agent)

    # ── Workers ──
    _add_rag_worker(workflow, "curriculum", retrieve_curriculum, CURRICULUM_SYSTEM)
    _add_rag_worker(workflow, "regulation", retrieve_regulations, REGULATION_SYSTEM)

    # ── Fallback ──
    workflow.add_node("general_respond", general_respond_node)
    workflow.add_edge("general_respond", END)

    return workflow.compile()