from langgraph.graph import END, StateGraph
from src.agent.state import AgentState
from src.agent.nodes import router_node, retrieve_node, generate_node, off_topic_node

def route_by_intent(state: AgentState) -> str:
    return state["intent"]

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("off_topic", off_topic_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "related": "retrieve",
            "unrelated": "off_topic",
        },
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("off_topic", END)

    return graph.compile()
