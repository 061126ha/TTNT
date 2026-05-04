"""
LangGraph multi-agent state definition.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class AgentState(TypedDict):
    """State for the multi-agent HAUI RAG graph.

    query_type: set by the Supervisor node to route to the correct worker.
      Values: "curriculum" | "regulations" | "general"
    retry_count: tracks rewrite iterations per turn; reset to 0 at invocation start.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    query_type: str
    retry_count: int
