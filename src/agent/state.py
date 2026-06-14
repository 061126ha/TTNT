"""
LangGraph multi-agent state definition.
"""

from typing import Annotated, TypedDict, Literal

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


QueryType = Literal["curriculum", "regulation", "general", ""]


class AgentState(TypedDict):
    """
    Shared state for HAUI multi-agent RAG system.

    Fields:
        messages: conversation history (LangGraph-managed)
        query_type: routing decision from supervisor
        retry_count: number of rewrite attempts for current turn
    """

    messages: Annotated[list[AnyMessage], add_messages]
    query_type: QueryType
    retry_count: int