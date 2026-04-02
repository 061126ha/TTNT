"""
LangGraph agent state definition.
"""

import operator
from typing import Annotated, TypedDict

from src.models import RetrievedChunk


class AgentState(TypedDict):
    """State that flows through the LangGraph agent.

    Fields
    ------
    query : str
        The user's current question.
    intent : str
        Classification result — ``"related"`` or ``"unrelated"``.
    retrieved_chunks : list[RetrievedChunk]
        Chunks returned by the FAISS retrieval step.
    messages : list[dict]
        Full conversation history (role/content dicts for the OpenAI API).
        Uses ``operator.add`` as a reducer so that nodes can return only
        *new* messages and LangGraph will append them automatically.
    answer : str
        The generated answer for the current turn.
    """

    query: str
    intent: str
    retrieved_chunks: list[RetrievedChunk]
    messages: Annotated[list[dict], operator.add]
    answer: str
