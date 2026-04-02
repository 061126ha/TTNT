"""
High-level wrapper around the LangGraph agent.

Provides the HAUIAgent class which maintains conversation history
across turns and exposes a simple ``chat()`` / ``reset()`` API.
"""

import logging

from .graph import build_graph
from .state import AgentState
from src.models import AgentResponse, RetrievedChunk

logger = logging.getLogger(__name__)


class HAUIAgent:
    """Stateful chatbot agent. Maintains conversation history across turns."""

    def __init__(self) -> None:
        self.graph = build_graph()
        self._messages: list[dict] = []
        logger.info("HAUIAgent initialised")

    def chat(self, query: str) -> AgentResponse:
        """Run one conversational turn and return a typed AgentResponse."""
        state: AgentState = {
            "query": query,
            "intent": "",
            "retrieved_chunks": [],
            "messages": self._messages,
            "answer": "",
        }

        result = self.graph.invoke(state)
        self._messages = result["messages"]

        sources = [
            RetrievedChunk(
                chunk_id=c.chunk_id if isinstance(c, RetrievedChunk) else c["chunk_id"],
                section=c.section if isinstance(c, RetrievedChunk) else c.get("section", ""),
                subsection=c.subsection if isinstance(c, RetrievedChunk) else c.get("subsection", ""),
                score=round(c.score if isinstance(c, RetrievedChunk) else c.get("score", 0), 4),
            )
            for c in result.get("retrieved_chunks", [])
        ]

        return AgentResponse(
            answer=result["answer"],
            intent=result["intent"],
            sources=sources,
        )

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages = []
        logger.info("Conversation history cleared")
