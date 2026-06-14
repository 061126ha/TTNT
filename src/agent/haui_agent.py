"""
High-level wrapper around the multi-agent LangGraph.

Maintains conversation history and exposes a simple chat() / reset() API.
"""

import logging

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from .graph import build_graph
from src.models import AgentResponse

logger = logging.getLogger(__name__)


class HAUIAgent:
    """Stateful multi-agent chatbot using Supervisor-Worker LangGraph pattern."""

    def __init__(self) -> None:
        self.graph = build_graph()
        self._messages: list = []
        logger.info("HAUIAgent initialised (multi-agent Supervisor-Worker)")

    def chat(self, query: str) -> AgentResponse:
        """Run one conversational turn and return a typed AgentResponse."""

        logger.info("╔══════════════════════════════════════════════════╗")
        logger.info("║          HAUIAgent — New Turn                   ║")
        logger.info("╚══════════════════════════════════════════════════╝")

        logger.info("User query: %s", query)
        logger.info("History size: %d", len(self._messages))

        # ── append user message ──
        self._messages.append(HumanMessage(content=query))

        # ── run graph ──
        result = self.graph.invoke({
            "messages": self._messages,
            "query_type": "",
        })

        self._messages = result["messages"]

        query_type = result.get("query_type", "")
        logger.info("Graph finished | query_type=%s", query_type)

        # ─────────────────────────────
        # extract answer (last AIMessage)
        # ─────────────────────────────
        answer = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
                break

        # ─────────────────────────────
        # detect tool usage safely
        # ─────────────────────────────
        tool_was_called = any(
            isinstance(msg, ToolMessage) for msg in result["messages"]
        )

        if query_type:
            intent = query_type
        elif tool_was_called:
            intent = "related"
        else:
            intent = "unrelated"

        logger.info("Intent=%s | Answer preview=%.100s", intent, answer)

        return AgentResponse(
            answer=answer,
            intent=intent,
            sources=[],
        )

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages = []
        logger.info("Conversation history cleared")