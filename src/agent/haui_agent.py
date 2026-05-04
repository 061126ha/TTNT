"""
High-level wrapper around the multi-agent LangGraph.

Maintains conversation history and exposes a simple chat() / reset() API.
"""

import logging

from langchain_core.messages import HumanMessage

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
        logger.info("  User query: %s", query)
        logger.info("  History: %d messages", len(self._messages))

        self._messages.append(HumanMessage(content=query))

        result = self.graph.invoke({
            "messages": self._messages,
            "query_type": "",
            "retry_count": 0,
        })

        self._messages = result["messages"]
        query_type = result.get("query_type", "")
        logger.info("  Graph complete. query_type=%s, messages=%d", query_type, len(self._messages))

        for i, msg in enumerate(result["messages"]):
            msg_type = getattr(msg, "type", "unknown")
            content_preview = ""
            if hasattr(msg, "content") and msg.content:
                content_preview = msg.content[:60].replace("\n", " ")
            logger.info("  [%d] %s: %.60s", i, msg_type, content_preview)

        # Extract answer from the last AI message
        answer = ""
        for msg in reversed(result["messages"]):
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                answer = msg.content
                break

        # Intent: "related" when a RAG worker ran (tool message exists), else "unrelated"
        tool_was_called = any(
            hasattr(msg, "type") and msg.type == "tool"
            for msg in result["messages"]
        )
        intent = query_type if query_type else ("related" if tool_was_called else "unrelated")

        logger.info("  Intent: %s | Answer: %.100s", intent, answer)
        logger.info("══════════════════════════════════════════════════")

        return AgentResponse(
            answer=answer,
            intent=intent,
            sources=[],
        )

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages = []
        logger.info("Conversation history cleared")
