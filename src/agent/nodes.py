"""
LangGraph node functions for the HAUI chatbot agent.

Each function receives the current AgentState and returns a dict
containing only the fields that need updating (LangGraph convention).
"""

import logging

from src.agent.state import AgentState
from src.llm.client import get_client
from src.config import settings
from src.agent.prompt import ROUTER_SYSTEM, GENERATE_SYSTEM, OFF_TOPIC_SYSTEM
from src.service.vectorstore import retrieve_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chat_completion(
    system_prompt: str,
    history: list[dict],
    user_content: str,
    *,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    max_history_pairs: int = 10,
) -> str:
    """Call the chat model and return the assistant's reply text.

    Centralises the OpenAI call so that generate_node and off_topic_node
    share the same error-handling / logging logic without duplicating code.

    Parameters
    ----------
    max_history_pairs : int
        Maximum number of user/assistant pairs to keep from **history**
        to avoid token overflow (each pair = 2 messages).
    """
    client = get_client()

    # Trim history to the most recent N pairs
    max_messages = max_history_pairs * 2
    trimmed_history = history[-max_messages:] if len(history) > max_messages else history

    messages = (
        [{"role": "system", "content": system_prompt}]
        + trimmed_history
        + [{"role": "user", "content": user_content}]
    )
    kwargs: dict = {
        "model": settings.chat_model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> dict:
    """Classify the user query as *related* or *unrelated* to the handbook.

    Passes recent conversation history so that follow-up questions like
    "nói thêm về cái đó" are correctly classified based on context.
    """
    try:
        client = get_client()
        history = state.get("messages", [])

        # Keep at most 6 recent messages for the router (enough context,
        # but not too many tokens for a simple classification task).
        recent_history = history[-6:] if len(history) > 6 else history

        messages = (
            [{"role": "system", "content": ROUTER_SYSTEM}]
            + recent_history
            + [{"role": "user", "content": state["query"]}]
        )

        response = client.chat.completions.create(
            model=settings.chat_model,
            messages=messages,
            temperature=0,
            max_tokens=10,
        )
        verdict = response.choices[0].message.content.strip().lower()
        intent = "related" if verdict == "related" else "unrelated"
    except Exception:
        logger.exception("Router LLM call failed — defaulting to 'related'")
        intent = "related"

    logger.debug("Router intent for query '%.50s…': %s", state["query"], intent)
    return {"intent": intent}


def retrieve_node(state: AgentState) -> dict:
    """Embed the query and retrieve the top-K most similar chunks from FAISS."""
    try:
        chunks = retrieve_context(state["query"])
    except Exception:
        logger.exception("Retrieval failed — continuing with empty context")
        chunks = []

    return {"retrieved_chunks": chunks}


def generate_node(state: AgentState) -> dict:
    """Generate an answer grounded in the retrieved handbook context."""
    # Build context string from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(state["retrieved_chunks"], 1):
        header = " > ".join(filter(None, [chunk.section, chunk.subsection]))
        context_parts.append(f"[{i}] {header}\n{chunk.content}")
    context = "\n\n---\n\n".join(context_parts)

    user_message = (
        f"Context from the student handbook:\n\n{context}\n\n"
        f"Question: {state['query']}"
    )

    history = state.get("messages", [])

    try:
        answer = _chat_completion(
            GENERATE_SYSTEM,
            history,
            user_message,
            temperature=0.3,
        )
    except Exception:
        logger.exception("Generate LLM call failed")
        answer = "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại."

    # Return only the NEW messages — the operator.add reducer in
    # AgentState will append them to the existing history automatically.
    new_messages = [
        {"role": "user", "content": state["query"]},
        {"role": "assistant", "content": answer},
    ]

    return {"answer": answer, "messages": new_messages}


def off_topic_node(state: AgentState) -> dict:
    """Politely redirect the user back to handbook-related questions."""
    history = state.get("messages", [])

    try:
        answer = _chat_completion(
            OFF_TOPIC_SYSTEM,
            history,
            state["query"],
            temperature=0.3,
            max_tokens=256,
        )
    except Exception:
        logger.exception("Off-topic LLM call failed")
        answer = (
            "Xin lỗi, tôi chỉ hỗ trợ các câu hỏi liên quan đến sổ tay sinh viên HAUI. "
            "Vui lòng thử lại với câu hỏi khác."
        )

    # Return only the NEW messages — the operator.add reducer in
    # AgentState will append them to the existing history automatically.
    new_messages = [
        {"role": "user", "content": state["query"]},
        {"role": "assistant", "content": answer},
    ]

    return {"answer": answer, "messages": new_messages}
