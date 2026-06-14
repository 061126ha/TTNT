"""
LangGraph node functions for the multi-agent HAUI RAG chatbot.
"""

import logging
from typing import Callable, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from src.agent.state import AgentState
from src.llm.client import get_chat_model

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CURRICULUM_SYSTEM = (
    "Bạn là trợ lý AI chuyên về chương trình đào tạo HAUI-SICT. "
    "Luôn dùng tool retrieve_curriculum trước khi trả lời."
)

REGULATION_SYSTEM = (
    "Bạn là trợ lý AI về thông tin quy chế và tổ chức HAUI-SICT. "
    "Luôn dùng tool retrieve_regulations trước khi trả lời."
)

GENERAL_SYSTEM = (
    "Bạn là trợ lý AI thân thiện của HAUI-SICT. "
    "Trả lời các câu hỏi ngoài phạm vi một cách ngắn gọn và gợi ý người dùng hỏi đúng chủ đề."
)

GRADE_PROMPT = (
    "Đánh giá mức độ liên quan của tài liệu với câu hỏi.\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n\n"
    "Trả lời yes hoặc no."
)

REWRITE_PROMPT = (
    "Viết lại câu hỏi để truy xuất tốt hơn:\n{question}"
)

GENERATE_PROMPT = (
    "Trả lời dựa trên context:\n\n"
    "Question: {question}\n\n"
    "Context: {context}"
)


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="yes or no")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_last_user_message(messages):
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _get_last_context(messages):
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return msg.content
    return ""


# ---------------------------------------------------------------------------
# Generate node
# ---------------------------------------------------------------------------

def make_generate_node(tool, system_prompt: str, node_name: str) -> Callable:
    model = get_chat_model().bind_tools([tool])

    def node(state: AgentState) -> dict:
        logger.info("═══ %s ═══", node_name)

        messages = list(state["messages"])

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages

        response = model.invoke(messages)

        if getattr(response, "tool_calls", None):
            logger.info("→ CALL TOOL")
        else:
            logger.info("→ DIRECT RESPONSE")

        return {"messages": [response]}

    return node


# ---------------------------------------------------------------------------
# Answer node
# ---------------------------------------------------------------------------

def make_answer_node(node_name: str) -> Callable:

    def node(state: AgentState) -> dict:
        logger.info("═══ %s ═══", node_name)

        question = _get_last_user_message(state["messages"])
        context = _get_last_context(state["messages"])

        model = get_chat_model()

        prompt = GENERATE_PROMPT.format(
            question=question,
            context=context
        )

        response = model.invoke([HumanMessage(content=prompt)])

        logger.info("Answer ready")
        return {"messages": [response]}

    return node


# ---------------------------------------------------------------------------
# Rewrite node
# ---------------------------------------------------------------------------

def make_rewrite_node(node_name: str) -> Callable:

    def node(state: AgentState) -> dict:
        logger.info("═══ %s ═══", node_name)

        question = _get_last_user_message(state["messages"])
        retry = state.get("retry_count", 0)

        model = get_chat_model()

        prompt = REWRITE_PROMPT.format(question=question)
        response = model.invoke([HumanMessage(content=prompt)])

        return {
            "messages": [HumanMessage(content=response.content)],
            "retry_count": retry + 1,
        }

    return node


# ---------------------------------------------------------------------------
# Grade edge
# ---------------------------------------------------------------------------

def make_grade_edge(node_name: str) -> Callable:

    def edge(state: AgentState) -> Literal["generate_answer", "rewrite_question"]:
        logger.info("═══ %s ═══", node_name)

        retry = state.get("retry_count", 0)

        if retry >= MAX_RETRIES:
            return "generate_answer"

        question = _get_last_user_message(state["messages"])
        context = _get_last_context(state["messages"])

        model = get_chat_model().with_structured_output(GradeDocuments)

        prompt = GRADE_PROMPT.format(
            question=question,
            context=context
        )

        try:
            result = model.invoke([HumanMessage(content=prompt)])
            score = result.binary_score
        except Exception:
            score = "no"

        if score == "yes":
            return "generate_answer"
        return "rewrite_question"


    return edge


# ---------------------------------------------------------------------------
# General fallback
# ---------------------------------------------------------------------------

def general_respond_node(state: AgentState) -> dict:
    logger.info("═══ general_respond ═══")

    messages = list(state["messages"])

    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=GENERAL_SYSTEM)] + messages

    response = get_chat_model().invoke(messages)

    return {"messages": [response]}