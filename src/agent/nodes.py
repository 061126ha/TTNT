"""
LangGraph node functions for the multi-agent HAUI RAG chatbot.

Architecture: Supervisor-Worker with 3 RAG workers + 1 general fallback.

Factory functions (make_generate_node, make_answer_node) create per-agent
node closures bound to a specific tool and system prompt, avoiding code
duplication across curriculum / regulation / web workers.

Shared nodes (grade_documents, rewrite_question) are reused by all workers.
"""

import logging
from typing import Callable, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.llm.client import get_chat_model

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CURRICULUM_SYSTEM = (
    "Bạn là trợ lý AI chuyên về chương trình đào tạo của Trường Công nghệ "
    "Thông tin và Truyền thông - ĐH Công nghiệp Hà Nội (HAUI - SICT). "
    "Bạn có quyền truy cập vào cơ sở dữ liệu chương trình đào tạo, bao gồm "
    "các ngành học, mục tiêu đào tạo (PEO), chuẩn đầu ra (SO), tiêu chí đánh giá (PI) "
    "và chương trình khung. "
    "Hãy dùng tool retrieve_curriculum để tìm thông tin trước khi trả lời. "
    "Trả lời bằng cùng ngôn ngữ với câu hỏi của người dùng."
)

REGULATION_SYSTEM = (
    "Bạn là trợ lý AI về thông tin chung của Trường Công nghệ "
    "Thông tin và Truyền thông - ĐH Công nghiệp Hà Nội (HAUI - SICT). "
    "Bạn có quyền truy cập vào cơ sở dữ liệu về giới thiệu nhà trường, các khoa, "
    "phòng ban, chiến lược phát triển và các chính sách của nhà trường. "
    "Hãy dùng tool retrieve_regulations để tìm thông tin trước khi trả lời. "
    "Trả lời bằng cùng ngôn ngữ với câu hỏi của người dùng."
)

GENERAL_SYSTEM = (
    "Bạn là trợ lý AI thân thiện của Trường Công nghệ Thông tin và Truyền thông "
    "- ĐH Công nghiệp Hà Nội (HAUI - SICT). "
    "Câu hỏi này không thuộc phạm vi cơ sở dữ liệu của trường. "
    "Hãy trả lời thân thiện và gợi ý người dùng đặt câu hỏi liên quan đến "
    "chương trình đào tạo, quy chế hoặc thông tin của trường nếu phù hợp. "
    "Trả lời bằng cùng ngôn ngữ với câu hỏi của người dùng."
)

GRADE_PROMPT = (
    "Bạn là người đánh giá mức độ liên quan của tài liệu được truy xuất "
    "đối với câu hỏi của người dùng.\n\n"
    "Tài liệu được truy xuất:\n\n{context}\n\n"
    "Câu hỏi của người dùng: {question}\n\n"
    "Nếu tài liệu chứa từ khóa hoặc ý nghĩa ngữ nghĩa liên quan đến câu hỏi, "
    "hãy đánh giá là có liên quan.\n"
    "Cho điểm nhị phân 'yes' hoặc 'no' để chỉ ra mức độ liên quan."
)

REWRITE_PROMPT = (
    "Hãy phân tích câu hỏi sau và cải thiện nó để tìm kiếm hiệu quả hơn.\n"
    "Câu hỏi gốc:\n"
    "-------\n"
    "{question}\n"
    "-------\n"
    "Viết lại câu hỏi được cải thiện:"
)

GENERATE_PROMPT = (
    "Bạn là trợ lý trả lời câu hỏi về trường HAUI - SICT. "
    "Dùng các đoạn văn bản được truy xuất sau đây để trả lời câu hỏi. "
    "Nếu không biết câu trả lời, hãy nói thẳng là không có thông tin. "
    "Trả lời bằng cùng ngôn ngữ với câu hỏi. Ngắn gọn, chính xác và thân thiện.\n\n"
    "Câu hỏi: {question}\n\n"
    "Ngữ cảnh: {context}"
)


# ---------------------------------------------------------------------------
# Pydantic model for structured grading
# ---------------------------------------------------------------------------

class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


# ---------------------------------------------------------------------------
# Factory: generate_query_or_respond node
# ---------------------------------------------------------------------------

def make_generate_node(tool, system_prompt: str, node_name: str) -> Callable:
    """Create a generate_query_or_respond node bound to a specific tool and prompt.

    Args:
        tool: The LangChain @tool this agent should use for retrieval.
        system_prompt: Domain-specific system instructions for the LLM.
        node_name: Used only for logging (e.g. "curriculum_generate").
    """
    model_with_tool = get_chat_model().bind_tools([tool])

    def node(state: AgentState) -> dict:
        logger.info("═══ [Node] %s ═══", node_name)
        query_preview = state["messages"][-1].content if state["messages"] else "(empty)"
        logger.info("  Query: %.100s", query_preview)

        messages = list(state["messages"])
        if not messages or not (hasattr(messages[0], "type") and messages[0].type == "system"):
            messages = [SystemMessage(content=system_prompt)] + messages

        response = model_with_tool.invoke(messages)

        has_tc = hasattr(response, "tool_calls") and response.tool_calls
        if has_tc:
            logger.info("  ✅ Decision: CALL TOOL %s", tool.name)
        else:
            logger.info("  💬 Decision: RESPOND DIRECTLY")

        return {"messages": [response]}

    node.__name__ = node_name
    return node


# ---------------------------------------------------------------------------
# Factory: generate_answer node
# ---------------------------------------------------------------------------

def make_answer_node(node_name: str) -> Callable:
    """Create a generate_answer node (domain-agnostic — uses GENERATE_PROMPT)."""

    def node(state: AgentState) -> dict:
        logger.info("═══ [Node] %s ═══", node_name)

        question = ""
        for msg in state["messages"]:
            if hasattr(msg, "type") and msg.type == "human":
                question = msg.content

        context = state["messages"][-1].content
        logger.info("  Context: %d chars", len(context))

        model = get_chat_model()
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = model.invoke([{"role": "user", "content": prompt}])

        logger.info("  Answer: %.120s", response.content)
        return {"messages": [response]}

    node.__name__ = node_name
    return node


# ---------------------------------------------------------------------------
# Factory: rewrite_question node
# ---------------------------------------------------------------------------

def make_rewrite_node(node_name: str) -> Callable:
    """Create a rewrite_question node."""

    def node(state: AgentState) -> dict:
        logger.info("═══ [Node] %s ═══", node_name)

        question = ""
        for msg in state["messages"]:
            if hasattr(msg, "type") and msg.type == "human":
                question = msg.content

        current_count = state.get("retry_count", 0)
        logger.info("  Original: %.80s (retry %d/%d)", question, current_count, MAX_RETRIES)

        model = get_chat_model()
        prompt = REWRITE_PROMPT.format(question=question)
        response = model.invoke([{"role": "user", "content": prompt}])

        logger.info("  Rewritten: %.80s", response.content)
        return {
            "messages": [HumanMessage(content=response.content)],
            "retry_count": current_count + 1,
        }

    node.__name__ = node_name
    return node


# ---------------------------------------------------------------------------
# Shared: grade_documents (conditional edge — reused by all workers)
# ---------------------------------------------------------------------------

def make_grade_edge(node_name: str) -> Callable:
    """Create a grade_documents conditional edge."""

    def edge(state: AgentState) -> Literal["generate_answer", "rewrite_question"]:
        logger.info("═══ [Edge] %s ═══", node_name)

        retry_count = state.get("retry_count", 0)

        if retry_count >= MAX_RETRIES:
            logger.info("  ⚠️ Max retries (%d) reached → generate_answer", MAX_RETRIES)
            return "generate_answer"

        model = get_chat_model()
        grader = model.with_structured_output(GradeDocuments)

        question = ""
        for msg in state["messages"]:
            if hasattr(msg, "type") and msg.type == "human":
                question = msg.content

        context = state["messages"][-1].content
        logger.info("  Question: %.80s", question)
        logger.info("  Context length: %d chars (retry %d/%d)", len(context), retry_count, MAX_RETRIES)

        prompt = GRADE_PROMPT.format(question=question, context=context)
        response = grader.invoke([{"role": "user", "content": prompt}])
        score = response.binary_score

        if score == "yes":
            logger.info("  ✅ RELEVANT → generate_answer")
            return "generate_answer"
        else:
            logger.info("  ❌ NOT RELEVANT → rewrite_question")
            return "rewrite_question"

    edge.__name__ = node_name
    return edge


# ---------------------------------------------------------------------------
# General fallback node (no retrieval)
# ---------------------------------------------------------------------------

def general_respond_node(state: AgentState) -> dict:
    """Respond directly without RAG — used for off-topic queries."""
    logger.info("═══ [Node] general_respond ═══")

    messages = list(state["messages"])
    if not messages or not (hasattr(messages[0], "type") and messages[0].type == "system"):
        messages = [SystemMessage(content=GENERAL_SYSTEM)] + messages

    model = get_chat_model()
    response = model.invoke(messages)

    logger.info("  Response: %.100s", response.content)
    return {"messages": [response]}

