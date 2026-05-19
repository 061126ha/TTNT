"""
Supervisor node for the multi-agent HAUI RAG system.

Classifies the user's query into one of four domains and sets
state["query_type"] so the graph can route to the correct worker agent.
"""

import logging
from typing import Literal

from langchain_core.messages import SystemMessage

from src.agent.state import AgentState
from src.llm.client import get_chat_model

logger = logging.getLogger(__name__)

SUPERVISOR_PROMPT = """Bạn là bộ phân loại câu hỏi cho hệ thống AI của Trường Công nghệ Thông tin và Truyền thông - ĐH Công nghiệp Hà Nội (HAUI - SICT).

Nhiệm vụ: Phân loại câu hỏi của người dùng vào MỘT trong các danh mục sau:

- curriculum:  Câu hỏi về chương trình đào tạo, ngành học, môn học, tín chỉ, chuẩn đầu ra (SO/PEO/PI), khung chương trình, thống kê tuyển sinh/tốt nghiệp.
- regulations: Câu hỏi về thông tin chung của trường/khoa/phòng ban, cơ cấu tổ chức, đội ngũ giảng viên, cơ sở vật chất, chiến lược phát triển, quy chế học tập, chính sách chất lượng, sự kiện, thông báo.
- general:     Chào hỏi, câu hỏi hoàn toàn ngoài phạm vi trường HAUI, hoặc câu hỏi chung không liên quan đến nội dung trên.

Quy tắc:
1. Chỉ trả về ĐÚNG MỘT trong ba từ: curriculum, regulations, general
2. Không giải thích, không thêm dấu câu hay ký tự khác
3. Nếu câu hỏi có thể thuộc nhiều danh mục, chọn danh mục phù hợp nhất"""

_VALID_TYPES = {"curriculum", "regulations", "general"}


def supervisor_node(state: AgentState) -> dict:
    """Classify the user's query and set query_type in state."""
    logger.info("═══ [Node] supervisor ═══")

    # Find the most recent human message
    query = ""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "human":
            query = msg.content
            break

    logger.info("  Query: %.100s", query)

    model = get_chat_model()
    response = model.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        {"role": "user", "content": query},
    ])

    raw = response.content.strip().lower()
    query_type = raw if raw in _VALID_TYPES else "general"

    logger.info("  Classified as: %s (raw: '%s')", query_type, raw)
    return {"query_type": query_type}


def route_to_agent(
    state: AgentState,
) -> Literal["curriculum_generate", "regulation_generate", "general_respond"]:
    """Route to the appropriate worker based on query_type."""
    qt = state.get("query_type", "general")
    routes = {
        "curriculum":  "curriculum_generate",
        "regulations": "regulation_generate",
        "general":     "general_respond",
    }
    destination = routes.get(qt, "general_respond")
    logger.info("  Routing '%s' → %s", qt, destination)
    return destination
