"""
Supervisor node for the multi-agent HAUI RAG system.
"""

import logging
from typing import Literal

from langchain_core.messages import SystemMessage
from src.agent.state import AgentState
from src.llm.client import get_chat_model

logger = logging.getLogger(__name__)

SUPERVISOR_PROMPT = """Bạn là bộ phân loại câu hỏi cho hệ thống AI của Trường Công nghệ Thông tin và Truyền thông - ĐH Công nghiệp Hà Nội (HAUI - SICT).

Nhiệm vụ: Phân loại câu hỏi của người dùng vào MỘT trong các danh mục:

- curriculum: câu hỏi về chương trình đào tạo, ngành học, môn học, tín chỉ, chuẩn đầu ra (SO/PEO/PI), khung chương trình.
- regulation: câu hỏi về thông tin chung của trường/khoa/phòng ban, cơ cấu tổ chức, giảng viên, cơ sở vật chất, quy chế, chính sách, sự kiện.
- general: câu hỏi ngoài phạm vi HAUI hoặc chào hỏi / câu hỏi chung.

QUY TẮC:
- Chỉ trả về đúng 1 từ: curriculum | regulation | general
- Không giải thích
"""

_VALID_TYPES = {"curriculum", "regulation", "general"}


def supervisor_node(state: AgentState) -> dict:
    logger.info("═══ [Node] supervisor ═══")

    query = ""
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) == "human":
            query = msg.content
            break

    logger.info("  Query: %.100s", query)

    model = get_chat_model()
    response = model.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        {"role": "user", "content": query},
    ])

    raw = response.content.strip().lower()

    # normalize mạnh (fix lỗi LLM trả dư ký tự)
    if "curriculum" in raw:
        query_type = "curriculum"
    elif "regulation" in raw or "regulations" in raw:
        query_type = "regulation"
    else:
        query_type = "general"

    logger.info("  Classified: %s (raw: %s)", query_type, raw)
    return {"query_type": query_type}


def route_to_agent(state: AgentState):
    qt = state.get("query_type", "general")

    routes = {
        "curriculum": "curriculum_generate",
        "regulation": "regulation_generate",
        "general": "general_respond",
    }

    destination = routes.get(qt, "general_respond")

    logger.info("  Routing '%s' → %s", qt, destination)
    return destination