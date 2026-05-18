"""
Domain-specific retrieval tools for the multi-agent HAUI RAG system.

Each tool queries its own dedicated FAISS index built from web_chunks.json:
  - retrieve_curriculum  → faiss_curriculum.*   (programs, courses, admissions)
  - retrieve_regulations → faiss_regulation.*   (school info, departments, policies)
"""

import logging

from langchain_core.tools import tool

from src.config import settings
from src.service.reranker import reranker
from src.service.vectorstore import curriculum_store, regulation_store

logger = logging.getLogger(__name__)

# Populated by each tool call during one agent turn; cleared by the UI before each new turn.
rerank_debug: list[dict] = []


def _format_chunks(chunks) -> str:
    if not chunks:
        return "Không tìm thấy thông tin liên quan."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = " > ".join(filter(None, [chunk.section, chunk.subsection]))
        parts.append(f"[{i}] {header}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


@tool
def retrieve_curriculum(query: str) -> str:
    """Tìm thông tin về chương trình đào tạo của trường HAUI - SICT.

    Dùng tool này để tra cứu:
    - Các ngành đào tạo đại học và sau đại học
    - Mục tiêu đào tạo (PEO), chuẩn đầu ra (SO), tiêu chí đánh giá (PI)
    - Chương trình khung, danh sách môn học, số tín chỉ
    - Thống kê tuyển sinh và tốt nghiệp

    Args:
        query: Câu hỏi về chương trình đào tạo (tiếng Việt hoặc tiếng Anh).
    """
    logger.info("═══ [Tool] retrieve_curriculum: %s", query)
    fetch_k = settings.top_k * settings.rerank_fetch_multiplier if settings.reranker_type != "none" else None
    raw_chunks = curriculum_store.retrieve(query, top_k=fetch_k)
    ranked_chunks = reranker.rerank(query, raw_chunks, top_k=settings.rerank_top_k)
    rerank_debug.append({
        "tool": "curriculum",
        "reranker_type": settings.reranker_type,
        "fetch_k": fetch_k or settings.top_k,
        "retrieved": len(raw_chunks),
        "reranked": len(ranked_chunks),
        "chunks": ranked_chunks,
    })
    result = _format_chunks(ranked_chunks)
    logger.info("  Found %d chunks (%d chars)", len(ranked_chunks), len(result))
    return result


@tool
def retrieve_regulations(query: str) -> str:
    """Tìm thông tin chung về trường HAUI - SICT: giới thiệu, khoa, phòng ban, chính sách.

    Dùng tool này để tra cứu:
    - Giới thiệu về trường, cơ cấu tổ chức, đội ngũ giảng viên
    - Thông tin về các khoa, phòng ban, trung tâm nghiên cứu
    - Chiến lược phát triển, quy chế học tập, chính sách chất lượng
    - Hoạt động khoa học công nghệ, sự kiện, thông báo

    Args:
        query: Câu hỏi về thông tin trường, quy chế, chính sách (tiếng Việt hoặc tiếng Anh).
    """
    logger.info("═══ [Tool] retrieve_regulations: %s", query)
    fetch_k = settings.top_k * settings.rerank_fetch_multiplier if settings.reranker_type != "none" else None
    raw_chunks = regulation_store.retrieve(query, top_k=fetch_k)
    ranked_chunks = reranker.rerank(query, raw_chunks, top_k=settings.rerank_top_k)
    rerank_debug.append({
        "tool": "regulations",
        "reranker_type": settings.reranker_type,
        "fetch_k": fetch_k or settings.top_k,
        "retrieved": len(raw_chunks),
        "reranked": len(ranked_chunks),
        "chunks": ranked_chunks,
    })
    result = _format_chunks(ranked_chunks)
    logger.info("  Found %d chunks (%d chars)", len(ranked_chunks), len(result))
    return result
