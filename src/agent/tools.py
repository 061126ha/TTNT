"""
Domain-specific retrieval tools for HAUI RAG system.
"""

import logging
from langchain_core.tools import tool

import src.service.reranker as _reranker_module
from src.config import settings
from src.service.vectorstore import curriculum_store, regulation_store

logger = logging.getLogger(__name__)

rerank_debug: list[dict] = []


def _format_chunks(chunks) -> str:
    """Format LlamaIndex NodeWithScore chunks into readable text."""
    if not chunks:
        return "No relevant information found."
        
    formatted = []
    for c in chunks:
        # Xử lý an toàn để lấy metadata dù nó nằm ở đâu
        metadata = getattr(c, "metadata", {})
        if not metadata and hasattr(c, "node"):
            metadata = getattr(c.node, "metadata", {})
            
        section = metadata.get("section", "")
        subsection = metadata.get("subsection", "")
        header = " > ".join(filter(None, [section, subsection]))
        
        # Lấy nội dung text
        text = getattr(c, "text", "")
        if not text and hasattr(c, "node"):
            text = c.node.get_content()
            
        if header:
            formatted.append(f"[{header}]\n{text}")
        else:
            formatted.append(text)
            
    return "\n\n---\n\n".join(formatted)


def _safe_rerank(model, query, raw_chunks, top_k):
    """Protect reranker from None / empty / crash."""
    if not raw_chunks:
        return []

    try:
        return model.rerank(query, raw_chunks, top_k)
    except Exception as e:
        logger.warning("Reranker failed → fallback raw chunks (%s)", e)
        return raw_chunks[:top_k]


@tool
def retrieve_curriculum(query: str) -> str:
    """Retrieve curriculum info (programs, courses, PEO/SO/PI)."""

    logger.info("═══ [Tool] curriculum: %s", query)

    use_rerank = settings.reranker_type != "none"

    fetch_k = (
        settings.top_k * settings.rerank_fetch_multiplier
        if use_rerank
        else settings.top_k
    )

    raw_chunks = curriculum_store.retrieve(query, top_k=fetch_k)

    effective_top_k = settings.rerank_top_k if use_rerank else settings.top_k

    ranked_chunks = _safe_rerank(
        _reranker_module.reranker,
        query,
        raw_chunks,
        effective_top_k,
    )

    rerank_debug.append({
        "tool": "curriculum",
        "reranker_type": settings.reranker_type,
        "fetch_k": fetch_k,
        "retrieved": len(raw_chunks),
        "reranked": len(ranked_chunks),
        "top_k": effective_top_k,
    })

    result = _format_chunks(ranked_chunks)

    logger.info("  Returned %d chunks", len(ranked_chunks))

    return result


@tool
def retrieve_regulations(query: str) -> str:
    """Retrieve general university info (departments, policies, regulations)."""

    logger.info("═══ [Tool] regulations: %s", query)

    use_rerank = settings.reranker_type != "none"

    fetch_k = (
        settings.top_k * settings.rerank_fetch_multiplier
        if use_rerank
        else settings.top_k
    )

    raw_chunks = regulation_store.retrieve(query, top_k=fetch_k)

    effective_top_k = settings.rerank_top_k if use_rerank else settings.top_k

    ranked_chunks = _safe_rerank(
        _reranker_module.reranker,
        query,
        raw_chunks,
        effective_top_k,
    )

    rerank_debug.append({
        "tool": "regulations",
        "reranker_type": settings.reranker_type,
        "fetch_k": fetch_k,
        "retrieved": len(raw_chunks),
        "reranked": len(ranked_chunks),
        "top_k": effective_top_k,
    })

    result = _format_chunks(ranked_chunks)

    logger.info("  Returned %d chunks", len(ranked_chunks))

    return result