"""
Pydantic models for the HAUI Agent.

Type-safe structures for:
- Retrieval (FAISS + reranker)
- Agent response
"""

from pydantic import BaseModel, Field


# =========================
# Retrieved Chunk
# =========================
class RetrievedChunk(BaseModel):
    """A chunk retrieved from FAISS + reranker pipeline."""

    chunk_id: int = Field(..., description="Unique chunk ID")

    content: str = Field("", description="Raw text content")

    section: str = Field("", description="Top-level section")

    subsection: str = Field("", description="Subsection heading")

    score: float = Field(
        0.0,
        description="FAISS similarity score"
    )

    rerank_score: float | None = Field(
        None,
        description="Reranker score (if applied)"
    )

    # unified field (important fix)
    text: str = Field(
        "",
        description="Final text used for embedding/retrieval (header + content)"
    )

    class Config:
        extra = "ignore"
        validate_assignment = True


# =========================
# Agent Response
# =========================
class AgentResponse(BaseModel):
    """Final output of HAUIAgent.chat()."""

    answer: str = Field(..., description="Final generated answer")

    intent: str = Field(
        ...,
        description="curriculum | regulations | general | related | unrelated"
    )

    sources: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Retrieved chunks used for answer"
    )

    class Config:
        extra = "ignore"
        validate_assignment = True