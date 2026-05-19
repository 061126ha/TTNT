"""
Pydantic models for the HAUI Agent.

Provides type-safe data structures used across the agent pipeline:
  - ChatMessage: a single chat message with role + content
  - RetrievedChunk: a chunk returned from FAISS vector search
  - AgentResponse: the final response returned to the caller
"""

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(..., description="Message role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message text content")


class RetrievedChunk(BaseModel):
    """A chunk retrieved from the FAISS vector store."""
    chunk_id: int | str = Field(..., description="Unique identifier of the chunk")
    content: str = Field("", description="Text content of the chunk")
    section: str = Field("", description="Top-level section heading")
    subsection: str = Field("", description="Subsection heading")
    score: float = Field(0.0, description="Cosine similarity score")
    text: str = Field("", description="Full text used for embedding (header + content)")
    rerank_score: float | None = Field(None, description="Score assigned by the reranker (None if no reranking)")


class AgentResponse(BaseModel):
    """Response returned by HAUIAgent.chat()."""
    answer: str = Field(..., description="The generated answer text")
    intent: str = Field(..., description="Classified intent: 'related' or 'unrelated'")
    sources: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Retrieved source chunks (empty for off-topic queries)",
    )
