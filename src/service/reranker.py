"""
Reranker service layer (fixed version)
"""

import logging
import re
import os
from abc import ABC, abstractmethod

from src.models import RetrievedChunk

logger = logging.getLogger(__name__)

# =========================
# Base
# =========================
class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int):
        pass


# =========================
# Identity
# =========================
class IdentityReranker(BaseReranker):
    def rerank(self, query, chunks, top_k):
        return chunks[:top_k]


# =========================
# Cross Encoder
# =========================
class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    def _load(self):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query, chunks, top_k):
        if self.model is None:
            try:
                self._load()
            except Exception as e:
                logger.error("Load CrossEncoder failed: %s", e)
                return chunks[:top_k]

        texts = [c.text or c.content for c in chunks]
        scores = self.model.predict([(query, t) for t in texts])

        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

        return [
            c.model_copy(update={"rerank_score": float(s)})
            for s, c in ranked[:top_k]
        ]


# =========================
# Factory
# =========================
def get_reranker():
    from src.config import settings

    t = settings.reranker_type

    if t == "cross_encoder":
        return CrossEncoderReranker(settings.cross_encoder_model)

    if t == "llm":
        from .reranker import LLMReranker
        return LLMReranker()

    if t == "cohere":
        from .reranker import CohereReranker
        return CohereReranker(
            settings.cohere_api_key,
            settings.cohere_rerank_model
        )

    return IdentityReranker()


# =========================
# GLOBAL INSTANCE
# =========================
reranker = get_reranker()


def reload_reranker():
    """
    FIX: dùng khi đổi config trong Streamlit
    """
    global reranker
    reranker = get_reranker()