"""
Application settings (SAFE VERSION).

- Lazy environment resolution (Streamlit-friendly)
- Safe parsing
- Stronger validation
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Runtime-resolved configuration backed by environment variables."""

    # ─────────────────────────────
    # OpenRouter config
    # ─────────────────────────────

    @property
    def openrouter_base_url(self) -> str:
        return "https://openrouter.ai/api/v1"

    @property
    def openrouter_api_key(self) -> str | None:
        key = os.getenv("OPENROUTER_API_KEY")
        return key if key and key.strip() else None

    def require_api_key(self) -> str:
        key = self.openrouter_api_key
        if not key:
            raise ValueError("OPENROUTER_API_KEY is missing")
        return key

    @property
    def chat_model(self) -> str:
        return os.getenv("OPENROUTER_CHAT_MODEL", "google/gemini-2.0-flash-001")

    @property
    def embedding_model(self) -> str:
        return os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")

    # ─────────────────────────────
    # Retrieval config
    # ─────────────────────────────

    @property
    def top_k(self) -> int:
        return self._safe_int("RETRIEVAL_TOP_K", 5, min_val=1)

    @property
    def curriculum_index_file(self) -> str:
        return os.getenv("FAISS_CURRICULUM_INDEX", "faiss_curriculum.bin")

    @property
    def curriculum_metadata_file(self) -> str:
        return os.getenv("FAISS_CURRICULUM_META", "faiss_curriculum_meta.pkl")

    @property
    def regulation_index_file(self) -> str:
        return os.getenv("FAISS_REGULATION_INDEX", "faiss_regulation.bin")

    @property
    def regulation_metadata_file(self) -> str:
        return os.getenv("FAISS_REGULATION_META", "faiss_regulation_meta.pkl")

    # ─────────────────────────────
    # Reranker config
    # ─────────────────────────────

    @property
    def reranker_type(self) -> str:
        return os.getenv("RERANKER_TYPE", "cohere").lower()

    @property
    def rerank_fetch_multiplier(self) -> int:
        return self._safe_int("RERANK_FETCH_MULTIPLIER", 3, min_val=1)

    @property
    def rerank_top_k(self) -> int:
        """
        Safe rerank top_k:
        - never exceed fetch budget
        - ignore if reranker is disabled
        """
        if self.reranker_type == "none":
            return self.top_k

        env_val = os.getenv("RERANK_TOP_K")
        if not env_val:
            return self.top_k

        val = self._safe_int("RERANK_TOP_K", self.top_k)

        max_allowed = self.top_k * self.rerank_fetch_multiplier
        return min(val, max_allowed)

    @property
    def cross_encoder_model(self) -> str:
        return os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base")

    @property
    def cohere_api_key(self) -> str | None:
        key = os.getenv("COHERE_API_KEY")
        return key if key and key.strip() else None

    @property
    def cohere_rerank_model(self) -> str:
        return os.getenv("COHERE_RERANK_MODEL", "rerank-multilingual-v3.0")

    # ─────────────────────────────
    # Utils
    # ─────────────────────────────

    def _safe_int(self, key: str, default: int, min_val: int = 0) -> int:
        try:
            val = int(os.getenv(key, str(default)))
            return max(min_val, val)
        except Exception:
            return default


settings = Settings()