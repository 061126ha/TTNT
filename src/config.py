"""
Application settings.

Uses a Settings class with properties so that values are read from
os.environ at *access* time, not at *import* time.  This allows
Streamlit sidebar changes (which set os.environ) to take effect
without restarting the process.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Runtime-resolved configuration backed by environment variables."""

    @property
    def openrouter_base_url(self) -> str:
        return "https://openrouter.ai/api/v1"

    @property
    def openrouter_api_key(self) -> str | None:
        return os.getenv("OPENROUTER_API_KEY")

    @property
    def chat_model(self) -> str:
        return os.getenv("OPENROUTER_CHAT_MODEL", "google/gemini-2.0-flash-001")

    @property
    def embedding_model(self) -> str:
        return os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")

    @property
    def top_k(self) -> int:
        return int(os.getenv("RETRIEVAL_TOP_K", "5"))

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

settings = Settings()
