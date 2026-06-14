# =========================================================
# HAUI AGENT SYSTEM — API REFERENCE (OPENROUTER VERSION)
# =========================================================

"""
This system uses OpenRouter API (NOT OpenAI directly).

Supported models:
- Gemini (google/gemini-2.0-flash)
- DeepSeek
- Llama 3.1
"""

# =========================================================
# HAUIAgent
# =========================================================

class HAUIAgent:
    """
    Stateful LangGraph-based RAG Agent.
    """

    def __init__(self):
        """
        Initialize graph + memory.
        """
        pass

    def chat(self, query: str):
        """
        Send query → return structured response.

        Returns:
            AgentResponse
        """
        pass

    def reset(self):
        """
        Clear conversation history.
        """
        pass


# =========================================================
# RESPONSE SCHEMA
# =========================================================

class ChatMessage:
    role: str        # "user" | "assistant"
    content: str


class RetrievedChunk:
    chunk_id: str
    content: str
    section: str
    subsection: str
    score: float              # cosine similarity (approx 0–1)
    text: str
    rerank_score: float | None


class AgentResponse:
    answer: str
    intent: str               # curriculum | regulations | general | related | unrelated
    sources: list[RetrievedChunk]


# =========================================================
# VECTOR STORE
# =========================================================

class VectorStore:
    def retrieve(self, query: str, top_k: int = 5):
        """
        Return top-k relevant chunks.
        """
        pass

    @property
    def is_ready(self) -> bool:
        return True


curriculum_store = VectorStore()
regulation_store = VectorStore()


# =========================================================
# RERANKER SYSTEM
# =========================================================

class BaseReranker:
    def rerank(self, query, chunks, top_k):
        pass


class IdentityReranker(BaseReranker):
    """No rerank"""


class CrossEncoderReranker(BaseReranker):
    """HuggingFace reranker"""


class LLMReranker(BaseReranker):
    """Uses LLM scoring"""


class CohereReranker(BaseReranker):
    """Cohere API reranker"""


def get_reranker():
    """
    Return reranker based on:
        settings.reranker_type
    """
    pass


# IMPORTANT RULE:
# Always import like:
#   import src.service.reranker as reranker
#
# DO NOT:
#   from src.service.reranker import reranker


# =========================================================
# LLM CLIENT (OPENROUTER)
# =========================================================

def get_chat_model():
    """
    Returns OpenRouter-compatible chat model.

    NOT OpenAI-only.

    Supported:
        - Gemini
        - DeepSeek
        - Llama 3.1
    """
    pass


def get_client():
    """
    Embedding client (OpenAI-compatible via OpenRouter).
    """
    pass


# =========================================================
# LANGGRAPH
# =========================================================

def build_graph():
    """
    Build and return compiled LangGraph RAG pipeline.
    """
    pass


def make_generate_node(tool, system_prompt, node_name):
    pass


def make_grade_edge(node_name):
    pass


def make_answer_node(node_name):
    pass


def make_rewrite_node(node_name):
    pass


def general_respond_node(state):
    pass


# =========================================================
# SETTINGS (ENV BASED)
# =========================================================

class Settings:
    chat_model: str
    embedding_model: str
    top_k: int

    curriculum_index_file: str
    regulation_index_file: str

    reranker_type: str
    rerank_top_k: int
    rerank_fetch_multiplier: int

    cohere_api_key: str | None
    cross_encoder_model: str


settings = Settings()


# =========================================================
# KEY ARCHITECTURE NOTE
# =========================================================

"""
Pipeline:

User Query
   ↓
LangGraph Router
   ↓
FAISS Retrieval (curriculum/regulation)
   ↓
Reranker (Cohere / CrossEncoder / LLM / none)
   ↓
OpenRouter LLM (Gemini / DeepSeek / Llama)
   ↓
Final Answer
"""

# =========================================================
# END
# =========================================================