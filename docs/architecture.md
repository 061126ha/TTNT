"""
=========================================================
HAUI AGENT SYSTEM — ARCHITECTURE (OPENROUTER VERSION)
=========================================================

COPY-PASTE DOCUMENTATION + SYSTEM DESIGN

NOTE:
- LLM provider: OpenRouter (NOT OpenAI direct)
- Supported models:
    + Gemini Flash: google/gemini-2.0-flash
    + Llama 3.1: meta-llama/llama-3.1-8b-instruct
    + DeepSeek: deepseek-v3 / deepseek-v4
=========================================================
"""


# =========================================================
# 1. SUPERVISOR–WORKER ARCHITECTURE
# =========================================================

ARCHITECTURE = """
User Query
    ↓
Supervisor (LLM via OpenRouter)
    ↓
Intent Classification:
    - curriculum
    - regulations
    - general

Routing:
    curriculum  → Curriculum Worker → FAISS (curriculum index)
    regulations  → Regulation Worker → FAISS (regulation index)
    general      → Direct LLM Answer (no retrieval)
"""


# =========================================================
# 2. RAG PIPELINE FLOW
# =========================================================

RAG_FLOW = """
Query
  ↓
generate_query (OpenRouter LLM)
  ↓
FAISS Retrieval (top_k × multiplier)
  ↓
Reranker (Cohere / CrossEncoder / LLM / None)
  ↓
Grade Documents (LLM judge)
  ↓
Answer Generation (OpenRouter LLM)
  ↓
Final Response
"""


# =========================================================
# 3. ENV CONFIG (IMPORTANT)
# =========================================================

ENV_CONFIG = """
OPENROUTER_API_KEY=your_key_here

# Chat model (change here)
OPENROUTER_CHAT_MODEL=google/gemini-2.0-flash

# Embedding model
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small

# Reranker
RERANKER_TYPE=cohere
"""


# =========================================================
# 4. SUPERVISOR LOGIC
# =========================================================

def route_query(query_type: str) -> str:
    """
    Route user query to correct worker.
    """
    if query_type == "curriculum":
        return "curriculum_worker"
    elif query_type == "regulations":
        return "regulation_worker"
    else:
        return "general_llm"


# =========================================================
# 5. VECTOR STORE (FAISS)
# =========================================================

class VectorStore:
    """
    FAISS-based retrieval system
    """

    def retrieve(self, query: str, top_k: int):
        """
        Steps:
        1. Embed query via OpenRouter embedding model
        2. Normalize vector (L2)
        3. FAISS similarity search
        4. Return ranked chunks
        """
        pass


# Singleton stores
curriculum_store = VectorStore()
regulation_store = VectorStore()


# =========================================================
# 6. RERANKER SYSTEM
# =========================================================

class BaseReranker:
    def rerank(self, query, chunks, top_k):
        pass


class IdentityReranker(BaseReranker):
    """No reranking"""


class CohereReranker(BaseReranker):
    """Cohere API reranker"""


class CrossEncoderReranker(BaseReranker):
    """Local transformer reranker"""


class LLMReranker(BaseReranker):
    """LLM-based reranker"""


def get_reranker():
    """
    Return reranker based on:
        RERANKER_TYPE env variable
    """
    pass


# IMPORTANT RULE:
# ALWAYS IMPORT LIKE THIS:
#     import src.service.reranker as reranker_module
#
# NEVER:
#     from src.service.reranker import reranker


# =========================================================
# 7. OPENROUTER LLM CLIENT
# =========================================================

def get_openrouter_client():
    """
    Create OpenRouter client (OpenAI-compatible API)
    """
    pass


def get_chat_model():
    """
    Chat model wrapper using OpenRouter.

    Supports:
        - Gemini Flash
        - Llama 3.1
        - DeepSeek
    """
    pass


# =========================================================
# 8. LANGGRAPH WORKFLOW
# =========================================================

def build_graph():
    """
    Build LangGraph pipeline:
    supervisor → worker → retrieval → rerank → answer
    """
    pass


def generate_node():
    pass


def retrieve_node():
    pass


def grade_node():
    pass


def rewrite_node():
    pass


def answer_node():
    pass


def general_node():
    pass


# =========================================================
# 9. AGENT (STATEFUL CHAT WRAPPER)
# =========================================================

class HAUIAgent:
    """
    Stateful RAG agent using LangGraph
    """

    def __init__(self):
        self.history = []

    def chat(self, query: str):
        """
        Flow:
        1. Add user message
        2. Run graph
        3. Extract final answer
        4. Detect intent
        5. Return response
        """
        pass

    def reset(self):
        """
        Clear chat history
        """
        self.history = []


# =========================================================
# 10. INTENT TYPES
# =========================================================

INTENTS = [
    "curriculum",
    "regulations",
    "general",
    "related",
    "unrelated"
]


# =========================================================
# 11. RAG PIPELINE SUMMARY
# =========================================================

PIPELINE = """
User Query
    ↓
Supervisor (Gemini Flash / Llama / DeepSeek)
    ↓
FAISS Retrieval
    ↓
Reranker (optional)
    ↓
LLM Answer Generation
    ↓
Final Response
"""


# =========================================================
# END OF SYSTEM
# =========================================================