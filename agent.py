"""
HAUI Student Handbook Chatbot Agent
────────────────────────────────────
Architecture (LangGraph):

    User Query
        │
        ▼
    [router]  ──── unrelated ────► [off_topic]  ──► Response
        │
      related
        │
        ▼
    [retrieve]  ──► [generate]  ──► Response

Nodes:
  • router    – classifies whether the question is about the student handbook
  • retrieve  – FAISS similarity search → top-K chunks
  • generate  – OpenRouter chat with retrieved context
  • off_topic – polite decline / redirect for unrelated questions
"""

from __future__ import annotations

import os
import pickle
from typing import Annotated, TypedDict

import faiss
import numpy as np
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import OpenAI

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "google/gemini-2.0-flash-001")
EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "faiss_index.bin")
METADATA_FILE = os.getenv("FAISS_METADATA_FILE", "faiss_metadata.pkl")

# ─── Singleton resources (loaded once) ────────────────────────────────────────

_client: OpenAI | None = None
_index: faiss.Index | None = None
_metadata: list[dict] | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    return _client


def _load_vectorstore():
    global _index, _metadata
    if _index is None:
        if not os.path.exists(INDEX_FILE):
            raise FileNotFoundError(
                f"FAISS index '{INDEX_FILE}' not found. "
                "Run `python build_vectorstore.py` first."
            )
        _index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            _metadata = pickle.load(f)


# ─── Agent State ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    intent: str                   # "related" | "unrelated"
    retrieved_chunks: list[dict]
    messages: Annotated[list[dict], "chat message history"]
    answer: str


# ─── Node: router ─────────────────────────────────────────────────────────────

ROUTER_SYSTEM = """You are an intent classifier for a university student-handbook chatbot.

The chatbot ONLY answers questions about the HAUI (Hanoi University of Industry) student handbook which covers:
- Academic programs and majors (Kỹ thuật phần mềm, Khoa học máy tính, Hệ thống thông tin, etc.)
- Programme objectives (PEO), learning outcomes (SO), and performance indicators (PI)
- Curriculum frameworks and course lists
- Student admission and graduation statistics
- Rules, regulations, and student affairs

Classify the user question as:
  "related"   – if it is about any of the above topics
  "unrelated" – if it is about something else (general knowledge, weather, coding help, etc.)

Reply with ONLY one word: related  OR  unrelated"""


def router_node(state: AgentState) -> AgentState:
    client = _get_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": state["query"]},
        ],
        temperature=0,
        max_tokens=10,
    )
    verdict = response.choices[0].message.content.strip().lower()
    intent = "related" if verdict == "related" else "unrelated"
    return {**state, "intent": intent}


# ─── Node: retrieve ───────────────────────────────────────────────────────────

def retrieve_node(state: AgentState) -> AgentState:
    _load_vectorstore()
    client = _get_client()

    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=state["query"])
    query_vec = np.array([resp.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(query_vec)

    scores, indices = _index.search(query_vec, TOP_K)
    chunks = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = dict(_metadata[idx])
        chunk["score"] = float(score)
        chunks.append(chunk)

    return {**state, "retrieved_chunks": chunks}


# ─── Node: generate ───────────────────────────────────────────────────────────

GENERATE_SYSTEM = """You are a helpful assistant for students at HAUI (Hanoi University of Industry - Đại học Công nghiệp Hà Nội).
Answer the user's question using ONLY the provided context from the student handbook.
If the context does not contain enough information to answer, say so honestly.
Answer in the same language as the user's question (Vietnamese or English).
Be concise, accurate, and friendly."""


def generate_node(state: AgentState) -> AgentState:
    client = _get_client()

    context_parts = []
    for i, chunk in enumerate(state["retrieved_chunks"], 1):
        header = " > ".join(filter(None, [chunk.get("section", ""), chunk.get("subsection", "")]))
        context_parts.append(f"[{i}] {header}\n{chunk['content']}")
    context = "\n\n---\n\n".join(context_parts)

    user_message = (
        f"Context from the student handbook:\n\n{context}\n\n"
        f"Question: {state['query']}"
    )

    history = state.get("messages", [])
    messages = (
        [{"role": "system", "content": GENERATE_SYSTEM}]
        + history
        + [{"role": "user", "content": user_message}]
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
    )
    answer = response.choices[0].message.content

    # Store original query (without injected context) in history
    new_history = history + [
        {"role": "user", "content": state["query"]},
        {"role": "assistant", "content": answer},
    ]

    return {**state, "answer": answer, "messages": new_history}


# ─── Node: off_topic ──────────────────────────────────────────────────────────

OFF_TOPIC_SYSTEM = """You are a helpful assistant for students at HAUI (Hanoi University of Industry).
You only have knowledge about the HAUI student handbook.
Politely inform the user that you can only answer questions related to the student handbook
and suggest they rephrase if their question might actually be handbook-related.
Answer in the same language as the user's question."""


def off_topic_node(state: AgentState) -> AgentState:
    client = _get_client()
    history = state.get("messages", [])

    messages = (
        [{"role": "system", "content": OFF_TOPIC_SYSTEM}]
        + history
        + [{"role": "user", "content": state["query"]}]
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=256,
    )
    answer = response.choices[0].message.content

    new_history = history + [
        {"role": "user", "content": state["query"]},
        {"role": "assistant", "content": answer},
    ]

    return {**state, "answer": answer, "messages": new_history}


# ─── Routing function ─────────────────────────────────────────────────────────

def route_by_intent(state: AgentState) -> str:
    return state["intent"]  # "related" | "unrelated"


# ─── Build the graph ──────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("off_topic", off_topic_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "related": "retrieve",
            "unrelated": "off_topic",
        },
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("off_topic", END)

    return graph.compile()


# ─── Public API ───────────────────────────────────────────────────────────────

class HAUIAgent:
    """Stateful chatbot agent. Maintains conversation history across turns."""

    def __init__(self):
        self.graph = build_graph()
        self._messages: list[dict] = []

    def chat(self, query: str) -> dict:
        """
        Send a message and return a result dict with keys:
          - answer  : str
          - intent  : "related" | "unrelated"
          - sources : list of retrieved chunk metadata (empty for off-topic)
        """
        state: AgentState = {
            "query": query,
            "intent": "",
            "retrieved_chunks": [],
            "messages": self._messages,
            "answer": "",
        }

        result = self.graph.invoke(state)
        self._messages = result["messages"]

        sources = [
            {
                "chunk_id": c["chunk_id"],
                "section": c.get("section", ""),
                "subsection": c.get("subsection", ""),
                "score": round(c.get("score", 0), 4),
            }
            for c in result.get("retrieved_chunks", [])
        ]

        return {
            "answer": result["answer"],
            "intent": result["intent"],
            "sources": sources,
        }

    def reset(self):
        """Clear conversation history."""
        self._messages = []