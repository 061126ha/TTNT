# System Architecture

## Overview

HAUI Agent uses a **Supervisor-Worker** pattern with LangGraph. Each query is classified by a Supervisor, then routed to a specialized Worker for retrieval and response generation.

```
User Query
    │
    ▼
┌─────────────────┐
│  Supervisor     │  ← LLM classifier (classifies intent)
└─────────────────┘
    │
    ├─ "curriculum"  ──→  Curriculum Worker  ──→  FAISS curriculum index
    ├─ "regulations" ──→  Regulation Worker  ──→  FAISS regulation index
    └─ "general"     ──→  General Fallback   ──→  Direct LLM response
                                │
                                ▼
                         AgentResponse
                    (answer + intent + sources)
```

## RAG Worker Processing Flow

Each RAG worker (curriculum / regulation) follows this pipeline:

```
generate_query
    │
    ├─ (no tool call) ──→ answer  ──→  END
    │
    └─ tool_calls present
         │
         ▼
    retrieve_<domain>   ← FAISS search (top_k × RERANK_FETCH_MULTIPLIER candidates)
         │
         ▼
    rerank_chunks       ← Cross-encoder / LLM / Cohere → final top_k chunks
         │                 (identity pass-through when RERANKER_TYPE=none)
         ▼
    grade_documents     ← relevance check
         │
         ├─ "yes" ──→  answer  ──→  END
         │
         └─ "no"  ──→  rewrite_question  ──→  generate_query (retry)
```

## Architecture Layers

### 1. Presentation Layer

| File | Role |
|------|------|
| `app.py` | Streamlit web UI — chat display, intent badge, node trace, sources |
| `main.py` | Interactive CLI — simple input/output loop |

### 2. Agent Layer (`src/agent/`)

| File | Role |
|------|------|
| `haui_agent.py` | `HAUIAgent` — stateful wrapper, manages conversation history |
| `graph.py` | `build_graph()` — builds the LangGraph `StateGraph` |
| `state.py` | `AgentState` — TypedDict holding messages and query_type |
| `supervisor.py` | Classifies intent and routes to the appropriate worker |
| `nodes.py` | Factory functions creating nodes: generate, grade, answer, rewrite |
| `tools.py` | `@tool` functions for FAISS retrieval |

### 3. Service Layer (`src/service/`)

| File | Role |
|------|------|
| `vectorstore.py` | `VectorStore` — lazy-loaded FAISS wrapper |

### 4. LLM Layer (`src/llm/`)

| File | Role |
|------|------|
| `client.py` | `get_client()` (embeddings), `get_chat_model()` (chat) |

### 5. Configuration Layer

| File | Role |
|------|------|
| `src/config.py` | `Settings` — Pydantic BaseSettings reading from `.env` |
| `src/models.py` | Pydantic schemas: `ChatMessage`, `RetrievedChunk`, `AgentResponse` |

## Agent State

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Full conversation history — append-only via add_messages reducer

    query_type: str
    # "curriculum" | "regulations" | "general"
    # Set by the supervisor, used for routing
```

## Supervisor Node

**Input:** user query  
**Output:** `{"query_type": "curriculum" | "regulations" | "general"}`

The Supervisor calls an LLM with a system prompt instructing it to return exactly one word. If the result does not match one of the three valid types, it falls back to `"general"`.

**Routing logic:**

```python
def route_to_agent(state) -> str:
    match state["query_type"]:
        case "curriculum":   return "curriculum_generate"
        case "regulations":  return "regulation_generate"
        case _:              return "general_respond"
```

## Worker Nodes (Node Factories)

Nodes for both workers (curriculum and regulation) are created via factory functions to avoid code duplication:

### `make_generate_node(tool, system_prompt, node_name)`

- Binds the retrieval tool to the chat model
- Prepends the system prompt to messages
- Calls the LLM → returns an AI message (may contain `tool_calls`)

### `make_grade_edge(node_name)` — Conditional Edge

- Receives retrieved chunks from the tool message
- Calls the LLM with `GRADE_PROMPT`, requesting structured output `GradeDocuments(binary_score)`
- `"yes"` → routes to the answer node
- `"no"` → routes to the rewrite node

### `make_answer_node(node_name)`

- Extracts the original question and retrieved context from messages
- Calls the LLM with `GENERATE_PROMPT` to synthesize the answer
- Returns an AI message grounded in the retrieved sources

### `make_rewrite_node(node_name)`

- Calls the LLM with `REWRITE_PROMPT` to improve the query
- Returns a new `HumanMessage` to restart the retrieval loop

### `general_respond_node(state)`

- Calls the LLM with `GENERAL_SYSTEM` prompt
- No retrieval; responds directly or redirects the user

## LangGraph Structure

```
START
  │
  └─ supervisor_node
        │
        ├─ curriculum_generate ──→ [tool_call?]
        │       │                       │
        │       │                  curriculum_retrieve
        │       │                       │
        │       │                  curriculum_grade
        │       │                    /         \
        │       │              (yes)             (no)
        │       │                │                │
        │       │          curriculum_answer  curriculum_rewrite
        │       │                │                │
        │       │               END        curriculum_generate (loop)
        │
        ├─ regulation_generate ──→ [same pattern as above]
        │
        └─ general_respond ──→ END
```

## HAUIAgent — Stateful Chat Wrapper

```python
class HAUIAgent:
    graph: CompiledGraph    # LangGraph compiled graph
    _messages: list         # Persistent conversation history

    def chat(query: str) -> AgentResponse:
        # 1. Append HumanMessage
        # 2. graph.invoke({"messages": self._messages, "query_type": ""})
        # 3. Extract final AIMessage as answer
        # 4. Determine intent (query_type, tool usage, or "unrelated")
        # 5. Return AgentResponse(answer, intent, sources)

    def reset():
        self._messages = []
```

**Intent classification in response:**
- `"curriculum"` / `"regulations"` / `"general"` — from `query_type` set by the supervisor
- `"related"` — if a tool call occurred but no `query_type` was set
- `"unrelated"` — if no retrieval occurred

## Vector Store

```python
class VectorStore:
    def __init__(self, index_file: str, metadata_file: str):
        # Lazy init — only loads on first retrieve() call

    def retrieve(query: str, top_k: int = None) -> list[RetrievedChunk]:
        # 1. Embed query via OpenAI text-embedding-3-small
        # 2. L2-normalize the embedding vector
        # 3. FAISS.IndexFlatIP.search(embedding, top_k)
        # 4. Map results → list[RetrievedChunk]
```

**Index type:** `FAISS.IndexFlatIP` with L2-normalized embeddings = cosine similarity  
**Dimension:** 1536 (text-embedding-3-small)

Two singleton instances at module level:
```python
curriculum_store = VectorStore("faiss_curriculum.bin", "faiss_curriculum_meta.pkl")
regulation_store = VectorStore("faiss_regulation.bin", "faiss_regulation_meta.pkl")
```
