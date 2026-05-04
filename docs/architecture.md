# Kiến trúc hệ thống

## Tổng quan

HAUI Agent sử dụng mô hình **Supervisor-Worker** với LangGraph. Mỗi câu hỏi được phân loại bởi một Supervisor, sau đó chuyển đến Worker chuyên biệt để truy xuất và trả lời.

```
User Query
    │
    ▼
┌─────────────────┐
│  Supervisor     │  ← LLM classifier (phân loại ý định)
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

## Luồng xử lý RAG Worker

Mỗi RAG worker (curriculum / regulation) thực hiện quy trình sau:

```
generate_query
    │
    ├─ (không gọi tool) ──→ answer  ──→  END
    │
    └─ tool_calls present
         │
         ▼
    retrieve_<domain>   ← gọi FAISS search
         │
         ▼
    grade_documents     ← kiểm tra độ liên quan
         │
         ├─ "yes" ──→  answer  ──→  END
         │
         └─ "no"  ──→  rewrite_question  ──→  generate_query (retry)
```

## Các Layer kiến trúc

### 1. Presentation Layer

| File | Vai trò |
|------|---------|
| `app.py` | Streamlit web UI — hiển thị chat, badge intent, trace nodes, sources |
| `main.py` | CLI tương tác — vòng lặp nhập/xuất đơn giản |

### 2. Agent Layer (`src/agent/`)

| File | Vai trò |
|------|---------|
| `haui_agent.py` | `HAUIAgent` — wrapper stateful, quản lý conversation history |
| `graph.py` | `build_graph()` — xây dựng LangGraph `StateGraph` |
| `state.py` | `AgentState` — TypedDict chứa messages và query_type |
| `supervisor.py` | Phân loại ý định, routing đến worker phù hợp |
| `nodes.py` | Factory functions tạo các node: generate, grade, answer, rewrite |
| `tools.py` | `@tool` functions cho FAISS retrieval |

### 3. Service Layer (`src/service/`)

| File | Vai trò |
|------|---------|
| `vectorstore.py` | `VectorStore` — lazy-loaded FAISS wrapper |

### 4. LLM Layer (`src/llm/`)

| File | Vai trò |
|------|---------|
| `client.py` | `get_client()` (embeddings), `get_chat_model()` (chat) |

### 5. Configuration Layer

| File | Vai trò |
|------|---------|
| `src/config.py` | `Settings` — Pydantic BaseSettings đọc từ `.env` |
| `src/models.py` | Pydantic schemas: `ChatMessage`, `RetrievedChunk`, `AgentResponse` |

## Agent State

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Toàn bộ lịch sử hội thoại — append-only nhờ add_messages reducer

    query_type: str
    # "curriculum" | "regulations" | "general"
    # Được đặt bởi supervisor, dùng để routing
```

## Supervisor Node

**Input:** câu hỏi của user  
**Output:** `{"query_type": "curriculum" | "regulations" | "general"}`

Supervisor gọi LLM với một system prompt tiếng Việt yêu cầu trả về đúng một từ. Nếu kết quả không khớp với ba loại hợp lệ, fallback về `"general"`.

**Routing logic:**

```python
def route_to_agent(state) -> str:
    match state["query_type"]:
        case "curriculum":   return "curriculum_generate"
        case "regulations":  return "regulation_generate"
        case _:              return "general_respond"
```

## Worker Nodes (Node Factories)

Các node của hai worker (curriculum và regulation) được tạo bằng factory functions để tránh trùng lặp code:

### `make_generate_node(tool, system_prompt, node_name)`

- Bind retrieval tool vào chat model
- Prepend system prompt vào messages
- Gọi LLM → trả về AI message (có thể có `tool_calls`)

### `make_grade_edge(node_name)` — Conditional Edge

- Nhận retrieved chunks từ tool message
- Gọi LLM với `GRADE_PROMPT`, yêu cầu structured output `GradeDocuments(binary_score)`
- `"yes"` → chuyển đến answer node
- `"no"` → chuyển đến rewrite node

### `make_answer_node(node_name)`

- Extract câu hỏi gốc và retrieved context từ messages
- Gọi LLM với `GENERATE_PROMPT` để tổng hợp câu trả lời
- Trả về AI message với nội dung grounded vào sources

### `make_rewrite_node(node_name)`

- Gọi LLM với `REWRITE_PROMPT` để cải thiện câu hỏi
- Trả về `HumanMessage` mới để restart vòng lặp retrieval

### `general_respond_node(state)`

- Gọi LLM với `GENERAL_SYSTEM` prompt
- Không truy xuất, trả lời trực tiếp hoặc chuyển hướng

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

**Intent classification trong response:**
- `"curriculum"` / `"regulations"` / `"general"` — từ `query_type` do supervisor đặt
- `"related"` — nếu có tool call nhưng không có `query_type`
- `"unrelated"` — nếu không có retrieval nào xảy ra

## Vector Store

```python
class VectorStore:
    def __init__(self, index_file: str, metadata_file: str):
        # Lazy init — chỉ load khi retrieve() được gọi lần đầu

    def retrieve(query: str, top_k: int = None) -> list[RetrievedChunk]:
        # 1. Embed query via OpenAI text-embedding-3-small
        # 2. L2-normalize embedding vector
        # 3. FAISS.IndexFlatIP.search(embedding, top_k)
        # 4. Map results → list[RetrievedChunk]
```

**Index type:** `FAISS.IndexFlatIP` với L2-normalized embeddings = cosine similarity  
**Dimension:** 1536 (text-embedding-3-small)

Hai singleton instances tại module level:
```python
curriculum_store = VectorStore("faiss_curriculum.bin", "faiss_curriculum_meta.pkl")
regulation_store = VectorStore("faiss_regulation.bin", "faiss_regulation_meta.pkl")
```
