# API Reference

## `HAUIAgent` (`src/agent/haui_agent.py`)

Wrapper stateful cho LangGraph graph, quản lý conversation history qua nhiều lượt chat.

### Constructor

```python
agent = HAUIAgent()
```

Khởi tạo graph và message history rỗng.

### Methods

#### `chat(query: str) -> AgentResponse`

Gửi câu hỏi và nhận câu trả lời.

```python
response = agent.chat("Ngành Kỹ thuật phần mềm có những môn học gì?")

print(response.answer)   # str — câu trả lời
print(response.intent)   # str — "curriculum" | "regulations" | "general" | "related" | "unrelated"
print(response.sources)  # list[RetrievedChunk] — chunks được sử dụng
```

**Intent values:**

| Intent | Ý nghĩa |
|--------|---------|
| `"curriculum"` | Câu hỏi về chương trình đào tạo |
| `"regulations"` | Câu hỏi về quy định, thông tin nhà trường |
| `"general"` | Câu hỏi chung, không thuộc hai domain trên |
| `"related"` | Có retrieval xảy ra nhưng không có query_type |
| `"unrelated"` | Không có retrieval nào xảy ra |

#### `reset() -> None`

Xóa toàn bộ lịch sử hội thoại.

```python
agent.reset()
```

---

## Pydantic Schemas (`src/models.py`)

### `ChatMessage`

```python
class ChatMessage(BaseModel):
    role: str     # "user" | "assistant"
    content: str
```

### `RetrievedChunk`

Một chunk được truy xuất từ FAISS index.

```python
class RetrievedChunk(BaseModel):
    chunk_id: int
    content: str
    section: str
    subsection: str
    score: float    # cosine similarity score (0.0 – 1.0)
```

### `AgentResponse`

Kết quả trả về từ `HAUIAgent.chat()`.

```python
class AgentResponse(BaseModel):
    answer: str
    intent: str
    sources: list[RetrievedChunk]
```

---

## `VectorStore` (`src/service/vectorstore.py`)

### Constructor

```python
store = VectorStore(
    index_file="faiss_curriculum.bin",
    metadata_file="faiss_curriculum_meta.pkl"
)
```

### Methods

#### `retrieve(query: str, top_k: int = None) -> list[RetrievedChunk]`

Tìm kiếm chunks gần nhất với query.

```python
chunks = store.retrieve("chương trình kỹ thuật phần mềm", top_k=5)
for chunk in chunks:
    print(f"[{chunk.score:.3f}] {chunk.section} > {chunk.subsection}")
    print(chunk.content[:200])
```

**Nếu `top_k=None`**, sử dụng `settings.top_k`.

### Properties

#### `is_ready -> bool`

Kiểm tra cả hai file index và metadata đều tồn tại.

```python
if store.is_ready:
    chunks = store.retrieve(query)
```

### Singleton instances

```python
from src.service.vectorstore import curriculum_store, regulation_store
```

---

## LangGraph Tools (`src/agent/tools.py`)

### `retrieve_curriculum(query: str) -> str`

Tool được bind vào curriculum worker. Gọi `curriculum_store.retrieve()` và format kết quả.

**Output format:**
```
[1] Đào tạo > Kỹ thuật phần mềm
Chương trình đào tạo kỹ sư...

---

[2] Tuyển sinh > Chỉ tiêu
Năm 2024, chỉ tiêu tuyển sinh...
```

### `retrieve_regulations(query: str) -> str`

Tool được bind vào regulation worker. Gọi `regulation_store.retrieve()` và format tương tự.

---

## LLM Client (`src/llm/client.py`)

### `get_client() -> OpenAI`

Trả về OpenAI client (dùng cho embedding requests).

```python
from src.llm.client import get_client

client = get_client()
response = client.embeddings.create(
    model=settings.embedding_model,
    input="văn bản cần embed"
)
```

### `get_chat_model() -> ChatOpenAI`

Trả về LangChain `ChatOpenAI` instance.

```python
from src.llm.client import get_chat_model

llm = get_chat_model()
# llm.temperature = 0 (deterministic)
# Supports .bind_tools(), .with_structured_output()
```

---

## `build_graph()` (`src/agent/graph.py`)

Xây dựng và compile LangGraph `StateGraph`.

```python
from src.agent.graph import build_graph

graph = build_graph()
result = graph.invoke({
    "messages": [HumanMessage(content="Hỏi gì đó")],
    "query_type": ""
})
```

### `_add_rag_worker(builder, domain, tool, system_prompt)`

Internal helper — thêm toàn bộ nodes và edges của một RAG worker vào graph builder.

**Nodes được tạo ra:**
- `{domain}_generate`
- `{domain}_retrieve`
- `{domain}_grade` (conditional edge)
- `{domain}_answer`
- `{domain}_rewrite`

---

## `Settings` (`src/config.py`)

```python
from src.config import settings

settings.openrouter_base_url      # str
settings.openrouter_api_key       # str
settings.chat_model               # str — model name
settings.embedding_model          # str — embedding model name
settings.top_k                    # int — retrieval top-k
settings.curriculum_index_file    # str — path to FAISS index
settings.curriculum_meta_file     # str — path to metadata pickle
settings.regulation_index_file    # str
settings.regulation_meta_file     # str
```

---

## Node Factory Functions (`src/agent/nodes.py`)

### `make_generate_node(tool, system_prompt, node_name)`

```python
node_fn = make_generate_node(
    tool=retrieve_curriculum,
    system_prompt="Bạn là trợ lý...",
    node_name="curriculum_generate"
)
# node_fn(state: AgentState) -> dict
```

### `make_grade_edge(node_name)`

Trả về conditional routing function, không phải node thông thường.

```python
edge_fn = make_grade_edge("curriculum")
# edge_fn(state: AgentState) -> "generate_answer" | "rewrite_question"
```

### `make_answer_node(node_name)`

```python
node_fn = make_answer_node("curriculum")
# node_fn(state: AgentState) -> {"messages": [AIMessage(...)]}
```

### `make_rewrite_node(node_name)`

```python
node_fn = make_rewrite_node("curriculum")
# node_fn(state: AgentState) -> {"messages": [HumanMessage(...)]}
```

### `general_respond_node(state: AgentState) -> dict`

Direct response node, không có factory wrapper.
