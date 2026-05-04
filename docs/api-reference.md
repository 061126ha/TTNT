# API Reference

## `HAUIAgent` (`src/agent/haui_agent.py`)

Stateful wrapper for the LangGraph graph, managing conversation history across multiple turns.

### Constructor

```python
agent = HAUIAgent()
```

Initializes the graph and an empty message history.

### Methods

#### `chat(query: str) -> AgentResponse`

Send a query and receive a response.

```python
response = agent.chat("What courses are in the Software Engineering program?")

print(response.answer)   # str — the answer
print(response.intent)   # str — "curriculum" | "regulations" | "general" | "related" | "unrelated"
print(response.sources)  # list[RetrievedChunk] — chunks used to generate the answer
```

**Intent values:**

| Intent | Meaning |
|--------|---------|
| `"curriculum"` | Query about academic programs |
| `"regulations"` | Query about university regulations or information |
| `"general"` | General query not belonging to either domain |
| `"related"` | Retrieval occurred but no query_type was set |
| `"unrelated"` | No retrieval occurred |

#### `reset() -> None`

Clear the entire conversation history.

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

A chunk retrieved from the FAISS index.

```python
class RetrievedChunk(BaseModel):
    chunk_id: int
    content: str
    section: str
    subsection: str
    score: float    # cosine similarity score (0.0 – 1.0)
```

### `AgentResponse`

The result returned by `HAUIAgent.chat()`.

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

Find the nearest chunks to the query.

```python
chunks = store.retrieve("software engineering program", top_k=5)
for chunk in chunks:
    print(f"[{chunk.score:.3f}] {chunk.section} > {chunk.subsection}")
    print(chunk.content[:200])
```

If `top_k=None`, uses `settings.top_k`.

### Properties

#### `is_ready -> bool`

Check that both the index file and metadata file exist.

```python
if store.is_ready:
    chunks = store.retrieve(query)
```

### Singleton Instances

```python
from src.service.vectorstore import curriculum_store, regulation_store
```

---

## LangGraph Tools (`src/agent/tools.py`)

### `retrieve_curriculum(query: str) -> str`

Tool bound to the curriculum worker. Calls `curriculum_store.retrieve()` and formats results.

**Output format:**
```
[1] Training > Software Engineering
The software engineering program...

---

[2] Admissions > Quotas
In 2024, the enrollment quota...
```

### `retrieve_regulations(query: str) -> str`

Tool bound to the regulation worker. Calls `regulation_store.retrieve()` and formats results the same way.

---

## LLM Client (`src/llm/client.py`)

### `get_client() -> OpenAI`

Returns an OpenAI client (used for embedding requests).

```python
from src.llm.client import get_client

client = get_client()
response = client.embeddings.create(
    model=settings.embedding_model,
    input="text to embed"
)
```

### `get_chat_model() -> ChatOpenAI`

Returns a LangChain `ChatOpenAI` instance.

```python
from src.llm.client import get_chat_model

llm = get_chat_model()
# llm.temperature = 0 (deterministic)
# Supports .bind_tools(), .with_structured_output()
```

---

## `build_graph()` (`src/agent/graph.py`)

Builds and compiles the LangGraph `StateGraph`.

```python
from src.agent.graph import build_graph

graph = build_graph()
result = graph.invoke({
    "messages": [HumanMessage(content="Ask something")],
    "query_type": ""
})
```

### `_add_rag_worker(builder, domain, tool, system_prompt)`

Internal helper — adds all nodes and edges for a RAG worker to the graph builder.

**Nodes created:**
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
    system_prompt="You are an assistant...",
    node_name="curriculum_generate"
)
# node_fn(state: AgentState) -> dict
```

### `make_grade_edge(node_name)`

Returns a conditional routing function, not a regular node.

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

Direct response node with no factory wrapper.
