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

A chunk retrieved from the FAISS index, optionally annotated with a rerank score.

```python
class RetrievedChunk(BaseModel):
    chunk_id: int
    content: str
    section: str
    subsection: str
    score: float             # cosine similarity from FAISS (0.0 – 1.0)
    text: str                # raw text (may differ from content)
    rerank_score: float | None = None  # score assigned by the reranker; None if not reranked
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

## Reranker Service (`src/service/reranker.py`)

### Abstract Base

```python
class BaseReranker(ABC):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...
```

### Implementations

| Class | `RERANKER_TYPE` | Description |
|-------|-----------------|-------------|
| `IdentityReranker` | `none` | Returns first `top_k` chunks unchanged |
| `CrossEncoderReranker` | `cross_encoder` | Local cross-encoder via `sentence-transformers`; lazy-loads model on first call |
| `LLMReranker` | `llm` | Scores each chunk with one LLM call; no extra dependencies |
| `CohereReranker` | `cohere` | Cohere Rerank API; requires `COHERE_API_KEY` |

### `get_reranker() -> BaseReranker`

Factory that reads `settings.reranker_type` and returns the appropriate instance. Raises `ValueError` if `RERANKER_TYPE=cohere` and `COHERE_API_KEY` is unset.

### Module-level singleton

```python
from src.service import reranker as _reranker_module

# Always use module-attribute access — direct import creates a stale binding
_reranker_module.reranker.rerank(query, chunks, top_k)
```

The `reranker` module attribute is replaced at runtime when the Streamlit sidebar changes the reranker type. If initialization fails (e.g. missing `COHERE_API_KEY`), the module falls back to `IdentityReranker` and logs a warning.

### `rerank_debug` (`src/agent/tools.py`)

```python
rerank_debug: list[dict]
```

A module-level list appended to by each tool call. The Streamlit UI calls `rerank_debug.clear()` before each new user turn and reads it after to display the **Rerank info** expander. Each entry contains:

| Key | Type | Description |
|-----|------|-------------|
| `tool` | `str` | `"curriculum"` or `"regulations"` |
| `reranker_type` | `str` | Active `RERANKER_TYPE` |
| `reranker_model` | `str \| None` | Model name (Cohere or cross-encoder); `None` for `llm`/`none` |
| `fetch_k` | `int` | Candidates fetched from FAISS |
| `retrieved` | `int` | Candidates actually returned by FAISS |
| `reranked` | `int` | Chunks after reranking |
| `effective_top_k` | `int` | `min(rerank_top_k, fetch_k)` — actual cap applied |
| `chunks` | `list[RetrievedChunk]` | Final ranked chunks |

---

## LangGraph Tools (`src/agent/tools.py`)

### `retrieve_curriculum(query: str) -> str`

Tool bound to the curriculum worker. Fetches `top_k × RERANK_FETCH_MULTIPLIER` candidates from `curriculum_store`, reranks them, then formats the top `RERANK_TOP_K` results.

**Output format:**
```
[1] Training > Software Engineering
The software engineering program...

---

[2] Admissions > Quotas
In 2024, the enrollment quota...
```

### `retrieve_regulations(query: str) -> str`

Tool bound to the regulation worker. Same pipeline as `retrieve_curriculum` but uses `regulation_store`.

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
settings.top_k                    # int — FAISS retrieval top-k
settings.curriculum_index_file    # str — path to FAISS index
settings.curriculum_meta_file     # str — path to metadata pickle
settings.regulation_index_file    # str
settings.regulation_meta_file     # str

# Reranker properties (all read @property from os.environ at access time)
settings.reranker_type            # str — "cohere" | "cross_encoder" | "llm" | "none"
settings.rerank_top_k             # int — chunks to keep after reranking
settings.rerank_fetch_multiplier  # int — FAISS over-fetch multiplier
settings.cohere_api_key           # str | None
settings.cohere_rerank_model      # str — default "rerank-multilingual-v3.0"
settings.cross_encoder_model      # str — HuggingFace model name
```

All properties are `@property` decorators that read from `os.environ` at access time, so env var changes (e.g. from the Streamlit sidebar) take effect immediately without restarting the application.

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
