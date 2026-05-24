# Development Guide

## Environment Setup

```bash
# Clone and install
git clone <repo-url>
cd Haui_Agent
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Fill in OPENROUTER_API_KEY

# Ensure FAISS indexes are built
python scripts/build_index.py
```

## Design Principles

This project follows **OOP** and **SOLID** principles:

- **S** — Each class/module has a single responsibility
- **O** — Extend by adding subclasses, never modify base classes
- **L** — Every implementation must be a drop-in replacement for its abstraction
- **I** — Keep interfaces minimal; don't bloat them with methods only some subclasses need
- **D** — Depend on abstractions, not concrete classes

## Adding a New RAG Domain

Example: adding a "research" domain for scientific research information.

### 1. Prepare Data

```bash
# Create JSON chunks for the new domain
# data/research_chunks.json

# Build FAISS index
python scripts/build_index.py --input data/research_chunks.json \
    --output faiss_research.bin --meta faiss_research_meta.pkl
```

### 2. Update VectorStore (`src/service/vectorstore.py`)

```python
research_store = VectorStore("faiss_research.bin", "faiss_research_meta.pkl")
```

### 3. Add Retrieval Tool (`src/agent/tools.py`)

```python
import src.service.reranker as _reranker_module
from src.config import settings

@tool
def retrieve_research(query: str) -> str:
    """Find information about scientific research at SICT HAUI."""
    fetch_k = settings.top_k * settings.rerank_fetch_multiplier if settings.reranker_type != "none" else None
    raw_chunks = research_store.retrieve(query, top_k=fetch_k)
    effective_top_k = min(settings.rerank_top_k, fetch_k) if fetch_k is not None else settings.rerank_top_k
    ranked_chunks = _reranker_module.reranker.rerank(query, raw_chunks, top_k=effective_top_k)
    return _format_chunks(ranked_chunks)
```

> **Important:** Always import the reranker as a module (`import src.service.reranker as _reranker_module`) and access the singleton via `_reranker_module.reranker`. A direct `from ... import reranker` creates a stale local binding that will not reflect live reranker changes from the Streamlit sidebar.

### 4. Update Supervisor (`src/agent/supervisor.py`)

Add `"research"` to the list of valid types and update routing:

```python
VALID_TYPES = {"curriculum", "regulations", "general", "research"}

def route_to_agent(state: AgentState) -> str:
    match state["query_type"]:
        case "curriculum":  return "curriculum_generate"
        case "regulations": return "regulation_generate"
        case "research":    return "research_generate"
        case _:             return "general_respond"
```

### 5. Update the Supervisor System Prompt

Add classification instructions for the "research" type to the prompt.

### 6. Add Worker to Graph (`src/agent/graph.py`)

```python
RESEARCH_SYSTEM = "You are an assistant for scientific research at SICT HAUI..."

def build_graph() -> CompiledGraph:
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)

    _add_rag_worker(builder, "curriculum", retrieve_curriculum, CURRICULUM_SYSTEM)
    _add_rag_worker(builder, "regulation", retrieve_regulations, REGULATION_SYSTEM)
    _add_rag_worker(builder, "research", retrieve_research, RESEARCH_SYSTEM)  # Add this line

    builder.add_node("general_respond", general_respond_node)
    # ... edges ...
```

## Adding a Custom Node Type

### Node Without Retrieval

```python
def my_custom_node(state: AgentState) -> dict:
    llm = get_chat_model()
    messages = [SystemMessage(content="Custom prompt...")] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

Register it in the graph:
```python
builder.add_node("my_custom", my_custom_node)
builder.add_edge("supervisor", "my_custom")
builder.add_edge("my_custom", END)
```

### Conditional Edge

```python
def my_routing_edge(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if condition:
        return "node_a"
    return "node_b"

builder.add_conditional_edges("some_node", my_routing_edge, {
    "node_a": "node_a",
    "node_b": "node_b"
})
```

## Changing the LLM Provider

All LLM calls go through `src/llm/client.py`. To switch providers:

### Another Provider (OpenAI-compatible API)

```python
# src/llm/client.py
def get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        openai_api_base="https://api.other-provider.com/v1",
        openai_api_key=settings.other_api_key,
        temperature=0,
    )
```

### Native LangChain Provider (e.g., Google Gemini)

```python
from langchain_google_genai import ChatGoogleGenerativeAI

def get_chat_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=settings.google_api_key,
        temperature=0,
    )
```

## Visualize Graph

```bash
python visualize_graph.py
```

Outputs `graph_schema.png` — a diagram of all LangGraph nodes and edges.

## Debug and Troubleshooting

### Check FAISS Index

```python
from src.service.vectorstore import curriculum_store, regulation_store

print("Curriculum ready:", curriculum_store.is_ready)
print("Regulation ready:", regulation_store.is_ready)

# Test retrieval
chunks = curriculum_store.retrieve("software engineering", top_k=3)
for c in chunks:
    print(f"[{c.score:.3f}] {c.section} > {c.subsection}")
```

### Check LLM Connection

```python
from src.llm.client import get_chat_model
from langchain_core.messages import HumanMessage

llm = get_chat_model()
response = llm.invoke([HumanMessage(content="Hello")])
print(response.content)
```

### Log Agent Trace

The Streamlit UI displays traces automatically. For CLI, add:

```python
import logging
logging.getLogger("src.agent").setLevel(logging.DEBUG)
```

### Rebuild Index When Data Changes

```bash
# Full rebuild
python scripts/crawl_website.py
python scripts/split_chunks.py
python scripts/build_index.py

# Rebuild index only (skip crawl)
python scripts/build_index.py
```

## Detailed Directory Structure

```
src/
├── config.py           # Settings — read-only, no circular imports
├── models.py           # Pydantic schemas — do not import from src/agent/
│
├── agent/
│   ├── __init__.py     # Logging setup
│   ├── state.py        # AgentState — do not import from other agent modules
│   ├── tools.py        # @tool functions — imports from service/vectorstore
│   ├── nodes.py        # Node factories — imports tools, llm/client
│   ├── supervisor.py   # Supervisor node — imports llm/client
│   ├── graph.py        # build_graph() — imports all nodes, supervisor
│   └── haui_agent.py   # HAUIAgent — imports graph, models
│
├── llm/
│   ├── __init__.py
│   └── client.py       # get_client(), get_chat_model() — imports config
│
└── service/
    ├── __init__.py
    ├── vectorstore.py  # VectorStore — imports llm/client, models, config
    └── reranker.py     # BaseReranker hierarchy — imports models, config, llm/client (LLMReranker only)
```

**Import order** (no circular imports allowed):
```
config → llm/client → service/vectorstore → agent/tools → agent/nodes
config → models
config → service/reranker (lazy llm/client import inside LLMReranker.rerank)
agent/state → agent/supervisor → agent/nodes → agent/graph → agent/haui_agent
```
