# Hướng dẫn phát triển

## Thiết lập môi trường

```bash
# Clone và cài đặt
git clone <repo-url>
cd Haui_Agent
pip install -r requirements.txt

# Cấu hình
cp .env.example .env
# Điền OPENROUTER_API_KEY

# Đảm bảo FAISS indexes đã được build
python scripts/build_index.py
```

## Nguyên tắc thiết kế

Dự án tuân theo **OOP** và **SOLID**:

- **S** — Mỗi class/module có một trách nhiệm duy nhất
- **O** — Mở rộng bằng cách thêm subclass, không sửa base class
- **L** — Mọi implementation phải thay thế được cho abstraction
- **I** — Interface tối giản, không nhồi thêm method không cần thiết
- **D** — Depend vào abstraction, không depend vào concrete class

## Thêm một RAG domain mới

Ví dụ: thêm domain "research" cho thông tin nghiên cứu khoa học.

### 1. Chuẩn bị dữ liệu

```bash
# Tạo chunks JSON cho domain mới
# data/research_chunks.json

# Build FAISS index
python scripts/build_index.py --input data/research_chunks.json \
    --output faiss_research.bin --meta faiss_research_meta.pkl
```

### 2. Cập nhật VectorStore (`src/service/vectorstore.py`)

```python
research_store = VectorStore("faiss_research.bin", "faiss_research_meta.pkl")
```

### 3. Thêm retrieval tool (`src/agent/tools.py`)

```python
@tool
def retrieve_research(query: str) -> str:
    """Find information about scientific research at SICT HAUI."""
    chunks = research_store.retrieve(query)
    return _format_chunks(chunks)
```

### 4. Cập nhật Supervisor (`src/agent/supervisor.py`)

Thêm `"research"` vào danh sách valid types và cập nhật routing:

```python
VALID_TYPES = {"curriculum", "regulations", "general", "research"}

def route_to_agent(state: AgentState) -> str:
    match state["query_type"]:
        case "curriculum":  return "curriculum_generate"
        case "regulations": return "regulation_generate"
        case "research":    return "research_generate"
        case _:             return "general_respond"
```

### 5. Cập nhật system prompt của Supervisor

Thêm hướng dẫn phân loại "research" vào prompt.

### 6. Thêm worker vào graph (`src/agent/graph.py`)

```python
RESEARCH_SYSTEM = "Bạn là trợ lý về nghiên cứu khoa học tại SICT HAUI..."

def build_graph() -> CompiledGraph:
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)

    _add_rag_worker(builder, "curriculum", retrieve_curriculum, CURRICULUM_SYSTEM)
    _add_rag_worker(builder, "regulation", retrieve_regulations, REGULATION_SYSTEM)
    _add_rag_worker(builder, "research", retrieve_research, RESEARCH_SYSTEM)  # Thêm dòng này

    builder.add_node("general_respond", general_respond_node)
    # ... edges ...
```

## Thêm một loại node tùy chỉnh

### Node không sử dụng retrieval

```python
def my_custom_node(state: AgentState) -> dict:
    llm = get_chat_model()
    messages = [SystemMessage(content="Custom prompt...")] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

Đăng ký vào graph:
```python
builder.add_node("my_custom", my_custom_node)
builder.add_edge("supervisor", "my_custom")
builder.add_edge("my_custom", END)
```

### Conditional edge

```python
def my_routing_edge(state: AgentState) -> str:
    # Logic quyết định next node
    last_message = state["messages"][-1]
    if condition:
        return "node_a"
    return "node_b"

builder.add_conditional_edges("some_node", my_routing_edge, {
    "node_a": "node_a",
    "node_b": "node_b"
})
```

## Thay đổi LLM provider

Tất cả LLM calls đều đi qua `src/llm/client.py`. Để đổi provider:

### Sang provider khác (vẫn dùng OpenAI-compatible API)

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

### Sang LangChain provider khác (ví dụ: Google Gemini native)

```python
from langchain_google_genai import ChatGoogleGenerativeAI

def get_chat_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=settings.google_api_key,
        temperature=0,
    )
```

## Visualize graph

```bash
python visualize_graph.py
```

Xuất file `graph_schema.png` — sơ đồ toàn bộ LangGraph nodes và edges.

## Debug và troubleshooting

### Kiểm tra FAISS index

```python
from src.service.vectorstore import curriculum_store, regulation_store

print("Curriculum ready:", curriculum_store.is_ready)
print("Regulation ready:", regulation_store.is_ready)

# Test retrieval
chunks = curriculum_store.retrieve("kỹ thuật phần mềm", top_k=3)
for c in chunks:
    print(f"[{c.score:.3f}] {c.section} > {c.subsection}")
```

### Kiểm tra LLM kết nối

```python
from src.llm.client import get_chat_model
from langchain_core.messages import HumanMessage

llm = get_chat_model()
response = llm.invoke([HumanMessage(content="Hello")])
print(response.content)
```

### Log agent trace

Streamlit UI hiển thị trace tự động. Trong CLI, thêm:

```python
import logging
logging.getLogger("src.agent").setLevel(logging.DEBUG)
```

### Rebuild index khi dữ liệu thay đổi

```bash
# Full rebuild
python scripts/crawl_website.py
python scripts/split_chunks.py
python scripts/build_index.py

# Chỉ rebuild index (không crawl lại)
python scripts/build_index.py
```

## Cấu trúc thư mục chi tiết

```
src/
├── config.py           # Settings — chỉ đọc, không import circular
├── models.py           # Pydantic schemas — không import từ src/agent/
│
├── agent/
│   ├── __init__.py     # Logging setup
│   ├── state.py        # AgentState — không import từ agent modules khác
│   ├── tools.py        # @tool functions — import từ service/vectorstore
│   ├── nodes.py        # Node factories — import tools, llm/client
│   ├── supervisor.py   # Supervisor node — import llm/client
│   ├── graph.py        # build_graph() — import tất cả nodes, supervisor
│   └── haui_agent.py   # HAUIAgent — import graph, models
│
├── llm/
│   ├── __init__.py
│   └── client.py       # get_client(), get_chat_model() — import config
│
└── service/
    ├── __init__.py
    └── vectorstore.py  # VectorStore — import llm/client, models, config
```

**Import order** (không được tạo circular imports):
```
config → llm/client → service/vectorstore → agent/tools → agent/nodes
config → models
agent/state → agent/supervisor → agent/nodes → agent/graph → agent/haui_agent
```
