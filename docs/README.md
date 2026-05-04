# HAUI Agent — Project Overview

**HAUI Agent** is a RAG (Retrieval-Augmented Generation) chatbot that helps students and faculty at Hanoi University of Industry (HAUI) — Faculty of Information and Communication Technology (SICT) — look up information about academic programs, university regulations, and the student handbook.

## Key Features

- **Intelligent Query Classification** — A Supervisor classifies user intent before routing to the appropriate agent
- **Context Retrieval (RAG)** — Two separate FAISS indexes for academic programs and university regulations
- **Relevance Grading** — Automatically evaluates retrieved documents and rewrites the query if results are not relevant
- **Multiple Interfaces** — Streamlit web UI and interactive CLI
- **Vietnamese Language Support** — All prompts and data are in Vietnamese

## Agent Graph

The diagram below shows the full LangGraph node structure — how queries flow from the Supervisor through each RAG worker and back to the end.

![Agent Graph](../graph_schema.png)

The Supervisor routes each query to one of three paths: **curriculum** (academic programs), **regulations** (university policies), or **general** (direct LLM response). Each RAG path includes a retrieve → grade → answer loop with an automatic query rewrite if retrieved documents are not relevant.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Fill in OPENROUTER_API_KEY and other environment variables
```

### 3. Run the Application

```bash
# Web interface (Streamlit)
streamlit run app.py

# Interactive CLI
python main.py
```

## Directory Structure

```
Haui_Agent/
├── app.py                      # Streamlit web UI
├── main.py                     # CLI entry point
├── requirements.txt
├── .env.example
│
├── data/                       # Source data (JSON chunks)
│   ├── curriculum_chunks.json
│   ├── regulation_chunks.json
│   └── web_chunks.json
│
├── scripts/                    # Data pipeline
│   ├── crawl_website.py        # Crawl data from the website
│   ├── split_chunks.py         # Classify chunks by domain
│   └── build_index.py          # Build FAISS indexes
│
├── faiss_curriculum.bin/.pkl   # Vector index — academic programs
├── faiss_regulation.bin/.pkl   # Vector index — university regulations
│
├── src/
│   ├── config.py               # Runtime configuration
│   ├── models.py               # Pydantic schemas
│   ├── agent/                  # LangGraph multi-agent
│   ├── llm/                    # LLM client
│   └── service/                # VectorStore
│
└── docs/                       # Project documentation (this directory)
```

## Documentation

| File | Contents |
|------|----------|
| [architecture.md](architecture.md) | System architecture, request flow, components |
| [data-pipeline.md](data-pipeline.md) | Data collection, classification, vector index building |
| [configuration.md](configuration.md) | Environment variables, model configuration, customization |
| [development.md](development.md) | Extension guide, adding new agents/nodes |
| [api-reference.md](api-reference.md) | API class and function reference |

## Technology Stack

| Library | Role |
|---------|------|
| LangGraph | Multi-agent workflow orchestration |
| LangChain | Abstractions for LLM and tool calling |
| FAISS | Vector similarity search |
| OpenRouter | LLM API gateway (Gemini, Claude, etc.) |
| Streamlit | Web UI |
| Pydantic | Data validation |
