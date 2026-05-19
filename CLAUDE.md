# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the development server
uvicorn src.main:app --reload

# Install dependencies
pip install -r requirements.txt
```

API available at:
- Swagger UI: `http://localhost:8000/docs`
- Chat: `POST /api/v1/chat` — `{"query": "...", "session_id": "..."}`
- Health: `GET /api/v1/health`

## Design Principles

All code in this project must follow **OOP** and **SOLID** principles. The existing structure embodies these — new code must continue to do so.

| Principle | How it applies here |
|-----------|---------------------|
| **S** — Single Responsibility | Routes are HTTP adapters only. Services hold business logic. Providers handle LLM access. Never mix these concerns. |
| **O** — Open/Closed | Add new LLM providers by subclassing `BaseProvider`, new services by subclassing `BaseService`. Never modify existing base classes to add variant behaviour. |
| **L** — Liskov Substitution | Any `BaseProvider` subclass must be a drop-in replacement for `OpenRouterProvider`. Any `BaseService` subclass must honour the `async process(request) -> response` contract. |
| **I** — Interface Segregation | Keep abstract base classes minimal and focused (`BaseService` has only `process`; `BaseProvider` has only `get_llm`). Don't bloat interfaces with methods that only some subclasses need. |
| **D** — Dependency Inversion | Agent nodes depend on the `BaseProvider` abstraction, not `OpenRouterProvider` directly. Services depend on the compiled `graph` abstraction, not individual nodes. |

**Practical rules:**
- Business logic always lives in a `BaseService` subclass — never in a route handler.
- New LLM back-ends must implement `BaseProvider`; swap them via `factory.py`, not by editing callers.
- Favour composition and dependency injection over global state where possible. Module-level singletons (`provider`, `chat_service`, `graph`) are acceptable at the boundary layer only.
- Every new class must have a clear single reason to change.

## Architecture

### Request Flow
`POST /api/v1/chat` → `routes/chat.py` (thin HTTP adapter) → `chat_service.process()` → `graph.ainvoke()` → `chat_agent` node → OpenRouter LLM → response

### Key Layers

**`src/providers/`** — LLM abstraction layer. `BaseProvider` defines `get_llm(task: Task) -> BaseChatModel`. `OpenRouterProvider` wraps `ChatOpenAI` with OpenRouter's base URL and caches instances by model name. The `provider` singleton is created at import time in `__init__.py` and used by the graph.

**`src/agents/graph.py`** — LangGraph workflow. `AgentState` holds a messages list (append-only via `add_messages`). The `chat_agent` node prepends `CHAT_AGENT_PROMPT` as a `SystemMessage`, calls `provider.get_llm(Task.CHAT)`, and returns the AI response. The compiled `graph` is a module-level singleton imported by `ChatService`.

**`src/services/`** — Business logic layer. All services inherit `BaseService(ABC)` and implement `async process(request, response)`. `ChatService` wraps graph invocation and maps between Pydantic schemas and LangChain messages. The `chat_service` singleton is created in `__init__.py`.

**`src/database.py`** — Lazy-init singletons for MongoDB (`AsyncIOMotorClient`) and ChromaDB (`PersistentClient`). Both are initialized on startup via the FastAPI `lifespan` handler in `main.py`. ChromaDB's default collection is `haui_knowledge`.

**`src/config.py`** — Single `Settings` object (Pydantic `BaseSettings`, reads `.env`). `settings.get_model_for_task(task)` maps `Task` enum values to per-task model names (`model_chat`, `model_summarization`).

### Adding New Agent Nodes
1. Add prompt to `src/agents/prompts.py`
2. Add node function to `src/agents/graph.py` using `provider.get_llm(Task.<TASK>)`
3. Wire into the graph via `builder.add_node` / `builder.add_edge`
4. Add a corresponding `Task` enum value in `src/providers/base.py` and a `model_<task>` field in `src/config.py` if a separate model is needed

### Adding New API Endpoints
1. Add Pydantic schemas to `src/api/schemas/`
2. Create a service in `src/services/` inheriting `BaseService`
3. Add route in `src/api/routes/` — delegate entirely to the service
4. Register the router in `src/api/router.py`

## Environment Setup

Copy `.env.example` to `.env` and fill in:
- `OPENROUTER_API_KEY` — required
- `MONGODB_URI` — defaults to `mongodb://localhost:27017`
- `CHROMA_PERSIST_DIR` — defaults to `./chroma_db`
- `MODEL_CHAT` / `MODEL_SUMMARIZATION` — default to `google/gemini-2.5-flash`S