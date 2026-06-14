# System Configuration

## Environment Variables (`.env`)

Copy the `.env.example` file and fill in the values:

```bash
cp .env.example .env
```

### Required Variables

| Variable | Example | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | `sk-or-v1-...` | API key from [openrouter.ai](https://openrouter.ai) |

### Optional Variables (with defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_CHAT_MODEL` | `google/gemini-2.0-flash-001` | Model for chat, supervisor, and grading |
| `OPENROUTER_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model for FAISS |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks returned per retrieval |
| `FAISS_CURRICULUM_INDEX` | `faiss_curriculum.bin` | Path to the curriculum FAISS index |
| `FAISS_CURRICULUM_META` | `faiss_curriculum_meta.pkl` | Path to the curriculum metadata |
| `FAISS_REGULATION_INDEX` | `faiss_regulation.bin` | Path to the regulation FAISS index |
| `FAISS_REGULATION_META` | `faiss_regulation_meta.pkl` | Path to the regulation metadata |
| `RERANKER_TYPE` | `cohere` | Reranker to use: `cohere`, `cross_encoder`, `llm`, or `none` |
| `RERANK_TOP_K` | *(same as `RETRIEVAL_TOP_K`)* | Number of chunks kept after reranking |
| `RERANK_FETCH_MULTIPLIER` | `3` | How many extra candidates to fetch before reranking (`top_k × multiplier`) |
| `COHERE_API_KEY` | *(required for `cohere`)* | API key from [cohere.com](https://cohere.com) |
| `COHERE_RERANK_MODEL` | `rerank-multilingual-v3.0` | Cohere rerank model name |
| `CROSS_ENCODER_MODEL` | `BAAI/bge-reranker-base` | HuggingFace cross-encoder model for local reranking |

### Full `.env` File

```env
# Required
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Model (optional)
OPENROUTER_CHAT_MODEL=google/gemini-2.0-flash-001
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small

# Retrieval (optional)
RETRIEVAL_TOP_K=5

# FAISS index paths (optional)
FAISS_CURRICULUM_INDEX=faiss_curriculum.bin
FAISS_CURRICULUM_META=faiss_curriculum_meta.pkl
FAISS_REGULATION_INDEX=faiss_regulation.bin
FAISS_REGULATION_META=faiss_regulation_meta.pkl

# Reranker (optional — defaults to cohere)
RERANKER_TYPE=cohere
RERANK_TOP_K=5
RERANK_FETCH_MULTIPLIER=3
COHERE_API_KEY=your-cohere-api-key
COHERE_RERANK_MODEL=rerank-multilingual-v3.0
CROSS_ENCODER_MODEL=BAAI/bge-reranker-base
```

## Settings Class (`src/config.py`)

Configuration is read at runtime through the `Settings` class (Pydantic `BaseSettings`). Properties are resolved lazily via `@property` so `.env` changes take effect without a restart.

```python
from src.config import settings

# Read configuration values
model = settings.chat_model          # str
top_k = settings.top_k              # int
base_url = settings.openrouter_base_url  # str
```

## Choosing a Model

### Recommended Models via OpenRouter

| Model | OpenRouter ID | Best when |
|-------|--------------|-----------|
| Gemini 2.0 Flash (default) | `google/gemini-2.0-flash-001` | Low cost, fast responses |
| Gemini 2.5 Pro | `google/gemini-2.5-pro-preview-03-25` | Complex answers, higher cost |
| Claude 3.5 Sonnet | `anthropic/claude-3.5-sonnet` | Good Vietnamese support, mid cost |
| DeepSeek Chat V3 | `deepseek/deepseek-chat-v3-0324` | Very low cost |
| Llama 3.3 70B | `meta-llama/llama-3.3-70b-instruct` | Open-source, low cost |

### Changing Model via Streamlit UI

Sidebar → **Chat Model** → select from dropdown  
The change will reinitialize `HAUIAgent` with the new model.

### Changing Model via `.env`

```env
OPENROUTER_CHAT_MODEL=anthropic/claude-3.5-sonnet
```

## Retrieval Configuration

### `RETRIEVAL_TOP_K`

The number of FAISS chunks returned per query. A higher value provides richer context but increases token usage.

- Streamlit UI: Sidebar → **Top-K Retrieval** slider (1–10)
- `.env`: `RETRIEVAL_TOP_K=5`

## Reranker Configuration

After FAISS retrieval, chunks are re-scored before being passed to the answer node. Configure via `RERANKER_TYPE`.

### Reranker Types

| Type | Description | Requirements |
|------|-------------|--------------|
| `cohere` (default) | Cohere Rerank API — best multilingual quality | `COHERE_API_KEY` |
| `cross_encoder` | Local cross-encoder model — no API key, higher CPU usage | `sentence-transformers` |
| `llm` | LLM scoring — one call per chunk, highest latency | none |
| `none` | No reranking — first `top_k` FAISS results used | none |

### Reranker Parameters

| Variable | Description |
|----------|-------------|
| `RERANK_TOP_K` | Number of chunks to keep after reranking (≤ `RETRIEVAL_TOP_K`) |
| `RERANK_FETCH_MULTIPLIER` | Multiplier for FAISS candidates before reranking (`top_k × N`) |

**Example:** `RETRIEVAL_TOP_K=5`, `RERANK_FETCH_MULTIPLIER=3` → FAISS fetches 15 candidates, reranker selects top `RERANK_TOP_K`.

> **Note:** `RERANK_TOP_K` is automatically capped at `RERANK_FETCH_MULTIPLIER × RETRIEVAL_TOP_K` to prevent the reranker from requesting more documents than were fetched.

### Changing Reranker via Streamlit UI

Sidebar → **Reranker** dropdown. Changing the selection reinitializes the reranker singleton and resets the agent.

### Changing Reranker via `.env`

```env
RERANKER_TYPE=cross_encoder
CROSS_ENCODER_MODEL=BAAI/bge-reranker-base
RERANK_TOP_K=3
RERANK_FETCH_MULTIPLIER=3
```

### Changing the FAISS Index Path

If you store FAISS indexes in a different directory:

```env
FAISS_CURRICULUM_INDEX=/path/to/my_curriculum.bin
FAISS_CURRICULUM_META=/path/to/my_curriculum_meta.pkl
FAISS_REGULATION_INDEX=/path/to/my_regulation.bin
FAISS_REGULATION_META=/path/to/my_regulation_meta.pkl
```

## Logging

Logging is configured in `src/agent/__init__.py`:

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
```

To suppress verbose logs in production:

```python
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
```