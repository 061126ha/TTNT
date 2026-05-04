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
