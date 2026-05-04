# Cấu hình hệ thống

## Biến môi trường (`.env`)

Sao chép file `.env.example` và điền các giá trị:

```bash
cp .env.example .env
```

### Các biến bắt buộc

| Biến | Ví dụ | Mô tả |
|------|-------|-------|
| `OPENROUTER_API_KEY` | `sk-or-v1-...` | API key từ [openrouter.ai](https://openrouter.ai) |

### Các biến tùy chọn (có giá trị mặc định)

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `OPENROUTER_CHAT_MODEL` | `google/gemini-2.0-flash-001` | Model cho chat, supervisor, grading |
| `OPENROUTER_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Model embedding cho FAISS |
| `RETRIEVAL_TOP_K` | `5` | Số lượng chunks trả về mỗi lần retrieve |
| `FAISS_CURRICULUM_INDEX` | `faiss_curriculum.bin` | Đường dẫn đến FAISS index curriculum |
| `FAISS_CURRICULUM_META` | `faiss_curriculum_meta.pkl` | Đường dẫn đến metadata curriculum |
| `FAISS_REGULATION_INDEX` | `faiss_regulation.bin` | Đường dẫn đến FAISS index regulation |
| `FAISS_REGULATION_META` | `faiss_regulation_meta.pkl` | Đường dẫn đến metadata regulation |

### File `.env` đầy đủ

```env
# Bắt buộc
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Model (tùy chọn)
OPENROUTER_CHAT_MODEL=google/gemini-2.0-flash-001
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small

# Retrieval (tùy chọn)
RETRIEVAL_TOP_K=5

# FAISS index paths (tùy chọn)
FAISS_CURRICULUM_INDEX=faiss_curriculum.bin
FAISS_CURRICULUM_META=faiss_curriculum_meta.pkl
FAISS_REGULATION_INDEX=faiss_regulation.bin
FAISS_REGULATION_META=faiss_regulation_meta.pkl
```

## Settings class (`src/config.py`)

Cấu hình được đọc runtime thông qua `Settings` (Pydantic `BaseSettings`). Các thuộc tính được resolve lazily qua `@property` để `.env` changes có hiệu lực ngay mà không cần restart.

```python
from src.config import settings

# Đọc giá trị cấu hình
model = settings.chat_model          # str
top_k = settings.top_k              # int
base_url = settings.openrouter_base_url  # str
```

## Chọn model

### Models đề xuất qua OpenRouter

| Model | OpenRouter ID | Phù hợp khi |
|-------|--------------|-------------|
| Gemini 2.0 Flash (mặc định) | `google/gemini-2.0-flash-001` | Chi phí thấp, tốc độ nhanh |
| Gemini 2.5 Pro | `google/gemini-2.5-pro-preview-03-25` | Câu trả lời phức tạp, chi phí cao hơn |
| Claude 3.5 Sonnet | `anthropic/claude-3.5-sonnet` | Tiếng Việt tốt, chi phí trung bình |
| DeepSeek Chat V3 | `deepseek/deepseek-chat-v3-0324` | Chi phí rất thấp |
| Llama 3.3 70B | `meta-llama/llama-3.3-70b-instruct` | Open-source, chi phí thấp |

### Thay đổi model qua Streamlit UI

Sidebar → **Chat Model** → chọn từ dropdown  
Thay đổi sẽ reinitialize `HAUIAgent` với model mới.

### Thay đổi model qua `.env`

```env
OPENROUTER_CHAT_MODEL=anthropic/claude-3.5-sonnet
```

## Cấu hình retrieval

### `RETRIEVAL_TOP_K`

Số lượng chunks FAISS trả về cho mỗi câu hỏi. Giá trị cao hơn cho context phong phú hơn nhưng tăng token sử dụng.

- Streamlit UI: Sidebar → **Top-K Retrieval** slider (1–10)
- `.env`: `RETRIEVAL_TOP_K=5`

### Thay đổi FAISS index path

Nếu lưu FAISS indexes ở thư mục khác:

```env
FAISS_CURRICULUM_INDEX=/path/to/my_curriculum.bin
FAISS_CURRICULUM_META=/path/to/my_curriculum_meta.pkl
FAISS_REGULATION_INDEX=/path/to/my_regulation.bin
FAISS_REGULATION_META=/path/to/my_regulation_meta.pkl
```

## Logging

Logging được cấu hình trong `src/agent/__init__.py`:

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
```

Để tắt log verbose khi production:

```python
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
```
