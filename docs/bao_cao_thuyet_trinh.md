# BÁO CÁO DỰ ÁN
## HAUI AGENT — HỆ THỐNG HỎI ĐÁP THÔNG MINH
### Trường Đại học Công nghiệp Hà Nội — Khoa Công nghệ Thông tin và Truyền thông

---

## I. GIỚI THIỆU CHUNG

### 1.1 Bối cảnh

Sinh viên và giảng viên Khoa CNTT&TT — ĐHCNHN thường xuyên cần tra cứu thông tin về:
- Chương trình đào tạo, học phần, chuẩn đầu ra
- Quy chế học vụ, chính sách nhà trường
- Thông tin tổ chức, phòng ban, khoa

Hiện tại, các thông tin này nằm rải rác trên website và tài liệu PDF, khiến người dùng phải tìm kiếm thủ công tốn nhiều thời gian.

### 1.2 Đề xuất giải pháp

Xây dựng **HAUI Agent** — một chatbot AI thông minh ứng dụng kiến trúc **RAG (Retrieval-Augmented Generation)** kết hợp **Multi-Agent** để:
- Hiểu ngôn ngữ tự nhiên (tiếng Việt)
- Truy xuất thông tin chính xác từ dữ liệu thực của trường
- Trả lời câu hỏi có căn cứ, không "hallucinate"

---

## II. MỤC TIÊU DỰ ÁN

| # | Mục tiêu | Mô tả |
|---|----------|-------|
| 1 | **Tự động hóa tra cứu** | Trả lời câu hỏi về chương trình đào tạo và quy chế nhà trường |
| 2 | **Độ chính xác cao** | Câu trả lời dựa trên dữ liệu thực, có nguồn trích dẫn |
| 3 | **Hỗ trợ tiếng Việt** | Giao tiếp hoàn toàn bằng tiếng Việt |
| 4 | **Dễ mở rộng** | Thêm miền kiến thức mới mà không cần viết lại hệ thống |
| 5 | **Giao diện thân thiện** | Streamlit Web UI và CLI cho nhiều đối tượng người dùng |

---

## III. KIẾN TRÚC TỔNG QUAN

### 3.1 Sơ đồ kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│          Streamlit Web UI  │  CLI (main.py)                 │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│                    HAUI AGENT LAYER                          │
│  ┌──────────────┐                                            │
│  │  Supervisor  │ ── phân loại ý định người dùng            │
│  └──────┬───────┘                                            │
│         │                                                    │
│    ┌────┴──────────────────────┐                             │
│    │                           │                             │
│  ┌─▼──────────────┐    ┌───────▼──────────┐    ┌──────────┐ │
│  │ Curriculum     │    │ Regulation       │    │ General  │ │
│  │ Worker (RAG)   │    │ Worker (RAG)     │    │ Fallback │ │
│  └─────────────┬──┘    └──────────┬───────┘    └──────────┘ │
│                │                  │                          │
│          ┌─────▼──────────────────▼─────┐                   │
│          │    Grade + Rewrite Loop      │                   │
│          └─────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│                    SERVICE LAYER                             │
│        VectorStore (FAISS)  │  LLM Client (OpenRouter)      │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│                    DATA LAYER                                │
│   FAISS Curriculum Index  │  FAISS Regulation Index         │
│   (faiss_curriculum.bin)  │  (faiss_regulation.bin)         │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Các tầng chính

| Tầng | Thành phần | Trách nhiệm |
|------|-----------|-------------|
| **Presentation** | `app.py`, `main.py` | Giao diện Web & CLI |
| **Agent** | `graph.py`, `supervisor.py`, `nodes.py`, `tools.py` | Luồng xử lý LangGraph |
| **Service** | `vectorstore.py` | Quản lý FAISS, tìm kiếm ngữ nghĩa |
| **LLM** | `client.py` | Client kết nối OpenRouter API |
| **Config** | `config.py`, `models.py` | Cấu hình, Pydantic schemas |
| **Data Pipeline** | `scripts/` | Thu thập, phân loại, đánh chỉ số dữ liệu |

---

## IV. CÔNG NGHỆ SỬ DỤNG

### 4.1 Stack công nghệ chính

| Công nghệ | Vai trò | Lý do chọn |
|-----------|---------|------------|
| **LangGraph** | Orchestration đa agent | Luồng xử lý phức tạp, có điều kiện, dễ debug |
| **LangChain** | Trừu tượng hóa LLM, Tool Calling | Tích hợp tốt với LangGraph, nhiều connector |
| **FAISS (Facebook AI)** | Vector similarity search | Hiệu năng cao, phù hợp CPU, brute-force chính xác |
| **OpenRouter** | LLM Gateway | Truy cập nhiều model AI từ 1 API |
| **Streamlit** | Web UI framework | Prototyping nhanh, phù hợp ML/AI apps |
| **Pydantic v2** | Validation & Settings | Type-safe, tích hợp với FastAPI/LangChain |
| **OpenAI SDK** | Embeddings API | `text-embedding-3-small` chất lượng cao |
| **BeautifulSoup4** | Web scraping | Thu thập dữ liệu từ website HAUI |

### 4.2 Mô hình AI

| Mô hình | Nhiệm vụ | Provider |
|---------|----------|----------|
| `google/gemini-2.0-flash-001` | Sinh câu trả lời, phân loại | Google via OpenRouter |
| `openai/text-embedding-3-small` | Nhúng văn bản (1536 chiều) | OpenAI via OpenRouter |

### 4.3 Cấu trúc thư mục project

```
Haui_Agent/
├── app.py                    ← Streamlit Web UI
├── main.py                   ← CLI entry point
├── requirements.txt
├── data/                     ← Dữ liệu JSON đã phân loại
│   ├── curriculum_chunks.json
│   ├── regulation_chunks.json
│   └── web_chunks.json
├── scripts/                  ← Pipeline thu thập & đánh chỉ số
│   ├── crawl_website.py
│   ├── split_chunks.py
│   └── build_index.py
├── faiss_curriculum.bin/.pkl ← Vector index chương trình đào tạo
├── faiss_regulation.bin/.pkl ← Vector index quy chế nhà trường
└── src/
    ├── config.py
    ├── models.py
    ├── agent/                ← LangGraph multi-agent
    │   ├── graph.py
    │   ├── supervisor.py
    │   ├── nodes.py
    │   ├── tools.py
    │   └── state.py
    ├── llm/client.py
    └── service/vectorstore.py
```

---

## V. PIPELINE DỮ LIỆU (ETL)

### 5.1 Tổng quan quy trình

```
[Website HAUI]
     │
     ▼  BƯỚC 1: Thu thập
[crawl_website.py]
 • Crawl sict.haui.edu.vn (6 mục chính)
 • Depth ≤ 2 hops
 • Lọc đoạn văn ≥ 30 từ
 • Gắn metadata: section, subsection
     │
     ▼ OUTPUT: data/web_chunks.json
     │
     ▼  BƯỚC 2: Phân loại miền
[split_chunks.py]
 • Keyword-based classification
 • Curriculum: "Ngành", "Đào tạo", "Tuyển sinh", "Thạc sỹ"
 • Regulation: còn lại (trường, khoa, phòng ban, chính sách)
     │
     ├─ OUTPUT: data/curriculum_chunks.json
     └─ OUTPUT: data/regulation_chunks.json
     │
     ▼  BƯỚC 3: Đánh chỉ số FAISS
[build_index.py]
 • Gọi OpenRouter Embeddings API (text-embedding-3-small)
 • L2-normalize vector (cosine similarity)
 • Xây IndexFlatIP
 • Lưu .bin (index) + .pkl (metadata)
     │
     ├─ OUTPUT: faiss_curriculum.bin + faiss_curriculum_meta.pkl
     └─ OUTPUT: faiss_regulation.bin + faiss_regulation_meta.pkl
```

### 5.2 Thống kê dữ liệu

| Chỉ số | Giá trị |
|--------|---------|
| Kích thước index chương trình đào tạo | 1.3 MB + 681 KB metadata |
| Kích thước index quy chế nhà trường | 2.5 MB + 1.4 MB metadata |
| Chiều vector embedding | 1536 |
| Loại FAISS index | IndexFlatIP (Inner Product) |
| Phương pháp đo tương đồng | Cosine Similarity |

---

## VI. CHI TIẾT KIẾN TRÚC MULTI-AGENT

### 6.1 Luồng xử lý tổng quát

```
Người dùng nhập câu hỏi
        │
        ▼
   [SUPERVISOR]
   Phân loại: curriculum / regulations / general
        │
   ┌────┴────────────────┐
   │                     │                  │
   ▼                     ▼                  ▼
[CURRICULUM          [REGULATION        [GENERAL
 WORKER]              WORKER]            FALLBACK]
   │                     │
   └──────┬──────────────┘
          │
   ┌──────▼──────┐
   │   Generate  │ LLM quyết định: cần tool không?
   └──────┬──────┘
          │ Tool Call
   ┌──────▼──────┐
   │  Retrieve   │ FAISS tìm top-K chunks liên quan
   └──────┬──────┘
          │
   ┌──────▼──────┐
   │    Grade    │ LLM chấm: tài liệu có liên quan không?
   └──────┬──────┘
          │
     ┌────┴────┐
  Relevant  Not Relevant
     │           │
     ▼           ▼
  [Answer]   [Rewrite]  Cải thiện câu hỏi, thử lại (tối đa 2 lần)
     │
     ▼
Trả lời + Nguồn trích dẫn
```

### 6.2 State Management

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # Lịch sử hội thoại
    query_type: str      # "curriculum" | "regulations" | "general"
    retry_count: int     # Số lần thử lại (0-2)
```

### 6.3 Các Node chính

| Node | Chức năng | Kỹ thuật |
|------|-----------|----------|
| **supervisor** | Phân loại ý định | LLM prompt engineering, structured output |
| **generate_curriculum** | Sinh câu trả lời chương trình | Tool calling với `retrieve_curriculum` |
| **generate_regulations** | Sinh câu trả lời quy chế | Tool calling với `retrieve_regulations` |
| **retrieve** | Thực thi tìm kiếm FAISS | ToolNode (LangGraph built-in) |
| **grade_documents** | Đánh giá độ liên quan | Structured output (GradeDocuments) |
| **rewrite_question** | Cải thiện câu hỏi | LLM prompt rewriting |
| **answer** | Tổng hợp câu trả lời cuối | LLM + retrieved context |

### 6.4 Hệ thống Prompt (tiếng Việt)

| Prompt | Mục đích |
|--------|----------|
| `SUPERVISOR_PROMPT` | Phân loại câu hỏi vào 3 nhóm |
| `CURRICULUM_SYSTEM` | Hướng dẫn trả lời về đào tạo, PEO, PI, SO |
| `REGULATION_SYSTEM` | Hướng dẫn trả lời về trường, khoa, chính sách |
| `GENERAL_SYSTEM` | Từ chối lịch sự câu hỏi ngoài phạm vi |
| `GRADE_PROMPT` | Chấm điểm tài liệu (binary: yes/no) |
| `REWRITE_PROMPT` | Cải thiện diễn đạt câu hỏi |
| `GENERATE_PROMPT` | Tổng hợp câu trả lời có căn cứ |

### 6.5 Cơ chế Tool Calling

```python
# Định nghĩa tool
@tool
def retrieve_curriculum(query: str) -> str:
    """Tìm thông tin về chương trình đào tạo, ngành học, học phần..."""
    chunks = curriculum_store.retrieve(query)
    return format_chunks(chunks)

@tool
def retrieve_regulations(query: str) -> str:
    """Tìm thông tin chung về trường, khoa, phòng ban..."""
    chunks = regulation_store.retrieve(query)
    return format_chunks(chunks)

# Gắn tool vào model
model_with_tool = get_chat_model().bind_tools([retrieve_curriculum])
```

LLM tự quyết định khi nào cần gọi tool — không hardcode điều kiện.

---

## VII. VECTOR STORE & RETRIEVAL

### 7.1 Kiến trúc VectorStore

```python
class VectorStore:
    def __init__(self, index_file: str, metadata_file: str)
    def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]
    @property
    def is_ready(self) -> bool
```

**Lazy Loading:** Index chỉ được load khi có query đầu tiên — tiết kiệm bộ nhớ khi khởi động.

### 7.2 Quy trình Retrieval

```
Câu hỏi người dùng
       │
       ▼
  Embed query  →  text-embedding-3-small  →  vector 1536 chiều
       │
       ▼
  L2-normalize  →  vector unit
       │
       ▼
  FAISS IndexFlatIP.search(vector, top_k)  →  indices + scores
       │
       ▼
  Map indices → metadata (section, subsection, content)
       │
       ▼
  RetrievedChunk(content, section, subsection, score)
```

### 7.3 Cấu trúc dữ liệu trả về

```python
class RetrievedChunk(BaseModel):
    chunk_id: int | str
    content: str       # Nội dung đoạn văn
    section: str       # Mục chính (ví dụ: "Đào tạo")
    subsection: str    # Mục phụ (ví dụ: "Chương trình Kỹ thuật phần mềm")
    score: float       # Điểm tương đồng cosine (0.0 - 1.0)
```

---

## VIII. NGUYÊN TẮC THIẾT KẾ — OOP & SOLID

### 8.1 Tại sao OOP & SOLID?

Dự án quy mô vừa nhưng **cần khả năng mở rộng cao** — thêm miền mới, thêm provider LLM mới, thêm API mới. SOLID đảm bảo điều này mà không cần sửa code cũ.

### 8.2 Áp dụng cụ thể

| Nguyên tắc | Áp dụng trong dự án |
|-----------|---------------------|
| **S** — Single Responsibility | Route chỉ là HTTP adapter. Service chứa business logic. Provider quản lý LLM. Tách biệt hoàn toàn |
| **O** — Open/Closed | Thêm LLM provider mới bằng cách kế thừa `BaseProvider`. Không sửa code cũ |
| **L** — Liskov Substitution | Mọi `BaseProvider` đều dùng được thay cho `OpenRouterProvider` |
| **I** — Interface Segregation | `BaseService` chỉ có `process()`. Không bloat interface |
| **D** — Dependency Inversion | Agent node phụ thuộc `BaseProvider` trừu tượng, không phụ thuộc implementation cụ thể |

### 8.3 Phân cấp dependency

```
config.py
    └── llm/client.py
            └── service/vectorstore.py
                        └── agent/tools.py
                                    └── agent/nodes.py
                                                └── agent/supervisor.py
                                                            └── agent/graph.py
                                                                        └── agent/haui_agent.py
                                                                                    └── app.py / main.py
```

---

## IX. GIAO DIỆN NGƯỜI DÙNG

### 9.1 Streamlit Web UI (`app.py`)

**Tính năng:**
- Giao diện chat trực quan, lưu lịch sử hội thoại
- Sidebar cấu hình: chọn model AI, điều chỉnh Top-K retrieval
- **Agent Trace:** Hiển thị đường đi thực thi qua các node
- **Intent Badge:** Nhãn phân loại (chương trình / quy chế / chung)
- **Source Panel:** Expandable — xem các đoạn văn được trích dẫn + điểm tương đồng
- Nút Reset để xóa lịch sử hội thoại

**Truy cập:** `http://localhost:8501` sau khi chạy `streamlit run app.py`

### 9.2 CLI (`main.py`)

**Tính năng:**
- Vòng lặp hỏi đáp tương tác
- Lệnh `/reset` — xóa lịch sử
- Lệnh `/quit`, `/exit` — thoát
- Hiển thị ý định và nguồn tài liệu

---

## X. THIẾT LẬP & TRIỂN KHAI

### 10.1 Yêu cầu môi trường

```bash
# Cài dependencies
pip install -r requirements.txt

# Sao chép và điền thông tin môi trường
cp .env.example .env
# Điền OPENROUTER_API_KEY vào .env
```

### 10.2 Biến môi trường

| Biến | Bắt buộc | Mặc định |
|------|----------|---------|
| `OPENROUTER_API_KEY` | ✅ Có | — |
| `OPENROUTER_CHAT_MODEL` | Không | `google/gemini-2.0-flash-001` |
| `OPENROUTER_EMBEDDING_MODEL` | Không | `openai/text-embedding-3-small` |
| `RETRIEVAL_TOP_K` | Không | `5` |

### 10.3 Chạy pipeline dữ liệu (một lần)

```bash
python scripts/crawl_website.py    # Thu thập dữ liệu
python scripts/split_chunks.py     # Phân loại miền
python scripts/build_index.py      # Xây FAISS index
```

### 10.4 Khởi chạy ứng dụng

```bash
# Web UI
streamlit run app.py

# CLI
python main.py
```

---

## XI. VÍ DỤ SỬ DỤNG

### 11.1 Câu hỏi về chương trình đào tạo

> **Người dùng:** "Ngành Kỹ thuật phần mềm có những học phần nào bắt buộc?"

```
→ Supervisor phân loại: curriculum
→ Curriculum Worker gọi retrieve_curriculum("học phần bắt buộc kỹ thuật phần mềm")
→ FAISS tìm top-5 chunks liên quan
→ Grade: relevant ✅
→ Answer: Danh sách học phần + nguồn từ chương trình đào tạo
```

### 11.2 Câu hỏi về quy chế

> **Người dùng:** "Điều kiện để được xét học bổng khuyến khích học tập là gì?"

```
→ Supervisor phân loại: regulations
→ Regulation Worker gọi retrieve_regulations("điều kiện học bổng khuyến khích")
→ FAISS tìm top-5 chunks liên quan
→ Grade: relevant ✅
→ Answer: Điều kiện cụ thể + trích dẫn quy chế
```

### 11.3 Câu hỏi không liên quan

> **Người dùng:** "Thủ đô của Pháp là gì?"

```
→ Supervisor phân loại: general
→ General Fallback: từ chối lịch sự, gợi ý hỏi về thông tin trường
```

### 11.4 Trường hợp Query Rewriting

> **Người dùng:** "cho biết về PI?"

```
→ retrieve("PI") → chunks không liên quan
→ Grade: not relevant ❌
→ Rewrite: "Performance Indicators chương trình đào tạo"
→ retrieve lại → chunks liên quan ✅
→ Answer: Giải thích PI (Performance Indicators) trong chuẩn đầu ra
```

---

## XII. ĐIỂM MẠNH KỸ THUẬT

| Điểm mạnh | Mô tả |
|-----------|-------|
| **Multi-Domain RAG** | Index riêng biệt cho từng miền → truy xuất chính xác hơn |
| **Vòng lặp kiểm soát chất lượng** | Tự động viết lại câu hỏi khi context không liên quan |
| **Tool Calling thông minh** | LLM tự quyết định khi nào cần truy xuất, không hardcode |
| **Lazy Loading** | FAISS index chỉ load khi có query đầu tiên |
| **Tiếng Việt native** | Prompt được viết bằng tiếng Việt, phù hợp ngữ cảnh Việt Nam |
| **Trích dẫn nguồn** | Mỗi câu trả lời đi kèm nguồn tài liệu và điểm tương đồng |
| **Dễ mở rộng** | Thêm miền mới chỉ cần: JSON chunks → FAISS index → tool → supervisor rule |
| **Observable** | LangGraph trace + Streamlit visualization cho debugging |

---

## XIII. HƯỚNG PHÁT TRIỂN

| Tính năng | Mô tả | Độ ưu tiên |
|-----------|-------|-----------|
| **REST API** | FastAPI endpoint để tích hợp vào website HAUI | Cao |
| **MongoDB history** | Lưu lịch sử hội thoại theo session | Cao |
| **ChromaDB** | Thay thế FAISS bằng vector DB có metadata filtering | Trung bình |
| **Streaming response** | Hiển thị câu trả lời theo dạng stream | Trung bình |
| **Miền kiến thức mới** | Thêm: lịch thi, thời khóa biểu, thông báo sự kiện | Cao |
| **Đánh giá tự động** | RAGAS metrics: faithfulness, answer relevancy | Trung bình |
| **Xác thực người dùng** | Phân quyền sinh viên / giảng viên / admin | Thấp |
| **Fine-tuning embedding** | Huấn luyện embedding model trên dữ liệu HAUI | Thấp |

---

## XIV. KẾT LUẬN

**HAUI Agent** là một hệ thống RAG đa agent hoàn chỉnh, được xây dựng theo các nguyên tắc kỹ thuật phần mềm hiện đại:

✅ **Kiến trúc rõ ràng** — Phân tầng OOP/SOLID, dễ bảo trì và mở rộng

✅ **AI pipeline đầy đủ** — Từ thu thập dữ liệu, embedding, retrieval đến generation

✅ **Kiểm soát chất lượng** — Grade + Rewrite loop đảm bảo câu trả lời có căn cứ

✅ **Trải nghiệm người dùng tốt** — Giao diện Web + CLI, hỗ trợ tiếng Việt

✅ **Sẵn sàng mở rộng** — Kiến trúc thiết kế để dễ dàng thêm miền, model, endpoint mới

Dự án thể hiện khả năng ứng dụng **Large Language Models** và **Agentic AI** vào bài toán thực tế tại một cơ sở giáo dục, với thiết kế hướng đến sản phẩm sẵn sàng triển khai.

---

## PHỤ LỤC — THUẬT NGỮ

| Thuật ngữ | Giải thích |
|-----------|-----------|
| **RAG** | Retrieval-Augmented Generation — sinh câu trả lời dựa trên tài liệu được truy xuất |
| **LangGraph** | Framework xây dựng luồng xử lý AI dạng đồ thị (graph-based) |
| **FAISS** | Facebook AI Similarity Search — thư viện tìm kiếm vector tốc độ cao |
| **Embedding** | Biểu diễn văn bản thành vector số để so sánh ngữ nghĩa |
| **Cosine Similarity** | Đo độ tương đồng giữa hai vector (0: không liên quan, 1: giống hệt) |
| **Tool Calling** | LLM tự gọi hàm/API để lấy thêm thông tin |
| **Supervisor** | Node phân loại ý định, điều phối các worker agent |
| **IndexFlatIP** | FAISS index dùng inner product (tương đương cosine với vector normalized) |
| **PEO** | Program Educational Objectives — Mục tiêu giáo dục chương trình |
| **SO/PI** | Student Outcomes / Performance Indicators — Chuẩn đầu ra sinh viên |

---

*Báo cáo được tạo dựa trên mã nguồn thực tế tại `d:\Project\Haui_Agent`*
*Ngày tạo: 29/04/2026*
