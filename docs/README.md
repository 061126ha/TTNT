# HAUI Agent — Tổng quan dự án

**HAUI Agent** là chatbot RAG (Retrieval-Augmented Generation) hỗ trợ sinh viên và giảng viên trường Đại học Công nghiệp Hà Nội (HAUI) — Khoa Công nghệ Thông tin và Truyền thông (SICT) tra cứu thông tin về chương trình đào tạo, quy định nhà trường và sổ tay sinh viên.

## Tính năng chính

- **Phân loại câu hỏi thông minh** — Supervisor phân loại ý định người dùng trước khi chuyển tiếp đến agent phù hợp
- **Truy xuất ngữ cảnh (RAG)** — Hai chỉ mục FAISS riêng biệt cho chương trình đào tạo và quy định nhà trường
- **Kiểm tra độ liên quan** — Tự động đánh giá và viết lại câu hỏi nếu kết quả truy xuất không phù hợp
- **Giao diện đa dạng** — Streamlit web UI và CLI tương tác
- **Hỗ trợ tiếng Việt** — Toàn bộ prompt và dữ liệu bằng tiếng Việt

## Khởi động nhanh

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình môi trường

```bash
cp .env.example .env
# Điền OPENROUTER_API_KEY và các biến môi trường khác
```

### 3. Chạy ứng dụng

```bash
# Giao diện web (Streamlit)
streamlit run app.py

# CLI tương tác
python main.py
```

## Cấu trúc thư mục

```
Haui_Agent/
├── app.py                      # Streamlit web UI
├── main.py                     # CLI entry point
├── requirements.txt
├── .env.example
│
├── data/                       # Dữ liệu nguồn (JSON chunks)
│   ├── curriculum_chunks.json
│   ├── regulation_chunks.json
│   └── web_chunks.json
│
├── scripts/                    # Data pipeline
│   ├── crawl_website.py        # Thu thập dữ liệu từ website
│   ├── split_chunks.py         # Phân loại chunks theo domain
│   └── build_index.py          # Xây dựng FAISS index
│
├── faiss_curriculum.bin/.pkl   # Vector index — chương trình đào tạo
├── faiss_regulation.bin/.pkl   # Vector index — quy định nhà trường
│
├── src/
│   ├── config.py               # Cấu hình runtime
│   ├── models.py               # Pydantic schemas
│   ├── agent/                  # LangGraph multi-agent
│   ├── llm/                    # LLM client
│   └── service/                # VectorStore
│
└── docs/                       # Tài liệu dự án (thư mục này)
```

## Tài liệu chi tiết

| File | Nội dung |
|------|----------|
| [architecture.md](architecture.md) | Kiến trúc hệ thống, luồng xử lý, các component |
| [data-pipeline.md](data-pipeline.md) | Thu thập dữ liệu, phân loại, xây dựng vector index |
| [configuration.md](configuration.md) | Biến môi trường, cấu hình model, tùy chỉnh |
| [development.md](development.md) | Hướng dẫn mở rộng, thêm agent/node mới |
| [api-reference.md](api-reference.md) | Tham chiếu API classes và functions |

## Công nghệ sử dụng

| Thư viện | Vai trò |
|----------|---------|
| LangGraph | Điều phối multi-agent workflow |
| LangChain | Abstractions cho LLM và tool calling |
| FAISS | Vector similarity search |
| OpenRouter | LLM API gateway (Gemini, Claude, v.v.) |
| Streamlit | Web UI |
| Pydantic | Data validation |
