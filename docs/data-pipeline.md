# Data Pipeline

Tài liệu mô tả quy trình thu thập, xử lý và lập chỉ mục dữ liệu từ website SICT HAUI.

## Tổng quan pipeline

```
sict.haui.edu.vn
      │
      ▼
scripts/crawl_website.py
      │  data/web_chunks.json
      ▼
scripts/split_chunks.py
      │  data/curriculum_chunks.json
      │  data/regulation_chunks.json
      ▼
scripts/build_index.py
      │  faiss_curriculum.bin + faiss_curriculum_meta.pkl
      └  faiss_regulation.bin + faiss_regulation_meta.pkl
```

## Bước 1 — Thu thập dữ liệu (`scripts/crawl_website.py`)

**Nguồn:** `https://sict.haui.edu.vn/vn/`

**Các mục được crawl:**

| Mục | Nội dung |
|-----|----------|
| Giới thiệu | Tổng quan về khoa SICT |
| Đào tạo | Chương trình đào tạo, ngành học |
| Tuyển sinh | Thông tin tuyển sinh, chỉ tiêu |
| Khoa | Các bộ môn và giảng viên |
| Phòng & Trung tâm | Các phòng ban, trung tâm nghiên cứu |
| Khoa học - Công nghệ | Nghiên cứu khoa học |

**Quy trình:**
1. Lấy danh sách URL seed từ 6 mục menu chính
2. Crawl từng URL seed và các trang liên kết (depth ≤ 2)
3. Parse HTML với BeautifulSoup, trích xuất nội dung văn bản
4. Cấu trúc mỗi chunk với metadata: section, subsection, content
5. Lọc bỏ chunks có ít hơn 30 từ

**Output:** `data/web_chunks.json`

```json
[
  {
    "section": "Đào tạo",
    "subsection": "Chương trình Kỹ thuật phần mềm",
    "content": "Chương trình đào tạo kỹ sư...",
    "url": "https://sict.haui.edu.vn/vn/dao-tao/..."
  }
]
```

## Bước 2 — Phân loại theo domain (`scripts/split_chunks.py`)

Phân chia `web_chunks.json` thành hai domain riêng biệt dựa trên từ khóa trong section/subsection.

**Curriculum keywords** (chương trình đào tạo):
- `"Ngành"`, `"Thạc sỹ"`, `"Thạc sĩ"`, `"Đào tạo"`, `"Tuyển sinh"`

**Logic phân loại:**
- Chunk có section/subsection chứa curriculum keywords → `curriculum_chunks.json`
- Các chunk còn lại → `regulation_chunks.json`

**Output:**
- `data/curriculum_chunks.json` — thông tin ngành học, môn học, chuẩn đầu ra (PEO/SO/PI)
- `data/regulation_chunks.json` — thông tin nhà trường, bộ môn, chính sách

## Bước 3 — Xây dựng FAISS index (`scripts/build_index.py`)

Chạy hai lần, một lần cho mỗi domain.

**Quy trình cho mỗi domain:**

1. Load file JSON chunks
2. Tạo text đầy đủ cho mỗi chunk:
   ```
   {section} > {subsection}
   {content}
   ```
3. Gọi OpenRouter embedding API (`text-embedding-3-small`) để embed từng text
4. L2-normalize từng embedding vector
5. Xây dựng `faiss.IndexFlatIP(1536)` và add tất cả embeddings
6. Lưu index binary (`faiss.write_index`)
7. Lưu metadata list (`pickle.dump`)

**Metadata mỗi chunk:**
```python
{
    "chunk_id": int,
    "content": str,
    "section": str,
    "subsection": str,
    "text": str,      # full text dùng để embed
    "score": float    # similarity score (điền khi retrieve)
}
```

**Chạy lại pipeline:**

```bash
# Crawl dữ liệu mới
python scripts/crawl_website.py

# Phân loại lại
python scripts/split_chunks.py

# Rebuild index
python scripts/build_index.py
```

> **Lưu ý:** Chỉ cần chạy lại từ bước có thay đổi. Ví dụ nếu chỉ thay đổi logic phân loại, không cần crawl lại.

## Vector Index

| File | Kích thước | Nội dung |
|------|-----------|---------|
| `faiss_curriculum.bin` | ~1.3 MB | Curriculum FAISS index |
| `faiss_curriculum_meta.pkl` | — | Metadata cho curriculum chunks |
| `faiss_regulation.bin` | ~2.6 MB | Regulation FAISS index |
| `faiss_regulation_meta.pkl` | — | Metadata cho regulation chunks |

**Index type:** `IndexFlatIP` — brute-force inner product (cosine similarity với L2-norm)  
**Embedding dimension:** 1536  
**Embedding model:** `openai/text-embedding-3-small` qua OpenRouter

## Cấu trúc dữ liệu nguồn

### `data/web_chunks.json`

Raw chunks crawled từ website, chưa phân loại.

### `data/curriculum_chunks.json`

Thông tin chương trình đào tạo:
- Danh sách ngành học (Kỹ thuật phần mềm, CNTT, Điện tử...)
- Chỉ tiêu tuyển sinh
- Chuẩn đầu ra: Program Educational Outcomes (PEO), Student Outcomes (SO), Performance Indicators (PI)
- Danh sách môn học và tín chỉ

### `data/regulation_chunks.json`

Thông tin nhà trường và quy định:
- Giới thiệu về khoa SICT
- Cơ cấu tổ chức, các bộ môn
- Phòng ban, trung tâm
- Kế hoạch chiến lược
- Chính sách, quy định học vụ
