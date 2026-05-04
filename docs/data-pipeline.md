# Data Pipeline

This document describes the process for collecting, processing, and indexing data from the SICT HAUI website.

## Pipeline Overview

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

## Step 1 — Data Collection (`scripts/crawl_website.py`)

**Source:** `https://sict.haui.edu.vn/vn/`

**Crawled sections:**

| Section | Content |
|---------|---------|
| Introduction | Overview of SICT faculty |
| Training | Academic programs, majors |
| Admissions | Enrollment information, quotas |
| Faculty | Departments and lecturers |
| Offices & Centers | Administrative offices, research centers |
| Science & Technology | Scientific research |

**Process:**
1. Collect seed URLs from 6 main menu sections
2. Crawl each seed URL and linked pages (depth ≤ 2)
3. Parse HTML with BeautifulSoup, extract text content
4. Structure each chunk with metadata: section, subsection, content
5. Filter out chunks with fewer than 30 words

**Output:** `data/web_chunks.json`

```json
[
  {
    "section": "Training",
    "subsection": "Software Engineering Program",
    "content": "The software engineering program...",
    "url": "https://sict.haui.edu.vn/vn/dao-tao/..."
  }
]
```

## Step 2 — Domain Classification (`scripts/split_chunks.py`)

Splits `web_chunks.json` into two separate domains based on keywords in section/subsection.

**Curriculum keywords** (academic programs):
- `"Ngành"`, `"Thạc sỹ"`, `"Thạc sĩ"`, `"Đào tạo"`, `"Tuyển sinh"`

**Classification logic:**
- Chunks whose section/subsection contains curriculum keywords → `curriculum_chunks.json`
- All remaining chunks → `regulation_chunks.json`

**Output:**
- `data/curriculum_chunks.json` — majors, courses, program outcomes (PEO/SO/PI)
- `data/regulation_chunks.json` — university info, departments, policies

## Step 3 — Build FAISS Index (`scripts/build_index.py`)

Run once for each domain.

**Process for each domain:**

1. Load JSON chunks file
2. Build full text for each chunk:
   ```
   {section} > {subsection}
   {content}
   ```
3. Call OpenRouter embedding API (`text-embedding-3-small`) to embed each text
4. L2-normalize each embedding vector
5. Build `faiss.IndexFlatIP(1536)` and add all embeddings
6. Save binary index (`faiss.write_index`)
7. Save metadata list (`pickle.dump`)

**Per-chunk metadata:**
```python
{
    "chunk_id": int,
    "content": str,
    "section": str,
    "subsection": str,
    "text": str,      # full text used for embedding
    "score": float    # similarity score (filled at retrieval time)
}
```

**Re-running the pipeline:**

```bash
# Crawl new data
python scripts/crawl_website.py

# Re-classify
python scripts/split_chunks.py

# Rebuild index
python scripts/build_index.py
```

> **Note:** Only re-run from the step where changes occurred. For example, if only the classification logic changed, there is no need to re-crawl.

## Vector Index

| File | Size | Contents |
|------|------|---------|
| `faiss_curriculum.bin` | ~1.3 MB | Curriculum FAISS index |
| `faiss_curriculum_meta.pkl` | — | Metadata for curriculum chunks |
| `faiss_regulation.bin` | ~2.6 MB | Regulation FAISS index |
| `faiss_regulation_meta.pkl` | — | Metadata for regulation chunks |

**Index type:** `IndexFlatIP` — brute-force inner product (cosine similarity with L2-norm)  
**Embedding dimension:** 1536  
**Embedding model:** `openai/text-embedding-3-small` via OpenRouter

## Source Data Structure

### `data/web_chunks.json`

Raw chunks crawled from the website, not yet classified.

### `data/curriculum_chunks.json`

Academic program information:
- List of majors (Software Engineering, IT, Electronics, ...)
- Enrollment quotas
- Program outcomes: Program Educational Outcomes (PEO), Student Outcomes (SO), Performance Indicators (PI)
- Course list and credit hours

### `data/regulation_chunks.json`

University and policy information:
- Introduction to SICT faculty
- Organizational structure, departments
- Offices and centers
- Strategic plans
- Academic policies and regulations
