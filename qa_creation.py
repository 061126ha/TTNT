import os
import uuid
import pandas as pd
import asyncio
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import ChatMessage

from autorag.data.qa.generation_gt.llama_index_gen_gt import make_custom_gen_gt
from autorag.data.qa.schema import Raw, Corpus, QA

load_dotenv()

# ─────────────────────────────────────────────
# LLM (OpenRouter via LlamaIndex)
# ─────────────────────────────────────────────
llm = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    api_base="https://openrouter.ai/api/v1",
    model=os.getenv(
        "OPENROUTER_CHAT_MODEL",
        "google/gemini-2.5-flash"
    )
)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
raw_df = pd.read_parquet("parsed.parquet")
raw_instance = Raw(raw_df)

corpus_df = pd.read_parquet("corpus.parquet")
corpus_df["doc_id"] = corpus_df["doc_id"].astype(str)

corpus_instance = Corpus(corpus_df, raw_instance)

# ─────────────────────────────────────────────
# Build initial QA skeleton
# ─────────────────────────────────────────────
qa_df = pd.DataFrame({
    "qid": [str(uuid.uuid4()) for _ in range(len(corpus_df))],
    "query": ["" for _ in range(len(corpus_df))],
    "retrieval_gt": corpus_df["doc_id"].apply(lambda x: [[str(x)]]),
    "generation_gt": ["" for _ in range(len(corpus_df))],
})

initial_qa = QA(qa_df, corpus_instance)

# ─────────────────────────────────────────────
# Custom query generation (VI)
# ─────────────────────────────────────────────
async def custom_vi_query_gen(row, llm):
    # Ensure retrieval_gt_contents exists
    context = "\n".join(
        [
            str(c)
            for sublist in row.get("retrieval_gt_contents", [])
            for c in sublist
        ]
    )

    prompt = f"""Dựa trên văn bản sau đây về Trường CNTT&TT (SICT) - Đại học Công nghiệp Hà Nội:

{context}

Hãy đặt một câu hỏi QUAN TRỌNG, TRỰC TIẾP và bằng TIẾNG VIỆT để kiểm tra thông tin trong văn bản trên.

Yêu cầu: Chỉ trả về nội dung câu hỏi, không thêm lời dẫn."""

    messages = [ChatMessage(role="user", content=prompt)]
    response = await llm.achat(messages)

    row["query"] = response.message.content.strip()
    return row


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────
initial_qa = (
    initial_qa
    .make_retrieval_gt_contents()
    .batch_apply(
        custom_vi_query_gen,
        llm=llm
    )
    .batch_apply(
        make_custom_gen_gt,
        llm=llm,
        system_prompt=(
            "Bạn là trợ lý học thuật tại SICT-HaUI. "
            "Hãy trả lời bằng TIẾNG VIỆT chính xác dựa trên văn bản."
        )
    )
)

# ─────────────────────────────────────────────
# Save output
# ─────────────────────────────────────────────
initial_qa.to_parquet("./qa.parquet")

print(f"--- Hoàn thành! Đã sinh {len(initial_qa.data)} câu hỏi tiếng Việt ---")