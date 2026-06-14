"""
Vectorstore module - RAW FAISS RETRIEVER VERSION
"""

import os
import pickle
import logging
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core.schema import TextNode, NodeWithScore

load_dotenv()
logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"

class CustomFAISSRetriever:
    """Đọc trực tiếp file .bin và .pkl do build_index.py tạo ra"""
    
    def __init__(self, index_path: str, meta_path: str):
        self.index = None
        self.metadata = []
        
        # Khởi tạo API Client để nhúng câu hỏi
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            logger.warning(f"Index or metadata not found for {index_path}")
            return
            
        try:
            # Tải FAISS index và Metadata
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded {index_path} successfully!")
        except Exception as e:
            logger.error(f"Failed loading index {index_path}: {e}")

    def retrieve(self, query: str, top_k: int = 4):
        if not self.index:
            return []

        try:
            # 1. Nhúng câu hỏi của user thành vector (Embedding)
            res = self.client.embeddings.create(
                model=os.getenv("OPENROUTER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
                input=query
            )
            query_embedding = np.array([res.data[0].embedding], dtype=np.float32)
            faiss.normalize_L2(query_embedding)
            
            # 2. Tìm kiếm các đoạn tài liệu tương đồng nhất
            distances, indices = self.index.search(query_embedding, top_k)
            
            # 3. Đóng gói kết quả theo chuẩn LlamaIndex để tương thích với luồng LangGraph
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.metadata):
                    chunk_data = self.metadata[idx]
                    node = TextNode(
                        text=chunk_data.get("text", chunk_data.get("content", "")),
                        metadata=chunk_data
                    )
                    results.append(NodeWithScore(node=node, score=float(distances[0][i])))
            
            return results
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

# =========================
# LOAD INDEX HELPER
# =========================
def _load_store(base_name: str):
    return CustomFAISSRetriever(f"{base_name}.bin", f"{base_name}.pkl")

# =========================
# PUBLIC STORES (IMPORTANT)
# =========================
curriculum_store = _load_store("faiss_curriculum")
regulation_store = _load_store("faiss_regulation")