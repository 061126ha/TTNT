"""
FAISS vector store retrieval service.

Supports multiple named indexes (curriculum, regulation, web) via
the VectorStore class. Each domain agent gets its own isolated index.
"""

import logging
import os
import pickle

import faiss
import numpy as np

from src.config import settings
from src.llm.client import get_client
from src.models import RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Generic lazily-loaded FAISS index for one content domain."""

    def __init__(self, index_file: str, metadata_file: str) -> None:
        self._index_file = index_file
        self._metadata_file = metadata_file
        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict] | None = None

    def _load(self) -> None:
        if self._index is not None:
            return
        if not os.path.exists(self._index_file):
            raise FileNotFoundError(
                f"FAISS index '{self._index_file}' not found. "
                "Run the appropriate build_index script first."
            )
        if not os.path.exists(self._metadata_file):
            raise FileNotFoundError(
                f"FAISS metadata '{self._metadata_file}' not found. "
                "Run the appropriate build_index script first."
            )
        self._index = faiss.read_index(self._index_file)
        with open(self._metadata_file, "rb") as f:
            self._metadata = pickle.load(f)
        logger.info(
            "Loaded '%s' (%d vectors, %d metadata entries)",
            self._index_file,
            self._index.ntotal,
            len(self._metadata),
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Embed *query* and return the top-K most similar chunks."""
        self._load()
        k = top_k or settings.top_k
        client = get_client()

        resp = client.embeddings.create(model=settings.embedding_model, input=query)
        query_vec = np.array([resp.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = self._index.search(query_vec, k)

        chunks: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            chunks.append(
                RetrievedChunk(
                    chunk_id=meta["chunk_id"],
                    content=meta.get("content", ""),
                    section=meta.get("section", ""),
                    subsection=meta.get("subsection", ""),
                    score=float(score),
                    text=meta.get("text", ""),
                )
            )

        logger.debug(
            "Retrieved %d chunks from '%s' for: %.60s…",
            len(chunks),
            self._index_file,
            query,
        )
        return chunks

    @property
    def is_ready(self) -> bool:
        """Return True if index files exist on disk."""
        return os.path.exists(self._index_file) and os.path.exists(self._metadata_file)


# ---------------------------------------------------------------------------
# Domain-specific singletons (lazy-loaded on first use)
# ---------------------------------------------------------------------------

curriculum_store = VectorStore(
    settings.curriculum_index_file,
    settings.curriculum_metadata_file,
)

regulation_store = VectorStore(
    settings.regulation_index_file,
    settings.regulation_metadata_file,
)
