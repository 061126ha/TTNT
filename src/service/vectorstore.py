"""
FAISS vector store retrieval service.

Loads a pre-built FAISS index + metadata from disk and provides
a retrieve_context() function that returns the top-K most similar
chunks for a given query.
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

_index: faiss.IndexFlatIP | None = None
_metadata: list[dict] | None = None


def load_vectorstore() -> None:
    """Load the FAISS index and metadata from disk (lazy, one-time)."""
    global _index, _metadata
    if _index is not None:
        return

    index_file = settings.index_file
    metadata_file = settings.metadata_file

    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"FAISS index '{index_file}' not found. Run build_vectorstore.py first."
        )
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(
            f"FAISS metadata '{metadata_file}' not found. Run build_vectorstore.py first."
        )

    _index = faiss.read_index(index_file)
    with open(metadata_file, "rb") as f:
        _metadata = pickle.load(f)

    logger.info(
        "Loaded FAISS index (%d vectors) and metadata (%d entries)",
        _index.ntotal,
        len(_metadata),
    )


def retrieve_context(query: str) -> list[RetrievedChunk]:
    """Embed *query* and return the top-K most similar chunks."""
    load_vectorstore()
    client = get_client()

    resp = client.embeddings.create(model=settings.embedding_model, input=query)
    query_vec = np.array([resp.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(query_vec)

    scores, indices = _index.search(query_vec, settings.top_k)

    chunks: list[RetrievedChunk] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = _metadata[idx]
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

    logger.debug("Retrieved %d chunks for query: %.60s…", len(chunks), query)
    return chunks
