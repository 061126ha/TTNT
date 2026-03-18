"""
Build FAISS vector index from chunk_folder.json.
Run this once before starting the chatbot:
    python build_vectorstore.py
"""

import json
import os
import pickle

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHUNKS_FILE = "chunk_folder.json"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "faiss_metadata.pkl"
EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_client() -> OpenAI:
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Embed texts one at a time (OpenRouter does not support batch input)."""
    all_embeddings = []
    n_error_embeddings_text = 0
    for i, text in enumerate(texts):
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        except Exception as e:
            print(text)
            print("="*100)
            n_error_embeddings_text += 1
        all_embeddings.append(response.data[0].embedding)
        if (i + 1) % 10 == 0 or (i + 1) == len(texts):
            print(f"  Embedded {i + 1}/{len(texts)} chunks...")
    print(f"Finished embedding. {n_error_embeddings_text} texts had errors during embedding.")
    return np.array(all_embeddings, dtype=np.float32)


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an inner-product (cosine similarity) FAISS index."""
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main():
    print(f"Loading chunks from {CHUNKS_FILE}...")
    chunks = load_chunks(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks.")

    texts = []
    metadata_store = []
    for chunk in chunks:
        content = chunk["content"].strip('"""').strip()
        meta = chunk.get("metadata", {})
        section = meta.get("#", "")
        subsection = meta.get("###", "")
        header = " > ".join(filter(None, [section, subsection]))
        text = f"{header}\n{content}" if header else content
        texts.append(text)
        metadata_store.append(
            {
                "chunk_id": chunk["chunk_id"],
                "content": content,
                "section": section,
                "subsection": subsection,
                "text": text,
            }
        )

    client = make_client()

    print(f"Embedding {len(texts)} chunks with model '{EMBEDDING_MODEL}'...")
    embeddings = embed_texts(client, texts)

    print("Building FAISS index...")
    index = build_index(embeddings)

    print(f"Saving index to {INDEX_FILE}...")
    faiss.write_index(index, INDEX_FILE)

    print(f"Saving metadata to {METADATA_FILE}...")
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata_store, f)

    print(f"Done. Index contains {index.ntotal} vectors of dimension {embeddings.shape[1]}.")


if __name__ == "__main__":
    main()