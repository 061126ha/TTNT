"""
Generic FAISS index builder — works for any domain chunk file.
"""

import argparse
import json
import os
import pickle
import sys

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


# ─────────────────────────────────────────────
# UTF-8 FIX
# ─────────────────────────────────────────────
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# OPENROUTER CLIENT
# ─────────────────────────────────────────────
def make_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing")

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )


# ─────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────
def embed_texts(client: OpenAI, texts: list[str], model: str):
    embeddings = []
    valid_indices = []

    for i, text in enumerate(texts):
        text = (text or "").strip()

        if not text:
            print(f"[SKIP] Empty text at {i}")
            continue

        try:
            response = client.embeddings.create(
                model=model,
                input=text
            )

            embeddings.append(response.data[0].embedding)
            valid_indices.append(i)

        except Exception as exc:
            print(f"[ERROR] chunk {i}: {exc}")

        if (i + 1) % 10 == 0 or (i + 1) == len(texts):
            print(f"Embedded {i + 1}/{len(texts)}")

    if not embeddings:
        raise ValueError("No embeddings were created. Check input data or API key.")

    print(f"Done embedding. Valid: {len(embeddings)}, Skipped: {len(texts) - len(embeddings)}")

    return np.array(embeddings, dtype=np.float32), valid_indices


# ─────────────────────────────────────────────
# FAISS BUILD
# ─────────────────────────────────────────────
def build_index(embeddings: np.ndarray):
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D array")

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument(
        "--model",
        default=os.getenv(
            "OPENROUTER_EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL
        ),
    )

    args = parser.parse_args()

    print(f"Loading: {args.input}")
    chunks = load_chunks(args.input)
    print(f"Chunks loaded: {len(chunks)}")

    texts = []
    metadata_store = []

    for chunk in chunks:
        content = (chunk.get("content", "") or "").strip()
        meta = chunk.get("metadata", {})

        section = meta.get("#", "")
        subsection = meta.get("###", "")

        header = " > ".join(filter(None, [section, subsection]))
        text = f"{header}\n{content}" if header else content

        texts.append(text)
        metadata_store.append({
            "chunk_id": chunk.get("chunk_id"),
            "content": content,
            "section": section,
            "subsection": subsection,
            "text": text,
            "source": meta.get("source", "local"),
        })

    client = make_client()

    print(f"Embedding with model: {args.model}")
    embeddings, valid_indices = embed_texts(client, texts, args.model)

    # sync metadata safely
    metadata_store = [metadata_store[i] for i in valid_indices]

    print("Building FAISS index...")
    index = build_index(embeddings)

    print(f"Saving index → {args.index}")
    faiss.write_index(index, args.index)

    print(f"Saving metadata → {args.meta}")
    with open(args.meta, "wb") as f:
        pickle.dump(metadata_store, f)

    print(f"Done: {index.ntotal} vectors")


if __name__ == "__main__":
    main()