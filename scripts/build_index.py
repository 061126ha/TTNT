"""
Generic FAISS index builder — works for any domain chunk file.

Reads a JSON file in chunk_folder.json format, embeds each chunk,
and builds a FAISS IndexFlatIP (cosine similarity via L2-normalisation).

Usage:
    # Build curriculum index
    python scripts/build_index.py \
        --input data/curriculum_chunks.json \
        --index faiss_curriculum.bin \
        --meta  faiss_curriculum_meta.pkl

    # Build regulation index
    python scripts/build_index.py \
        --input data/regulation_chunks.json \
        --index faiss_regulation.bin \
        --meta  faiss_regulation_meta.pkl
"""

import argparse
import json
import os
import pickle
import sys

# ── Fix Windows console encoding for Vietnamese text and special chars ────────
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_client() -> OpenAI:
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


def embed_texts(
    client: OpenAI,
    texts: list[str],
    model: str,
) -> tuple[np.ndarray, list[int]]:
    """Embed texts one at a time. Returns (embeddings, valid_indices)."""
    all_embeddings: list[list[float]] = []
    valid_indices: list[int] = []
    n_errors = 0

    for i, text in enumerate(texts):
        try:
            response = client.embeddings.create(model=model, input=text)
            all_embeddings.append(response.data[0].embedding)
            valid_indices.append(i)
        except Exception as exc:
            print(f"  [ERROR] Failed to embed chunk {i}: {exc}")
            n_errors += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(texts):
            print(f"  Embedded {i + 1}/{len(texts)} chunks...")

    print(f"Finished embedding. {n_errors} errors skipped.")
    return np.array(all_embeddings, dtype=np.float32), valid_indices


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from chunk JSON")
    parser.add_argument("--input", required=True, help="Path to chunks JSON file")
    parser.add_argument("--index", required=True, help="Output FAISS index .bin file")
    parser.add_argument("--meta",  required=True, help="Output metadata .pkl file")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help=f"Embedding model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    args = parser.parse_args()

    print(f"Loading chunks from '{args.input}'...")
    chunks = load_chunks(args.input)
    print(f"Loaded {len(chunks)} chunks.")

    texts: list[str] = []
    metadata_store: list[dict] = []

    for chunk in chunks:
        # Support both raw content and triple-quoted content
        content = chunk.get("content", "").strip('"""').strip()
        meta = chunk.get("metadata", {})
        section = meta.get("#", "")
        subsection = meta.get("###", "")
        header = " > ".join(filter(None, [section, subsection]))
        text = f"{header}\n{content}" if header else content

        texts.append(text)
        metadata_store.append({
            "chunk_id": chunk.get("chunk_id", len(metadata_store) + 1),
            "content": content,
            "section": section,
            "subsection": subsection,
            "text": text,
            "source": meta.get("source", "local"),
        })

    client = make_client()
    print(f"Embedding {len(texts)} chunks with model '{args.model}'...")
    embeddings, valid_indices = embed_texts(client, texts, args.model)

    metadata_store = [metadata_store[i] for i in valid_indices]

    print("Building FAISS index...")
    index = build_index(embeddings)

    print(f"Saving index → '{args.index}'...")
    faiss.write_index(index, args.index)

    print(f"Saving metadata → '{args.meta}'...")
    with open(args.meta, "wb") as f:
        pickle.dump(metadata_store, f)

    print(f"Done. Index: {index.ntotal} vectors × {embeddings.shape[1]} dims.")


if __name__ == "__main__":
    main()
