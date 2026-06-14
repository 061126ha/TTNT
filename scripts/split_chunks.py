"""
Split web_chunks.json into curriculum / regulation domains.
"""

import argparse
import json
import os
import unicodedata


CURRICULUM_KEYWORDS = (
    "ngành",
    "thạc sỹ",
    "thạc sĩ",
    "đào tạo",
    "tuyển sinh",
)


# ─────────────────────────────
# Normalize text (VERY IMPORTANT)
# ─────────────────────────────
def normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    return text.lower().strip()


def classify_section(section: str) -> str:
    section_norm = normalize(section)

    for keyword in CURRICULUM_KEYWORDS:
        if keyword in section_norm:
            return "curriculum"

    return "regulation"


def main():
    parser = argparse.ArgumentParser(
        description="Split web_chunks.json into curriculum / regulation"
    )

    parser.add_argument("--input", default="data/web_chunks.json")
    parser.add_argument("--output-dir", default="data")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.input}...")

    with open(args.input, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    buckets = {
        "curriculum": [],
        "regulation": []
    }

    for chunk in chunks:
        section = chunk.get("metadata", {}).get("#", "")
        domain = classify_section(section)

        buckets[domain].append(chunk)

    # ─────────────────────────────
    # Save outputs
    # ─────────────────────────────
    for domain, domain_chunks in buckets.items():

        # stable chunk_id
        for i, chunk in enumerate(domain_chunks, start=1):
            chunk["chunk_id"] = i

        out_path = os.path.join(args.output_dir, f"{domain}_chunks.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(domain_chunks, f, ensure_ascii=False, indent=2)

        print(f"{domain}: {len(domain_chunks)} → {out_path}")

    print("\nDone.")
    print("Next:")
    print("python scripts/build_index.py --input data/curriculum_chunks.json --index faiss_curriculum.bin --meta faiss_curriculum_meta.pkl")
    print("python scripts/build_index.py --input data/regulation_chunks.json --index faiss_regulation.bin --meta faiss_regulation_meta.pkl")


if __name__ == "__main__":
    main()