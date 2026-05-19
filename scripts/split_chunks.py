"""
Split web_chunks.json (crawled from sict.haui.edu.vn) into domain-specific files.

Domains:
  curriculum  → sections about programs, courses, admissions
                (keywords: Ngành, Thạc sỹ/sĩ, Đào tạo, Tuyển sinh)
  regulation  → all other sections (school intro, departments, research, policy)

Usage:
    python scripts/split_chunks.py
    python scripts/split_chunks.py --input data/web_chunks.json --output-dir data/
"""

import argparse
import json
import os


CURRICULUM_KEYWORDS = ("Ngành", "Thạc sỹ", "Thạc sĩ", "Đào tạo", "Tuyển sinh")


def classify_section(section: str) -> str:
    for keyword in CURRICULUM_KEYWORDS:
        if keyword.lower() in section.lower():
            return "curriculum"
    return "regulation"


def main():
    parser = argparse.ArgumentParser(description="Split web_chunks.json into curriculum / regulation")
    parser.add_argument("--input", default="data/web_chunks.json")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    buckets: dict[str, list] = {"curriculum": [], "regulation": []}
    for chunk in chunks:
        section = chunk.get("metadata", {}).get("#", "")
        domain = classify_section(section)
        buckets[domain].append(chunk)

    for domain, domain_chunks in buckets.items():
        for i, chunk in enumerate(domain_chunks, start=1):
            chunk["chunk_id"] = i

        out_path = os.path.join(args.output_dir, f"{domain}_chunks.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(domain_chunks, f, ensure_ascii=False, indent=2)
        print(f"  {domain}: {len(domain_chunks)} chunks → {out_path}")

    print("\nDone. Next step: run scripts/build_index.py for each domain.")
    print("  python scripts/build_index.py --input data/curriculum_chunks.json --index faiss_curriculum.bin --meta faiss_curriculum_meta.pkl")
    print("  python scripts/build_index.py --input data/regulation_chunks.json --index faiss_regulation.bin --meta faiss_regulation_meta.pkl")


if __name__ == "__main__":
    main()
