"""
Web crawler for HAUI SICT website (https://sict.haui.edu.vn/vn/).

Crawls the menu sections from "Gioi thieu" to "Khoa hoc - cong nghe",
extracts text content, chunks by heading, and saves to JSON in the
same format as chunk_folder.json.

Usage:
    python scripts/crawl_website.py
    python scripts/crawl_website.py --output data/web_chunks.json --delay 1.0
"""

import argparse
import json
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Target URLs (in order: menu "Gioi thieu" → "Khoa hoc - cong nghe")
# ---------------------------------------------------------------------------

BASE_URL = "https://sict.haui.edu.vn"
SEED_URLS = [
    "https://sict.haui.edu.vn/vn/gioi-thieu/",
    "https://sict.haui.edu.vn/vn/dao-tao/",
    "https://sict.haui.edu.vn/vn/tuyen-sinh/",
    "https://sict.haui.edu.vn/vn/khoa/",
    "https://sict.haui.edu.vn/vn/phong-trung-tam/",
    "https://sict.haui.edu.vn/vn/khoa-hoc-cong-nghe/",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}

MAX_DEPTH = 2
MIN_WORDS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_same_domain(url: str) -> bool:
    return urlparse(url).netloc == urlparse(BASE_URL).netloc


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _fetch(url: str, session: requests.Session) -> BeautifulSoup | None:
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        return BeautifulSoup(resp.text, "lxml")
    except Exception as exc:
        print(f"  [WARN] Failed to fetch {url}: {exc}")
        return None


def _extract_section_name(soup: BeautifulSoup, url: str) -> str:
    """Infer section name from page title or URL path."""
    title_tag = soup.find("title")
    if title_tag:
        title = _clean_text(title_tag.get_text())
        # Strip site name suffix
        for sep in [" | ", " - ", " – "]:
            if sep in title:
                title = title.split(sep)[0].strip()
        if title:
            return title
    # Fallback: derive from URL path
    path = urlparse(url).path.rstrip("/").split("/")[-1]
    return path.replace("-", " ").title()


def _extract_main_content(soup: BeautifulSoup) -> BeautifulSoup | None:
    """Find the main content area, excluding nav/header/footer."""
    # Try common content selectors
    for selector in [
        "article", "main", ".content", "#content",
        ".post-content", ".entry-content", ".page-content",
        ".col-content", "#col-content", ".main-content",
    ]:
        el = soup.select_one(selector)
        if el:
            return el
    # Fallback: body minus nav/header/footer/sidebar
    body = soup.find("body")
    if body:
        for tag in body.find_all(["nav", "header", "footer", "aside", "script", "style"]):
            tag.decompose()
        return body
    return None


def _chunk_by_headings(content: BeautifulSoup, section: str) -> list[dict]:
    """Split content into chunks by heading tags."""
    chunks = []
    current_heading = ""
    current_texts: list[str] = []

    def flush(heading: str, texts: list[str]) -> None:
        body = _clean_text(" ".join(texts))
        words = body.split()
        if len(words) >= MIN_WORDS:
            chunk_text = f"{heading}\n{body}" if heading else body
            chunks.append({
                "section": section,
                "subsection": heading,
                "content": chunk_text,
                "word_count": len(words),
            })

    for el in content.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        tag = el.name
        text = _clean_text(el.get_text())
        if not text:
            continue

        if tag in ("h1", "h2", "h3", "h4"):
            # Save previous chunk before starting new heading
            flush(current_heading, current_texts)
            current_heading = text
            current_texts = []
        else:
            current_texts.append(text)

    # Save last accumulated chunk
    flush(current_heading, current_texts)
    return chunks


def _collect_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Collect all same-domain links from the page."""
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        parsed = urlparse(href)
        # Keep only http(s) links on the same domain, no fragments
        if parsed.scheme in ("http", "https") and _is_same_domain(href):
            clean = href.split("#")[0]
            if clean not in links:
                links.append(clean)
    return links


# ---------------------------------------------------------------------------
# Main crawler
# ---------------------------------------------------------------------------

def crawl(seed_urls: list[str], delay: float = 1.0) -> list[dict]:
    """BFS crawl from seed_urls up to MAX_DEPTH levels deep."""
    visited: set[str] = set()
    all_chunks: list[dict] = []
    session = requests.Session()

    queue = [(url, 0) for url in seed_urls]

    while queue:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        print(f"  [depth={depth}] Crawling: {url}")
        soup = _fetch(url, session)
        if soup is None:
            continue

        section = _extract_section_name(soup, url)
        content = _extract_main_content(soup)

        if content:
            chunks = _chunk_by_headings(content, section)
            print(f"    → Extracted {len(chunks)} chunks for section '{section}'")
            all_chunks.extend(chunks)
        else:
            print(f"    → No main content found")

        # Enqueue child links if within depth limit
        if depth < MAX_DEPTH:
            links = _collect_links(soup, url)
            for link in links:
                if link not in visited:
                    queue.append((link, depth + 1))

        time.sleep(delay)

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Crawl HAUI SICT website")
    parser.add_argument(
        "--output", default="data/web_chunks.json",
        help="Output JSON file path (default: data/web_chunks.json)"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Delay in seconds between requests (default: 1.0)"
    )
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    print(f"Starting crawl of {len(SEED_URLS)} seed URLs...")
    raw_chunks = crawl(SEED_URLS, delay=args.delay)
    print(f"\nTotal raw chunks collected: {len(raw_chunks)}")

    # Convert to chunk_folder.json format
    output = []
    for i, chunk in enumerate(raw_chunks, start=1):
        output.append({
            "chunk_id": i,
            "content": chunk["content"],
            "metadata": {
                "#": chunk["section"],
                "###": chunk.get("subsection", ""),
                "word_count": chunk["word_count"],
                "source": "website",
            },
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(output)} chunks to '{args.output}'")
    print("\nSection summary:")
    from collections import Counter
    sections = Counter(c["metadata"]["#"] for c in output)
    for section, count in sections.most_common():
        print(f"  {section}: {count} chunks")


if __name__ == "__main__":
    main()
