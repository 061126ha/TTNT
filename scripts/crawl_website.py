"""
Web crawler for HAUI SICT website (https://sict.haui.edu.vn/vn/).
"""

import argparse
import json
import re
import time
from collections import Counter
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


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


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _is_same_domain(url: str) -> bool:
    return urlparse(url).netloc == urlparse(BASE_URL).netloc


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def _fetch(url: str, session: requests.Session):
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        return BeautifulSoup(resp.text, "lxml")
    except Exception as exc:
        print(f"[WARN] Fetch failed: {url} | {exc}")
        return None


def _extract_section_name(soup: BeautifulSoup, url: str) -> str:
    title = soup.title.get_text(strip=True) if soup.title else ""
    title = _clean_text(title)

    for sep in [" | ", " - ", " – "]:
        if sep in title:
            title = title.split(sep)[0]

    if title:
        return title

    path = urlparse(url).path.strip("/").split("/")[-1]
    return path.replace("-", " ").title()


def _extract_main_content(soup: BeautifulSoup):
    for selector in [
        "article", "main",
        ".content", "#content",
        ".post-content", ".entry-content",
        ".page-content", ".main-content"
    ]:
        el = soup.select_one(selector)
        if el:
            return el

    body = soup.body
    if body:
        for tag in body.find_all(["nav", "header", "footer", "aside", "script", "style"]):
            tag.decompose()
        return body

    return None


def _chunk_by_headings(content: BeautifulSoup, section: str):
    chunks = []
    current_heading = ""
    texts = []

    def flush():
        body = _clean_text(" ".join(texts))
        if len(body.split()) >= MIN_WORDS:
            chunks.append({
                "section": section,
                "subsection": current_heading,
                "content": f"{current_heading}\n{body}" if current_heading else body,
                "word_count": len(body.split()),
            })

    for el in content.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = _clean_text(el.get_text())
        if not text:
            continue

        if el.name in ["h1", "h2", "h3", "h4"]:
            flush()
            current_heading = text
            texts = []
        else:
            texts.append(text)

    flush()
    return chunks


def _collect_links(soup: BeautifulSoup, base_url: str):
    links = set()

    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        parsed = urlparse(href)

        if (
            parsed.scheme in ("http", "https")
            and _is_same_domain(href)
        ):
            clean = href.split("#")[0]
            links.add(clean)

    return list(links)


# ─────────────────────────────────────────────
# CRAWLER
# ─────────────────────────────────────────────
def crawl(seed_urls, delay=1.0):
    visited = set()
    results = []
    session = requests.Session()

    queue = [(u, 0) for u in seed_urls]

    while queue:
        url, depth = queue.pop(0)

        if url in visited:
            continue

        visited.add(url)

        print(f"[depth={depth}] {url}")

        soup = _fetch(url, session)
        if not soup:
            continue

        section = _extract_section_name(soup, url)
        content = _extract_main_content(soup)

        if content:
            chunks = _chunk_by_headings(content, section)
            print(f"  → chunks: {len(chunks)}")
            results.extend(chunks)

        if depth < MAX_DEPTH:
            for link in _collect_links(soup, url):
                if link not in visited:
                    queue.append((link, depth + 1))

        time.sleep(delay)

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/web_chunks.json")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Start crawling {len(SEED_URLS)} seeds...")

    raw = crawl(SEED_URLS, delay=args.delay)

    print(f"Total chunks: {len(raw)}")

    output = []

    for i, c in enumerate(raw, start=1):
        output.append({
            "chunk_id": i,
            "content": c["content"],
            "metadata": {
                "#": c["section"],
                "###": c["subsection"],
                "word_count": c["word_count"],
                "source": "website"
            }
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved → {args.output}")

    print("\nSection summary:")
    counter = Counter(x["metadata"]["#"] for x in output)
    for k, v in counter.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()