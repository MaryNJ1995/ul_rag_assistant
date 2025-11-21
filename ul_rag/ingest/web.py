from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import httpx
from bs4 import BeautifulSoup

from ul_rag_assistant.ul_rag.logging import get_logger

log = get_logger(__name__)


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def _fetch_url(client: httpx.Client, url: str) -> Dict[str, str] | None:
    try:
        r = client.get(url, timeout=20.0)
        r.raise_for_status()
    except Exception as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return None

    text = _extract_text(r.text)
    title = url
    try:
        soup = BeautifulSoup(r.text, "html.parser")
        t = soup.find("title")
        if t and t.text:
            title = t.text.strip()
    except Exception:
        pass

    return {"url": url, "title": title, "text": text}


def fetch_ul_pages(seeds_path: str, out_jsonl: str) -> None:
    """Fetch a list of UL pages from a seeds JSONL file.

    Each line in `seeds_path` should be a JSON object with at least:
      {"url": "https://www.ul.ie/..."}
    """
    seeds: List[str] = []
    for line in Path(seeds_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        url = obj.get("url")
        if url:
            seeds.append(url)

    log.info(f"Fetching {len(seeds)} UL pages from seeds")

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(follow_redirects=True) as client, out_path.open("w", encoding="utf-8") as f_out:
        for i, url in enumerate(seeds, start=1):
            log.info(f"[{i}/{len(seeds)}] Fetching {url}")
            doc = _fetch_url(client, url)
            if not doc:
                continue
            f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")

    log.info(f"Wrote corpus JSONL to {out_jsonl}")



import json
import time
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from tqdm import tqdm
def is_ul_url(url: str) -> bool:
    # Only follow URLs that belong to UL or whitelisted domains
    parsed = urlparse(url)
    return parsed.netloc.endswith("ul.ie") or parsed.netloc.endswith("lero.ie") or parsed.netloc.endswith("pure.ul.ie")

def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    return text

def crawl_ul(seeds_path: str, out_path: str, max_depth: int = 3, max_pages: int = 2000, delay: float = 1.0):
    # 1. Load seeds
    seeds = []
    with open(seeds_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            url = obj.get("url")
            if url:
                seeds.append(url)

    visited = set()
    queue = [(url, 0) for url in seeds]  # (url, depth)
    n_pages = 0

    client = httpx.Client(timeout=20.0, follow_redirects=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        while queue and n_pages < max_pages:
            url, depth = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            if depth > max_depth:
                continue
            if not is_ul_url(url):
                continue

            try:
                resp = client.get(url)
            except Exception as e:
                print(f"[WARN] Fetch error {url}: {e}")
                continue

            if resp.status_code != 200:
                print(f"[WARN] Non-200 for {url}: {resp.status_code}")
                continue

            html = resp.text
            text = clean_html(html)
            if not text:
                continue

            # Simple title grab
            title = url
            soup = BeautifulSoup(html, "html.parser")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()

            doc = {"url": url, "title": title, "text": text}
            out_f.write(json.dumps(doc) + "\n")
            n_pages += 1
            print(f"[INFO] Crawled ({n_pages}) {url} depth={depth}")

            # Discover new links
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                new_url = urljoin(url, href)
                if new_url not in visited and is_ul_url(new_url):
                    queue.append((new_url, depth + 1))

            time.sleep(delay)

    client.close()
    print(f"[INFO] Done. Total pages written: {n_pages}")
