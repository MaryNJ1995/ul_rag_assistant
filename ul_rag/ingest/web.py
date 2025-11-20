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
