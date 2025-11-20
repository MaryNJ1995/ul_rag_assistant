from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from ul_rag_assistant.ul_rag.config import get_settings
from ul_rag_assistant.ul_rag.logging import get_logger

log = get_logger(__name__)


def _simple_chunk(text: str, max_tokens: int = 200) -> List[str]:
    """Very simple whitespace-based chunker."""
    words = text.split()
    chunks: List[str] = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i+max_tokens])
        if chunk:
            chunks.append(chunk)
    return chunks


def build_index_from_jsonl(input_jsonl: str, index_path: Optional[str] = None) -> None:
    """Build BM25 + dense embeddings index from a JSONL corpus.

    Each line in `input_jsonl` must contain:
      {"url": ..., "title": ..., "text": ...}
    """
    settings = get_settings()
    embed_model_name = settings.embed_model
    if index_path is None:
        index_path = settings.index_path

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for line in Path(input_jsonl).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        url = obj.get("url")
        title = obj.get("title") or url
        text = obj.get("text") or ""
        for chunk in _simple_chunk(text):
            texts.append(chunk)
            metas.append({"source_url": url, "title": title})

    log.info(f"Building index from {len(texts)} chunks")

    model = SentenceTransformer(embed_model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    tokenized_corpus = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    index = {
        "texts": texts,
        "metas": metas,
        "embeddings": embeddings,
        "bm25": bm25,
        "embed_model": embed_model_name,
    }

    idx_path = Path(index_path)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    with idx_path.open("wb") as f:
        pickle.dump(index, f)

    log.info(f"Saved index to {idx_path}")
