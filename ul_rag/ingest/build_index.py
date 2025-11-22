#!/usr/bin/env python
import argparse
import json
import os
from glob import glob
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

from ul_rag_assistant.ul_rag.config import get_settings
from ul_rag_assistant.ul_rag.logging import get_logger

log = get_logger(__name__)

settings = get_settings()
def load_jsonl_corpus(path: str) -> List[Dict[str, Any]]:
    docs = []
    if not os.path.exists(path):
        log.warning(f"JSONL input not found at {path}, skipping.")
        return docs

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "").strip()
            if not text:
                continue
            docs.append(
                {
                    "text": text,
                    "meta": {
                        "source_url": obj.get("url"),
                        "title": obj.get("title"),
                        "source": "web",
                    },
                }
            )
    log.info(f"Loaded {len(docs)} web docs from JSONL.")
    return docs


def load_md_dir(md_dir: str) -> List[Dict[str, Any]]:
    docs = []
    if not md_dir or not os.path.isdir(md_dir):
        log.info(f"MD directory {md_dir} not found or not a directory; skipping.")
        return docs

    md_paths = glob(os.path.join(md_dir, "**", "*.md"), recursive=True)
    for path in md_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception as e:
            log.warning(f"Failed to read MD file {path}: {e}")
            continue
        if not text:
            continue
        docs.append(
            {
                "text": text,
                "meta": {
                    "path": path,
                    "title": os.path.basename(path),
                    "source": "md",
                },
            }
        )
    log.info(f"Loaded {len(docs)} Markdown docs from {md_dir}.")
    return docs


def extract_text_from_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        pages_text = []
        for page in reader.pages:
            t = page.extract_text() or ""
            pages_text.append(t)
        text = "\n".join(pages_text)
        return " ".join(text.split())
    except Exception as e:
        log.warning(f"Failed to extract PDF text from {path}: {e}")
        return ""


def load_pdf_dir(pdf_dir: str) -> List[Dict[str, Any]]:
    docs = []
    if not pdf_dir or not os.path.isdir(pdf_dir):
        log.info(f"PDF directory {pdf_dir} not found or not a directory; skipping.")
        return docs

    pdf_paths = glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True)
    for path in pdf_paths:
        text = extract_text_from_pdf(path)
        if not text:
            continue
        docs.append(
            {
                "text": text,
                "meta": {
                    "path": path,
                    "title": os.path.basename(path),
                    "source": "pdf",
                },
            }
        )
    log.info(f"Loaded {len(docs)} PDF docs from {pdf_dir}.")
    return docs


def simple_chunk(text: str, max_tokens: int = 200) -> List[str]:
    """
    Very simple whitespace-based chunking.
    You can later replace this with a token-based or sentence-based splitter.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk_words = words[i : i + max_tokens]
        chunks.append(" ".join(chunk_words))
    return chunks


def build_index(
    input_jsonl: str,
    index_path: str,
    md_dir: str = None,
    pdf_dir: str = None,
    embed_model_name: str = None,
):
    # 1. Load all docs: web JSONL + MD + PDF
    all_docs: List[Dict[str, Any]] = []

    web_docs = load_jsonl_corpus(input_jsonl)
    all_docs.extend(web_docs)

    md_docs = load_md_dir(md_dir) if md_dir else []
    all_docs.extend(md_docs)

    pdf_docs = load_pdf_dir(pdf_dir) if pdf_dir else []
    all_docs.extend(pdf_docs)

    if not all_docs:
        log.error("No documents loaded! Aborting index build.")
        return

    log.info(f"Total raw documents before chunking: {len(all_docs)}")

    # 2. Chunk all docs
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    seen_chunks = set()  # for exact dedup based on normalized text

    for doc in all_docs:
        text = doc["text"]
        meta = doc.get("meta", {})
        chunks = simple_chunk(text, max_tokens=200)
        for ch in chunks:
            norm = " ".join(ch.split())  # normalize whitespace
            if not norm:
                continue
            if norm in seen_chunks:
                continue  # skip exact duplicate chunk
            seen_chunks.add(norm)
            texts.append(ch)
            metas.append(meta)

    log.info(f"Total chunks to index: {len(texts)}")

    # 3. Build embeddings
    embed_model_name = embed_model_name or settings.embed_model
    log.info(f"Loading embedding model: {embed_model_name}")
    embed_model = SentenceTransformer(embed_model_name)

    log.info("Encoding embeddings...")
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # 4. Build BM25 index
    tokenized_corpus = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # 5. Save index as pickle
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    index_obj = {
        "texts": texts,
        "metas": metas,
        "embeddings": embeddings,
        "bm25": bm25,
        "embed_model": embed_model_name,
    }

    import pickle

    with open(index_path, "wb") as f:
        pickle.dump(index_obj, f)

    log.info(f"Index saved to {index_path}.")
    log.info(
        f"Index stats: {len(texts)} chunks, {len(all_docs)} source docs "
        f"(web={len(web_docs)}, md={len(md_docs)}, pdf={len(pdf_docs)})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="/home/maryam_najafi/ul_bot/ul_rag_assistant/data/ul/ul_data.jsonl",
                        help="Input JSONL corpus.")
    parser.add_argument("--index_path", type=str,
                        default="/home/maryam_najafi/ul_bot/ul_rag_assistant/storage/index/ul_index.pkl",
                        help="Where to store index pickle (overrides INDEX_PATH env).")
    parser.add_argument("--md_dir", type=str, default="/home/maryam_najafi/ul_bot/ul_rag_assistant/data/ul/md", help="Directory with .md files to include.")
    parser.add_argument("--pdf_dir", type=str, default="/home/maryam_najafi/ul_bot/ul_rag_assistant/data/ul/pdf", help="Directory with .pdf files to include.")
    args = parser.parse_args()

    build_index(
        input_jsonl=args.input,
        index_path=args.index_path,
        md_dir=args.md_dir,
        pdf_dir=args.pdf_dir,
        embed_model_name=settings.embed_model,
    )




if __name__ == "__main__":
    main()
