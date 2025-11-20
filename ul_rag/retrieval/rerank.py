from __future__ import annotations

import os
from typing import List, Tuple

from sentence_transformers import CrossEncoder

from ul_rag_assistant.ul_rag.logging import get_logger

log = get_logger(__name__)


class Reranker:
    """Cross-encoder reranker."""

    def __init__(self):
        model_name = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        log.info(f"Loading rerank model {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Tuple[str, dict]]) -> List[Tuple[str, dict, float]]:
        if not docs:
            return []
        pairs = [(query, d[0]) for d in docs]
        scores = self.model.predict(pairs)
        out = []
        for (text, meta), score in zip(docs, scores):
            out.append((text, meta, float(score)))
        out.sort(key=lambda x: x[2], reverse=True)
        return out
