# ul_rag/retrieval/rerank.py
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional

from sentence_transformers import CrossEncoder

from ..config import get_settings
from ..logging import get_logger

log = get_logger(__name__)


class Reranker:
    def __init__(self, model_name: Optional[str] = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.rerank_model
        log.info(f"Loading rerank model {self.model_name}")
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        docs: List[Tuple[str, Dict[str, Any]]],
        domain_hint: Optional[str] = None,
        bias: float = 0.2,
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """
        Returns list of (score, text, meta) sorted by score desc.
        """
        if not docs:
            return []

        pairs = [(query, text) for (text, _) in docs]
        scores = self.model.predict(pairs)

        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for (text, meta), s in zip(docs, scores):
            s_adj = float(s)
            if domain_hint:
                host = (meta.get("source_host") or meta.get("host") or "").lower()
                url = (meta.get("source_url") or meta.get("path") or "").lower()
                if domain_hint.lower() in host or domain_hint.lower() in url:
                    s_adj += bias
            scored.append((s_adj, text, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored
