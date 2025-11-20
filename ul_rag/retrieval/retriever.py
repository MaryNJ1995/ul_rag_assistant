from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ul_rag_assistant.ul_rag.config import get_settings
from ul_rag_assistant.ul_rag.logging import get_logger
from ul_rag_assistant.ul_rag.retrieval.rerank import Reranker

log = get_logger(__name__)


class Retriever:
    def __init__(self):
        settings = get_settings()
        self.index_path = Path(settings.index_path)
        self.embed_model_name = settings.embed_model
        self._load_index()
        self._emb_model: SentenceTransformer | None = None
        self.reranker = Reranker()

    def _load_index(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found at {self.index_path}. Run ingestion first.")
        with self.index_path.open("rb") as f:
            data = pickle.load(f)
        self.texts: List[str] = data["texts"]
        self.metas: List[Dict[str, Any]] = data["metas"]
        self.embeddings: np.ndarray = data["embeddings"]
        self.bm25: BM25Okapi = data["bm25"]

    @property
    def emb_model(self) -> SentenceTransformer:
        if self._emb_model is None:
            self._emb_model = SentenceTransformer(self.embed_model_name)
        return self._emb_model

    def _dense_search(self, query: str, top_k: int = 20) -> List[Tuple[str, Dict[str, Any], float]]:
        q_emb = self.emb_model.encode([query], convert_to_numpy=True)[0]
        sims = np.dot(self.embeddings, q_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
        )
        idxs = np.argsort(-sims)[:top_k]
        results = []
        for i in idxs:
            i = int(i)
            results.append((self.texts[i], self.metas[i], float(sims[i])))
        return results

    def _sparse_search(self, query: str, top_k: int = 20) -> List[Tuple[str, Dict[str, Any], float]]:
        scores = self.bm25.get_scores(query.split())
        idxs = np.argsort(-scores)[:top_k]
        results = []
        for i in idxs:
            i = int(i)
            results.append((self.texts[i], self.metas[i], float(scores[i])))
        return results

    def retrieve(self, query: str, max_chunks: int = 8) -> List[Dict[str, Any]]:
        dense = self._dense_search(query, top_k=max_chunks * 3)
        sparse = self._sparse_search(query, top_k=max_chunks * 3)

        def rrf_rank(items):
            return { id(meta): rank for rank, (_, meta, _) in enumerate(items, start=1) }

        dense_map = rrf_rank(dense)
        sparse_map = rrf_rank(sparse)

        combined = {}
        for (text, meta, _) in dense + sparse:
            key = id(meta)
            r1 = dense_map.get(key)
            r2 = sparse_map.get(key)
            score = 0.0
            if r1 is not None:
                score += 1.0 / (60 + r1)
            if r2 is not None:
                score += 1.0 / (60 + r2)
            if key not in combined or combined[key][2] < score:
                combined[key] = (text, meta, score)

        fused = list(combined.values())
        fused.sort(key=lambda x: x[2], reverse=True)
        fused = fused[: max_chunks * 2]

        reranked = self.reranker.rerank(query, [(t, m) for (t, m, _) in fused])
        top = reranked[:max_chunks]

        docs = []
        for i, (t, m, score) in enumerate(top, start=1):
            docs.append({"text": t, "meta": m, "score": score, "rank": i})
        return docs
