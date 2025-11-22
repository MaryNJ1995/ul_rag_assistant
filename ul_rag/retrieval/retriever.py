# ul_rag/retrieval/retriever.py
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ..config import get_settings
from ..logging import get_logger
from .rerank import Reranker

log = get_logger(__name__)


@dataclass
class IndexedCorpus:
    texts: List[str]
    metas: List[Dict[str, Any]]
    embeddings: np.ndarray  # shape: (N, d)
    bm25: BM25Okapi


class Retriever:
    def __init__(self, index_path: Optional[Path] = None) -> None:
        self.settings = get_settings()
        self.index_path = index_path or self.settings.index_path

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Index file not found at {self.index_path}. "
                f"Run the index builder script first."
            )

        self.corpus: IndexedCorpus = self._load_index(self.index_path)
        self.embed_model = SentenceTransformer(self.settings.embed_model)
        self.reranker = Reranker(self.settings.rerank_model)

        log.info(
            f"Retriever initialised with {len(self.corpus.texts)} chunks "
            f"and model {self.settings.embed_model}"
        )

    def _load_index(self, path: Path) -> IndexedCorpus:
        log.info(f"Loading index from {path}")
        with path.open("rb") as f:
            data = pickle.load(f)

        texts: List[str] = data["texts"]
        metas: List[Dict[str, Any]] = data["metas"]
        embeddings: np.ndarray = data["embeddings"]
        bm25: BM25Okapi = data["bm25"]

        if embeddings.shape[0] != len(texts):
            raise ValueError(
                f"Embeddings count {embeddings.shape[0]} "
                f"!= texts count {len(texts)}"
            )

        log.info(
            f"Index stats: {len(texts)} chunks, emb_dim={embeddings.shape[1]}"
        )
        return IndexedCorpus(texts=texts, metas=metas, embeddings=embeddings, bm25=bm25)

    # --- core retrieval ---

    def _dense_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        q_emb = self.embed_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        docs_emb = self.corpus.embeddings
        scores = docs_emb @ q_emb  # cosine if normalised
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idx]

    def _sparse_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        tokens = query.lower().split()
        scores = self.corpus.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idx]

    def _rrf_fuse(
        self,
        dense: List[Tuple[int, float]],
        sparse: List[Tuple[int, float]],
        k_rrf: int = 60,
    ) -> List[int]:
        """
        Reciprocal Rank Fusion over dense + sparse results.
        Returns a list of doc indices sorted by fused score.
        """
        ranks: Dict[int, float] = {}
        for rank, (idx, _) in enumerate(dense):
            ranks[idx] = ranks.get(idx, 0.0) + 1.0 / (k_rrf + rank)
        for rank, (idx, _) in enumerate(sparse):
            ranks[idx] = ranks.get(idx, 0.0) + 1.0 / (k_rrf + rank)

        fused = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in fused]

    def retrieve(
        self,
        query: str,
        max_chunks: int = 6,
        domain_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full pipeline: dense + BM25 + RRF + cross-encoder rerank.
        """
        if not query.strip():
            return []

        # 1. dense + sparse
        k_candidates = max_chunks * 8
        dense = self._dense_search(query, k=k_candidates)
        sparse = self._sparse_search(query, k=k_candidates)

        # 2. fuse
        fused_idx = self._rrf_fuse(dense, sparse)
        fused_idx = fused_idx[: k_candidates]

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for idx in fused_idx:
            text = self.corpus.texts[idx]
            meta = self.corpus.metas[idx]
            candidates.append((text, meta))

        # 3. rerank with domain bias
        reranked = self.reranker.rerank(
            query=query,
            docs=candidates,
            # domain_hint=domain_hint,
        )

        top = reranked[:max_chunks]
        docs = []
        for i, (score, text, meta) in enumerate(top, start=1):
            docs.append(
                {
                    "text": text,
                    "meta": meta,
                    "score": float(score),
                    "rank": i,
                }
            )
        log.debug(f"Retriever: returned {len(docs)} docs for query={query!r}")
        return docs
