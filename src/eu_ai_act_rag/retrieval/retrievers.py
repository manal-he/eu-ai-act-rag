"""Sparse, dense and hybrid retrievers sharing one interface."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Literal, Protocol

import numpy as np

from eu_ai_act_rag.embeddings.providers import Embedder
from eu_ai_act_rag.retrieval.vectorstore import InMemoryVectorStore, VectorStore
from eu_ai_act_rag.schemas import Chunk, Lang, SearchHit
from eu_ai_act_rag.text import tokenize


class Retriever(Protocol):
    name: str

    def index(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, k: int, lang: Lang | None = None) -> list[SearchHit]: ...


def _ranked(pairs: list[tuple[Chunk, float]], source: str) -> list[SearchHit]:
    return [SearchHit(chunk=c, score=s, rank=i, source=source) for i, (c, s) in enumerate(pairs, 1)]


class BM25Retriever:
    """Okapi BM25 over an inverted index (no external dependency).

    Legal paragraphs vary from one line to two pages, so length normalisation (``b``) matters;
    the defaults k1=1.2, b=0.75 were kept rather than tuned on the golden set to avoid
    overfitting a 1-person evaluation set.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._len = np.zeros(0)
        self._langs = np.array([], dtype=object)
        self._avgdl = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self._postings = defaultdict(list)
        lens = []
        for i, c in enumerate(self.chunks):
            tf = Counter(tokenize(c.index_text))
            lens.append(sum(tf.values()))
            for term, n in tf.items():
                self._postings[term].append((i, n))
        self._len = np.asarray(lens, dtype=np.float64)
        self._avgdl = float(self._len.mean()) if lens else 0.0
        self._langs = np.array([c.lang for c in self.chunks], dtype=object)

    def scores(self, query: str) -> np.ndarray:
        n = len(self.chunks)
        scores = np.zeros(n)
        if not n:
            return scores
        norm = self.k1 * (1 - self.b + self.b * self._len / max(self._avgdl, 1e-9))
        for term in set(tokenize(query)):
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            idx = np.fromiter((p[0] for p in posting), dtype=np.int64, count=len(posting))
            tf = np.fromiter((p[1] for p in posting), dtype=np.float64, count=len(posting))
            scores[idx] += idf * tf * (self.k1 + 1) / (tf + norm[idx])
        return scores

    def search(self, query: str, k: int, lang: Lang | None = None) -> list[SearchHit]:
        scores = self.scores(query)
        if lang is not None and len(scores):
            scores = np.where(self._langs == lang, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:k]
        pairs = [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]
        return _ranked(pairs, self.name)


class DenseRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore | None = None) -> None:
        self.embedder = embedder
        self.store: VectorStore = store if store is not None else InMemoryVectorStore()
        self.name = f"dense[{embedder.name}]"

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = self.embedder.embed_documents([c.index_text for c in chunks])
        self.store.add(chunks, vecs)

    def search(self, query: str, k: int, lang: Lang | None = None) -> list[SearchHit]:
        return _ranked(self.store.search(self.embedder.embed_query(query), k, lang), self.name)


def reciprocal_rank_fusion(lists: list[list[SearchHit]], k: int = 60) -> list[SearchHit]:
    """score(d) = sum_i 1 / (k + rank_i(d)). Uses ranks only, so BM25 scores (unbounded) and
    cosine similarities (bounded) never need to be calibrated against each other."""
    fused: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}
    for hits in lists:
        for h in hits:
            fused[h.chunk.chunk_id] += 1.0 / (k + h.rank)
            chunks[h.chunk.chunk_id] = h.chunk
    order = sorted(fused, key=lambda cid: -fused[cid])
    return _ranked([(chunks[c], fused[c]) for c in order], "rrf")


def weighted_fusion(
    sparse: list[SearchHit], dense: list[SearchHit], alpha: float
) -> list[SearchHit]:
    """alpha * minmax(dense) + (1 - alpha) * minmax(sparse); a document absent from a list
    gets 0 for that list. Sensitive to score distributions - kept as an ablation baseline."""

    def norm(hits: list[SearchHit]) -> dict[str, float]:
        if not hits:
            return {}
        s = [h.score for h in hits]
        lo, hi = min(s), max(s)
        return {h.chunk.chunk_id: (h.score - lo) / (hi - lo) if hi > lo else 1.0 for h in hits}

    ns, nd = norm(sparse), norm(dense)
    chunks = {h.chunk.chunk_id: h.chunk for h in [*sparse, *dense]}
    fused = {cid: alpha * nd.get(cid, 0.0) + (1 - alpha) * ns.get(cid, 0.0) for cid in chunks}
    order = sorted(fused, key=lambda cid: -fused[cid])
    return _ranked([(chunks[c], fused[c]) for c in order], "weighted")


class HybridRetriever:
    def __init__(
        self,
        sparse: Retriever,
        dense: Retriever,
        fusion: Literal["rrf", "weighted"] = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
        candidate_k: int = 40,
    ) -> None:
        self.sparse, self.dense = sparse, dense
        self.fusion, self.rrf_k, self.alpha, self.candidate_k = fusion, rrf_k, alpha, candidate_k
        self.name = f"hybrid-{fusion}[{sparse.name}+{dense.name}]"

    def index(self, chunks: list[Chunk]) -> None:
        self.sparse.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, k: int, lang: Lang | None = None) -> list[SearchHit]:
        n = max(k, self.candidate_k)
        s = self.sparse.search(query, n, lang)
        d = self.dense.search(query, n, lang)
        fused = (
            reciprocal_rank_fusion([s, d], self.rrf_k)
            if self.fusion == "rrf"
            else weighted_fusion(s, d, self.alpha)
        )
        return fused[:k]
