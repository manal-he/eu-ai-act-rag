from __future__ import annotations

import numpy as np
import pytest

from eu_ai_act_rag.chunking.structure import StructureAwareChunker
from eu_ai_act_rag.embeddings.providers import CachedEmbedder, HashingEmbedder
from eu_ai_act_rag.evaluation.metrics import unit_ranking
from eu_ai_act_rag.retrieval.retrievers import (
    BM25Retriever,
    DenseRetriever,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from eu_ai_act_rag.retrieval.vectorstore import InMemoryVectorStore
from eu_ai_act_rag.schemas import Chunk, LegalUnit, SearchHit


def _hit(cid: str, rank: int, score: float = 1.0) -> SearchHit:
    c = Chunk(chunk_id=cid, unit_id=f"gdpr:art_{cid}", regulation="gdpr", lang="en", text=cid)
    return SearchHit(chunk=c, score=score, rank=rank)


@pytest.fixture(scope="module")
def bm25(units: list[LegalUnit]) -> BM25Retriever:
    r = BM25Retriever()
    r.index(StructureAwareChunker().chunk(units))
    return r


def test_bm25_finds_breach_notification(bm25: BM25Retriever) -> None:
    hits = bm25.search(
        "notify a personal data breach to the supervisory authority 72 hours", 5, "en"
    )
    assert unit_ranking(hits)[0] == "gdpr:art_33"
    assert all(h.chunk.lang == "en" for h in hits)


def test_bm25_language_filter_french(bm25: BM25Retriever) -> None:
    hits = bm25.search("violation de données notification autorité de contrôle", 5, "fr")
    assert hits and all(h.chunk.lang == "fr" for h in hits)
    assert "gdpr:art_33" in unit_ranking(hits)


def test_rrf_rewards_agreement() -> None:
    a = [_hit("1", 1), _hit("2", 2), _hit("3", 3)]
    b = [_hit("3", 1), _hit("1", 2), _hit("4", 3)]
    fused = reciprocal_rank_fusion([a, b], k=60)
    assert [h.chunk.chunk_id for h in fused[:2]] == ["1", "3"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_weighted_fusion_minmax() -> None:
    sparse = [_hit("1", 1, 10.0), _hit("2", 2, 5.0)]
    dense = [_hit("2", 1, 0.9), _hit("3", 2, 0.1)]
    fused = weighted_fusion(sparse, dense, alpha=0.7)
    # minmax: sparse {1: 1, 2: 0}, dense {2: 1, 3: 0} -> 2: 0.7, 1: 0.3, 3: 0.0
    assert [h.chunk.chunk_id for h in fused] == ["2", "1", "3"]
    assert fused[0].score == pytest.approx(0.7)


def test_dense_hashing_and_store(units: list[LegalUnit], tmp_path: object) -> None:
    chunks = StructureAwareChunker().chunk(units)
    dense = DenseRetriever(HashingEmbedder(), InMemoryVectorStore())
    dense.index(chunks)
    hits = dense.search("right to data portability machine-readable format", 5, "en")
    assert "gdpr:art_20" in unit_ranking(hits)


def test_cached_embedder_roundtrip(tmp_path: object) -> None:
    from pathlib import Path

    inner = HashingEmbedder(dim=64)
    cache = CachedEmbedder(inner, Path(str(tmp_path)))
    v1 = cache.embed_documents(["hello world", "bonjour"])
    v2 = CachedEmbedder(inner, Path(str(tmp_path))).embed_documents(["bonjour"])
    assert np.allclose(v1[1], v2[0])
