"""Second-stage rerankers. A cross-encoder reads (query, passage) jointly, which is far more
accurate than bi-encoder similarity but costs one forward pass per candidate - hence it only
sees the top ``candidate_k`` of the first stage. Latency is measured in the ablation."""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from eu_ai_act_rag.config import Settings
from eu_ai_act_rag.llm.base import ProviderNotConfigured
from eu_ai_act_rag.schemas import SearchHit


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]: ...


def _reorder(hits: list[SearchHit], scores: list[float], top_n: int, name: str) -> list[SearchHit]:
    order = sorted(range(len(hits)), key=lambda i: -scores[i])[:top_n]
    return [
        SearchHit(chunk=hits[i].chunk, score=float(scores[i]), rank=r, source=name)
        for r, i in enumerate(order, 1)
    ]


class NoopReranker:
    name = "none"

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        return hits[:top_n]


class LocalCrossEncoderReranker:
    """sentence-transformers CrossEncoder. Default: a multilingual MiniLM trained on mMARCO
    (handles FR and EN). ``BAAI/bge-reranker-v2-m3`` is stronger but ~5x slower on CPU."""

    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ProviderNotConfigured("pip install 'eu-ai-act-rag[local]'") from exc
        self._model = CrossEncoder(model, max_length=512)
        self.name = f"local:{model}"

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        if not hits:
            return []
        scores = self._model.predict([(query, h.chunk.index_text) for h in hits], batch_size=32)
        return _reorder(hits, [float(s) for s in scores], top_n, self.name)


class CohereReranker:  # pragma: no cover - requires a key
    def __init__(self, api_key: str, model: str = "rerank-multilingual-v3.0") -> None:
        if not api_key:
            raise ProviderNotConfigured("COHERE_API_KEY is not set")
        self._client = httpx.Client(
            base_url="https://api.cohere.com",
            timeout=30,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.model = model
        self.name = f"cohere:{model}"

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        r = self._client.post(
            "/v2/rerank",
            json={
                "model": self.model,
                "query": query,
                "documents": [h.chunk.index_text for h in hits],
                "top_n": len(hits),
            },
        )
        r.raise_for_status()
        scores = [0.0] * len(hits)
        for res in r.json()["results"]:
            scores[res["index"]] = res["relevance_score"]
        return _reorder(hits, scores, top_n, self.name)


class VertexRankingReranker:  # pragma: no cover - requires GCP
    """Vertex AI Search ranking API (Discovery Engine ``rankingConfigs:rank``)."""

    def __init__(self, project: str, model: str = "semantic-ranker-default@latest") -> None:
        try:
            import google.auth
            from google.auth.transport.requests import Request
        except ImportError as exc:
            raise ProviderNotConfigured("pip install 'eu-ai-act-rag[vertex]'") from exc
        if not project:
            raise ProviderNotConfigured("GOOGLE_CLOUD_PROJECT is not set")
        self._creds: Any
        self._creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self._request = Request
        self.url = (
            f"https://discoveryengine.googleapis.com/v1/projects/{project}/locations/global/"
            "rankingConfigs/default_ranking_config:rank"
        )
        self.model = model
        self.name = f"vertex:{model}"

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        self._creds.refresh(self._request())
        r = httpx.post(
            self.url,
            timeout=30,
            headers={"Authorization": f"Bearer {self._creds.token}"},
            json={
                "model": self.model,
                "query": query,
                "topN": len(hits),
                "records": [
                    {"id": str(i), "content": h.chunk.index_text} for i, h in enumerate(hits)
                ],
            },
        )
        r.raise_for_status()
        scores = [0.0] * len(hits)
        for rec in r.json().get("records", []):
            scores[int(rec["id"])] = float(rec.get("score", 0.0))
        return _reorder(hits, scores, top_n, self.name)


def build_reranker(settings: Settings, name: str | None = None) -> Reranker:
    choice = name or settings.reranker
    if choice == "none":
        return NoopReranker()
    if choice == "local":
        return LocalCrossEncoderReranker(settings.reranker_model)
    if choice == "cohere":
        return CohereReranker(settings.cohere_api_key)
    if choice == "vertex":
        return VertexRankingReranker(settings.google_cloud_project)
    raise ValueError(f"Unknown reranker: {choice}")
