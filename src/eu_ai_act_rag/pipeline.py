"""Wiring: configuration -> chunks -> index -> retrieval pipeline -> answerer -> service."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from eu_ai_act_rag.chunking.contextual import (
    LLMHeaderGenerator,
    TemplateHeaderGenerator,
    apply_headers,
)
from eu_ai_act_rag.chunking.fixed import FixedSizeChunker
from eu_ai_act_rag.chunking.structure import StructureAwareChunker
from eu_ai_act_rag.config import Settings
from eu_ai_act_rag.embeddings.providers import Embedder, build_embedder
from eu_ai_act_rag.generation.answerer import Answerer, StreamEvent
from eu_ai_act_rag.ingest.store import read_units
from eu_ai_act_rag.llm.base import LLMProvider
from eu_ai_act_rag.llm.providers import build_llm
from eu_ai_act_rag.retrieval.rerank import NoopReranker, Reranker, build_reranker
from eu_ai_act_rag.retrieval.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    Retriever,
)
from eu_ai_act_rag.retrieval.vectorstore import InMemoryVectorStore, PgVectorStore, VectorStore
from eu_ai_act_rag.schemas import Answer, Chunk, Lang, LegalUnit, SearchHit
from eu_ai_act_rag.text import detect_language

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalConfig:
    chunking: str = "structure"  # fixed | structure
    headers: str = "template"  # none | template | llm
    retriever: str = "hybrid"  # bm25 | dense | hybrid
    fusion: str = "rrf"  # rrf | weighted
    reranker: str = "none"
    candidate_k: int = 40
    rrf_k: int = 60
    alpha: float = 0.5
    fixed_words: int = 180
    fixed_overlap: int = 30
    max_chars: int = 1800

    @classmethod
    def from_settings(cls, s: Settings) -> RetrievalConfig:
        return cls(
            chunking=s.chunking,
            headers=s.contextual_headers,
            retriever=s.retriever,
            fusion=s.fusion,
            reranker=s.reranker,
            candidate_k=s.candidate_k,
            rrf_k=s.rrf_k,
            alpha=s.hybrid_alpha,
            fixed_words=s.fixed_chunk_words,
            fixed_overlap=s.fixed_overlap_words,
            max_chars=s.structure_max_chars,
        )

    def slug(self) -> str:
        return hashlib.sha1(repr(self).encode()).hexdigest()[:10]


def make_chunks(
    units: list[LegalUnit],
    cfg: RetrievalConfig,
    *,
    header_llm: LLMProvider | None = None,
    settings: Settings | None = None,
) -> list[Chunk]:
    chunker = (
        FixedSizeChunker(cfg.fixed_words, cfg.fixed_overlap)
        if cfg.chunking == "fixed"
        else StructureAwareChunker(cfg.max_chars)
    )
    chunks = chunker.chunk(units)
    if cfg.headers == "template":
        chunks = apply_headers(units, chunks, TemplateHeaderGenerator())
    elif cfg.headers == "llm":
        if header_llm is None or header_llm.name == "mock" or settings is None:
            log.warning("LLM headers need a real LLM provider; using template headers")
            chunks = apply_headers(units, chunks, TemplateHeaderGenerator())
        else:
            chunks = apply_headers(
                units, chunks, LLMHeaderGenerator(header_llm, settings.cache_dir)
            )
    return chunks


@dataclass
class RetrievalPipeline:
    retriever: Retriever
    reranker: Reranker = field(default_factory=NoopReranker)
    candidate_k: int = 40
    last_latency_ms: dict[str, float] = field(default_factory=dict)

    def search(self, query: str, k: int, lang: Lang | None) -> list[SearchHit]:
        t0 = time.perf_counter()
        n = max(k, self.candidate_k) if not isinstance(self.reranker, NoopReranker) else k
        hits = self.retriever.search(query, n, lang)
        t1 = time.perf_counter()
        hits = self.reranker.rerank(query, hits, k)
        t2 = time.perf_counter()
        self.last_latency_ms = {"retrieve": (t1 - t0) * 1e3, "rerank": (t2 - t1) * 1e3}
        return hits


def build_retriever(
    cfg: RetrievalConfig, embedder: Embedder | None, store: VectorStore | None = None
) -> Retriever:
    if cfg.retriever == "bm25":
        return BM25Retriever()
    if embedder is None:
        raise ValueError(f"retriever={cfg.retriever} needs an embedder")
    dense = DenseRetriever(embedder, store)
    if cfg.retriever == "dense":
        return dense
    return HybridRetriever(
        BM25Retriever(),
        dense,
        fusion=cfg.fusion,  # type: ignore[arg-type]
        rrf_k=cfg.rrf_k,
        alpha=cfg.alpha,
        candidate_k=cfg.candidate_k,
    )


def build_pipeline(
    units: list[LegalUnit],
    cfg: RetrievalConfig,
    *,
    embedder: Embedder | None,
    reranker: Reranker | None = None,
    store: VectorStore | None = None,
    header_llm: LLMProvider | None = None,
    settings: Settings | None = None,
) -> tuple[RetrievalPipeline, list[Chunk]]:
    chunks = make_chunks(units, cfg, header_llm=header_llm, settings=settings)
    retriever = build_retriever(cfg, embedder, store)
    retriever.index(chunks)
    return RetrievalPipeline(retriever, reranker or NoopReranker(), cfg.candidate_k), chunks


class RAGService:
    def __init__(
        self,
        settings: Settings,
        units: list[LegalUnit],
        pipeline: RetrievalPipeline,
        chunks: list[Chunk],
        llm: LLMProvider,
    ) -> None:
        self.settings = settings
        self.units = {(u.unit_id, u.lang): u for u in units}
        self.pipeline = pipeline
        self.chunks = chunks
        self.llm = llm
        self.answerer = Answerer(llm, settings.max_output_tokens)

    def resolve_lang(self, text: str, lang: Lang | None) -> Lang:
        return lang or detect_language(text)

    def search(
        self, query: str, k: int | None = None, lang: Lang | None = None
    ) -> tuple[Lang, list[SearchHit]]:
        q_lang = self.resolve_lang(query, lang)
        flt = q_lang if self.settings.language_filter == "auto" else None
        return q_lang, self.pipeline.search(query, k or self.settings.top_k, flt)

    def ask(self, question: str, lang: Lang | None = None) -> Answer:
        q_lang, hits = self.search(question, lang=lang)
        return self.answerer.answer(question, q_lang, hits)

    def ask_stream(
        self, question: str, lang: Lang | None = None
    ) -> tuple[list[SearchHit], Iterator[StreamEvent]]:
        q_lang, hits = self.search(question, lang=lang)
        return hits, self.answerer.stream(question, q_lang, hits)

    def get_unit(self, unit_id: str, lang: Lang) -> LegalUnit | None:
        return self.units.get((unit_id, lang)) or self.units.get(
            (unit_id, "en" if lang == "fr" else "fr")
        )


def build_service(settings: Settings) -> RAGService:
    units = read_units(settings.corpus_path)
    cfg = RetrievalConfig.from_settings(settings)
    embedder = build_embedder(settings) if cfg.retriever != "bm25" else None
    store: VectorStore | None = None
    if settings.vector_store == "pgvector" and embedder is not None:
        store = PgVectorStore(settings.database_url, f"{embedder.name}:{cfg.slug()}")
        if len(store):
            log.info("pgvector collection already populated; re-upserting (idempotent)")
    elif embedder is not None:
        store = InMemoryVectorStore()
    llm = build_llm(settings)
    pipeline, chunks = build_pipeline(
        units,
        cfg,
        embedder=embedder,
        reranker=build_reranker(settings),
        store=store,
        header_llm=llm,
        settings=settings,
    )
    log.info(
        "Indexed %d units -> %d chunks with %s", len(units), len(chunks), pipeline.retriever.name
    )
    return RAGService(settings, units, pipeline, chunks, llm)
