"""FastAPI application: /health, /search, /ask, /ask/stream (SSE), /units/{unit_id}."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from eu_ai_act_rag import __version__
from eu_ai_act_rag.config import Settings, get_settings
from eu_ai_act_rag.pipeline import RAGService, build_service
from eu_ai_act_rag.schemas import Answer, Lang, LegalUnit, SearchHit

log = logging.getLogger("eu_ai_act_rag.api")

DISCLAIMER = "Informational tool, not legal advice. Always check the official text on EUR-Lex."


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    k: int = Field(default=6, ge=1, le=50)
    lang: Lang | None = None


class SearchResponse(BaseModel):
    language: Lang
    hits: list[SearchHit]
    latency_ms: dict[str, float]


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    lang: Lang | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    units: int
    chunks: int
    retriever: str
    reranker: str
    llm: str
    disclaimer: str = DISCLAIMER


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service or build_service(settings)
        yield

    app = FastAPI(
        title="EU AI Act & GDPR RAG", version=__version__, lifespan=lifespan, description=DISCLAIMER
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def svc(request: Request) -> RAGService:
        s: RAGService = request.app.state.service
        return s

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        s = svc(request)
        return HealthResponse(
            status="ok",
            version=__version__,
            units=len(s.units),
            chunks=len(s.chunks),
            retriever=s.pipeline.retriever.name,
            reranker=s.pipeline.reranker.name,
            llm=s.llm.name,
        )

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest, request: Request) -> SearchResponse:
        s = svc(request)
        lang, hits = s.search(req.query, req.k, req.lang)
        return SearchResponse(language=lang, hits=hits, latency_ms=s.pipeline.last_latency_ms)

    @app.post("/ask", response_model=Answer)
    def ask(req: AskRequest, request: Request) -> Answer:
        return svc(request).ask(req.question, req.lang)

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest, request: Request) -> StreamingResponse:
        s = svc(request)

        def events() -> Iterator[str]:
            t0 = time.perf_counter()
            try:
                hits, stream = s.ask_stream(req.question, req.lang)
                yield _sse("retrieval", [h.model_dump() for h in hits])
                for ev in stream:
                    if ev.kind == "token":
                        yield _sse("token", {"text": ev.token})
                    elif ev.answer is not None:
                        payload = ev.answer.model_dump(exclude={"retrieval"})
                        payload["latency_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
                        yield _sse("final", payload)
            except Exception as exc:  # surfaced to the client instead of a broken stream
                log.exception("stream failed")
                yield _sse("error", {"detail": type(exc).__name__})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/units/{unit_id}", response_model=LegalUnit)
    def get_unit(unit_id: str, request: Request, lang: Lang = "en") -> LegalUnit:
        unit = svc(request).get_unit(unit_id, lang)
        if unit is None:
            raise HTTPException(status_code=404, detail="unit not found in loaded corpus")
        return unit

    return app


def app_factory() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return create_app()
