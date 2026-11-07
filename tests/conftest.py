from __future__ import annotations

from pathlib import Path

import pytest

from eu_ai_act_rag.config import PROJECT_ROOT, Settings
from eu_ai_act_rag.ingest.store import read_units
from eu_ai_act_rag.pipeline import RAGService, build_service
from eu_ai_act_rag.schemas import LegalUnit

FIXTURE = PROJECT_ROOT / "data" / "fixtures" / "corpus_fixture.jsonl"


@pytest.fixture(scope="session")
def units() -> list[LegalUnit]:
    return read_units(FIXTURE)


@pytest.fixture(scope="session")
def offline_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    cache: Path = tmp_path_factory.mktemp("cache")
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        corpus_path=FIXTURE,
        cache_dir=cache,
        llm_provider="mock",
        embedding_provider="hashing",
        retriever="hybrid",
        reranker="none",
        vector_store="memory",
        contextual_headers="template",
    )


@pytest.fixture(scope="session")
def service(offline_settings: Settings) -> RAGService:
    return build_service(offline_settings)
