"""Runtime configuration. Every value can be set through environment variables or `.env`.

Defaults are chosen so that the application runs fully offline with zero API keys:
fixture corpus, BM25 + hashing embeddings, extractive `mock` LLM, in-memory vector store.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- corpus -----------------------------------------------------------------------------
    corpus_path: Path = Field(
        default=PROJECT_ROOT / "data" / "fixtures" / "corpus_fixture.jsonl",
        description="JSONL of parsed LegalUnits. `make ingest` writes data/processed/corpus.jsonl",
    )
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"

    # --- chunking ---------------------------------------------------------------------------
    chunking: Literal["fixed", "structure"] = "structure"
    contextual_headers: Literal["none", "template", "llm"] = "template"
    fixed_chunk_words: int = 180
    fixed_overlap_words: int = 30
    structure_max_chars: int = 1800

    # --- retrieval --------------------------------------------------------------------------
    retriever: Literal["bm25", "dense", "hybrid"] = "hybrid"
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    hybrid_alpha: float = 0.5  # weight of dense scores for weighted fusion
    candidate_k: int = 40  # per-retriever candidates before fusion / reranking
    top_k: int = 6  # chunks passed to the generator
    reranker: Literal["none", "local", "cohere", "vertex"] = "none"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    language_filter: Literal["auto", "all"] = "auto"

    # --- embeddings / vector store ----------------------------------------------------------
    embedding_provider: Literal["hashing", "local", "vertex", "openai"] = "hashing"
    local_embedding_model: str = "intfloat/multilingual-e5-small"
    vertex_embedding_model: str = "text-multilingual-embedding-002"
    openai_embedding_model: str = "text-embedding-3-small"
    vector_store: Literal["memory", "pgvector"] = "memory"
    database_url: str = ""  # e.g. postgresql://rag:rag@localhost:5432/rag

    # --- generation -------------------------------------------------------------------------
    llm_provider: Literal["mock", "gemini", "anthropic", "openai"] = "mock"
    gemini_model: str = "gemini-2.5-flash"
    anthropic_model: str = "claude-opus-5-5"
    openai_model: str = "gpt-4o-mini"
    max_output_tokens: int = 1024
    abstain_min_score: float = 0.0  # extra guard for the mock provider (score of top hit)

    # --- credentials (never committed; see .env.example) ------------------------------------
    google_cloud_project: str = ""
    google_cloud_location: str = "europe-west1"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    cohere_api_key: str = ""

    # --- api --------------------------------------------------------------------------------
    cors_origins: str = "http://localhost:5173,http://localhost:8080"


@lru_cache
def get_settings() -> Settings:
    return Settings()
