"""Vector stores: in-memory (numpy, exact cosine) and Postgres/pgvector (HNSW)."""

from __future__ import annotations

import json
from typing import Any, Protocol

import numpy as np

from eu_ai_act_rag.embeddings.providers import Matrix
from eu_ai_act_rag.schemas import Chunk, Lang


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: Matrix) -> None: ...

    def search(
        self, query: Matrix, k: int, lang: Lang | None = None
    ) -> list[tuple[Chunk, float]]: ...

    def __len__(self) -> int: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._m: Matrix | None = None
        self._langs = np.array([], dtype=object)

    def add(self, chunks: list[Chunk], vectors: Matrix) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        self.chunks.extend(chunks)
        self._m = vectors if self._m is None else np.vstack([self._m, vectors])
        self._langs = np.array([c.lang for c in self.chunks], dtype=object)

    def search(self, query: Matrix, k: int, lang: Lang | None = None) -> list[tuple[Chunk, float]]:
        if self._m is None or not len(self.chunks):
            return []
        scores = self._m @ query.reshape(-1)
        if lang is not None:
            scores = np.where(self._langs == lang, scores, -np.inf)
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.chunks[i], float(scores[i])) for i in idx if np.isfinite(scores[i])]

    def __len__(self) -> int:
        return len(self.chunks)


class PgVectorStore:  # pragma: no cover - exercised in docker-compose, not in unit tests
    """Chunks live in one table; ``collection`` separates indexes built with different
    chunking/embedding configs so ablation runs never collide."""

    def __init__(self, dsn: str, collection: str) -> None:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise RuntimeError("pip install 'eu-ai-act-rag[pg]'") from exc
        self.collection = collection
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._dim: int | None = None

    def _ensure_table(self, dim: int) -> None:
        if self._dim == dim:
            return
        self._conn.execute(
            f"""CREATE TABLE IF NOT EXISTS chunks (
                collection text NOT NULL,
                chunk_id text NOT NULL,
                lang text NOT NULL,
                unit_id text NOT NULL,
                payload jsonb NOT NULL,
                embedding vector({dim}) NOT NULL,
                PRIMARY KEY (collection, chunk_id))"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )
        self._dim = dim

    def add(self, chunks: list[Chunk], vectors: Matrix) -> None:
        if not chunks:
            return
        self._ensure_table(int(vectors.shape[1]))
        with self._conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (collection, chunk_id, lang, unit_id, payload, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (collection, chunk_id) DO UPDATE "
                "SET payload = EXCLUDED.payload, embedding = EXCLUDED.embedding",
                [
                    (self.collection, c.chunk_id, c.lang, c.unit_id, json.dumps(c.model_dump()), v)
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    def search(self, query: Matrix, k: int, lang: Lang | None = None) -> list[tuple[Chunk, float]]:
        q = query.reshape(-1)
        rows: list[Any] = self._conn.execute(
            "SELECT payload, 1 - (embedding <=> %s) AS score FROM chunks "
            "WHERE collection = %s AND (%s::text IS NULL OR lang = %s) "
            "ORDER BY embedding <=> %s LIMIT %s",
            (q, self.collection, lang, lang, q, k),
        ).fetchall()
        return [(Chunk.model_validate(r[0]), float(r[1])) for r in rows]

    def __len__(self) -> int:
        row = self._conn.execute(
            "SELECT count(*) FROM chunks WHERE collection = %s", (self.collection,)
        ).fetchone()
        return int(row[0]) if row else 0
