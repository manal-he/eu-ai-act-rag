"""Embedding providers behind one interface, plus a persistent on-disk cache.

* ``hashing``  - offline, deterministic feature hashing of words + character trigrams. It is a
  *lexical* stand-in so the dense code path (vector store, fusion) is exercised without models
  or keys. It is not a semantic embedding and is labelled as such in every report.
* ``local``    - sentence-transformers, default ``intfloat/multilingual-e5-small`` (FR+EN in one
  space, so a French question can match English text if the language filter is disabled).
* ``vertex``   - Vertex AI ``text-multilingual-embedding-002`` via google-genai.
* ``openai``   - ``text-embedding-3-small``.
"""

from __future__ import annotations

import hashlib
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from eu_ai_act_rag.config import Settings
from eu_ai_act_rag.llm.base import ProviderNotConfigured
from eu_ai_act_rag.text import tokenize

Matrix = npt.NDArray[np.float32]


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> Matrix: ...

    def embed_query(self, text: str) -> Matrix: ...


def _l2(m: Matrix) -> Matrix:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: Matrix = (m / norms).astype(np.float32)
    return out


class HashingEmbedder:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing{dim}"

    def _vec(self, text: str) -> npt.NDArray[np.float32]:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        feats = list(toks)
        for t in toks:
            padded = f"#{t}#"
            feats.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        for f in feats:
            h = zlib.crc32(f.encode())
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> Matrix:
        return _l2(
            np.stack([self._vec(t) for t in texts])
            if texts
            else np.zeros((0, self.dim), dtype=np.float32)
        )

    def embed_query(self, text: str) -> Matrix:
        return _l2(self._vec(text)[None, :])


class LocalEmbedder:
    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ProviderNotConfigured("pip install 'eu-ai-act-rag[local]'") from exc
        self._model = SentenceTransformer(model)
        self._e5 = "e5" in model.lower()
        self.name = f"local:{model}"

    def _encode(self, texts: list[str]) -> Matrix:
        out = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 256
        )
        return np.asarray(out, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> Matrix:
        return self._encode([f"passage: {t}" if self._e5 else t for t in texts])

    def embed_query(self, text: str) -> Matrix:
        return self._encode([f"query: {text}" if self._e5 else text])


class VertexEmbedder:  # pragma: no cover - requires GCP
    def __init__(self, model: str, project: str, location: str) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ProviderNotConfigured("pip install 'eu-ai-act-rag[vertex]'") from exc
        if not project:
            raise ProviderNotConfigured("GOOGLE_CLOUD_PROJECT is not set")
        self._types = types
        self._client = genai.Client(vertexai=True, project=project, location=location)
        self.model = model
        self.name = f"vertex:{model}"

    def _embed(self, texts: list[str], task: str) -> Matrix:
        vecs: list[list[float]] = []
        for i in range(0, len(texts), 16):
            resp = self._client.models.embed_content(
                model=self.model,
                contents=texts[i : i + 16],
                config=self._types.EmbedContentConfig(task_type=task),
            )
            vecs.extend(e.values for e in resp.embeddings)
        return _l2(np.asarray(vecs, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> Matrix:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> Matrix:
        return self._embed([text], "RETRIEVAL_QUERY")


class OpenAIEmbedder:  # pragma: no cover - requires a key
    def __init__(self, model: str, api_key: str) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ProviderNotConfigured("pip install 'eu-ai-act-rag[openai]'") from exc
        if not api_key:
            raise ProviderNotConfigured("OPENAI_API_KEY is not set")
        self._client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.name = f"openai:{model}"

    def embed_documents(self, texts: list[str]) -> Matrix:
        vecs: list[list[float]] = []
        for i in range(0, len(texts), 256):
            resp = self._client.embeddings.create(model=self.model, input=texts[i : i + 256])
            vecs.extend(d.embedding for d in resp.data)
        return _l2(np.asarray(vecs, dtype=np.float32))

    def embed_query(self, text: str) -> Matrix:
        return self.embed_documents([text])


class CachedEmbedder:
    """Persists document embeddings keyed by sha1(provider, text). Re-running the ablation or
    restarting the API does not re-embed (or re-bill) unchanged chunks."""

    def __init__(self, inner: Embedder, cache_dir: Path) -> None:
        self.inner = inner
        self.name = inner.name
        slug = hashlib.sha1(inner.name.encode()).hexdigest()[:10]
        self.path = cache_dir / f"emb_{slug}.npz"
        self._cache: dict[str, npt.NDArray[np.float32]] = {}
        if self.path.exists():
            data = np.load(self.path)
            self._cache = dict(zip(data["keys"].tolist(), data["vecs"], strict=True))

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()

    def embed_documents(self, texts: list[str]) -> Matrix:
        keys = [self._key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            vecs = self.inner.embed_documents([texts[i] for i in missing])
            for i, v in zip(missing, vecs, strict=True):
                self._cache[keys[i]] = v
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                self.path,
                keys=np.array(list(self._cache)),
                vecs=np.stack(list(self._cache.values())),
            )
        if not keys:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack([self._cache[k] for k in keys]).astype(np.float32)

    def embed_query(self, text: str) -> Matrix:
        return self.inner.embed_query(text)


def build_embedder(
    settings: Settings, provider: str | None = None, *, cache: bool = True
) -> Embedder:
    name = provider or settings.embedding_provider
    inner: Embedder
    if name == "hashing":
        return HashingEmbedder()  # cheap enough that caching is pointless
    if name == "local":
        inner = LocalEmbedder(settings.local_embedding_model)
    elif name == "vertex":
        inner = VertexEmbedder(
            settings.vertex_embedding_model,
            settings.google_cloud_project,
            settings.google_cloud_location,
        )
    elif name == "openai":
        inner = OpenAIEmbedder(settings.openai_embedding_model, settings.openai_api_key)
    else:
        raise ValueError(f"Unknown embedding provider: {name}")
    return CachedEmbedder(inner, settings.cache_dir) if cache else inner
