from __future__ import annotations

from typing import Protocol

from eu_ai_act_rag.schemas import Chunk, LegalUnit


class Chunker(Protocol):
    name: str

    def chunk(self, units: list[LegalUnit]) -> list[Chunk]: ...


class HeaderGenerator(Protocol):
    """Produces a short context string prepended to a chunk at indexing time."""

    name: str

    def headers(self, unit: LegalUnit, chunks: list[Chunk]) -> list[str]: ...
