"""Naive baseline: sliding word windows over the whole document, ignoring legal structure.

Each regulation/language is flattened in document order (article headings included, as they
appear in the text) and cut every ``size`` words with ``overlap``. A chunk is attributed to the
unit that contributes the majority of its words - a chunk straddling Articles 5 and 6 counts as
one of them only, which is exactly the failure mode structure-aware chunking removes.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from eu_ai_act_rag.schemas import Chunk, LegalUnit


def _heading(u: LegalUnit) -> str:
    kind = {"article": "Article", "recital": "Recital", "annex": "Annex"}[u.kind]
    return f"{kind} {u.number} {u.title}".strip()


class FixedSizeChunker:
    def __init__(self, size: int = 180, overlap: int = 30) -> None:
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size, self.overlap = size, overlap
        self.name = f"fixed{size}"

    def chunk(self, units: list[LegalUnit]) -> list[Chunk]:
        streams: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        for u in units:
            words = f"{_heading(u)}\n{u.text}".split()
            streams[(u.regulation, u.lang)].extend((w, u.unit_id) for w in words)

        chunks: list[Chunk] = []
        step = self.size - self.overlap
        for (reg, lang), stream in streams.items():
            for i, start in enumerate(range(0, max(len(stream) - self.overlap, 1), step)):
                window = stream[start : start + self.size]
                if not window:
                    break
                unit_id = Counter(uid for _, uid in window).most_common(1)[0][0]
                chunks.append(
                    Chunk(
                        chunk_id=f"{reg}:{lang}:{self.name}:{i}",
                        unit_id=unit_id,
                        regulation=reg,
                        lang=lang,  # type: ignore[arg-type]
                        text=" ".join(w for w, _ in window),
                        strategy=self.name,
                    )
                )
        return chunks
