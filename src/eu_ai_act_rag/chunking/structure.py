"""Structure-aware chunking: one chunk per article paragraph / recital / annex block.

Legal text is cited at paragraph level ("Article 5(1)(h)"), so the paragraph is the natural
retrieval unit: it keeps citation metadata exact and never mixes two articles. Paragraphs longer
than ``max_chars`` (e.g. Article 5(1) with its eight prohibited practices, or the 68 definitions
of Article 3) are split on enumerated points, and the paragraph's lead-in sentence is carried
into every piece so that "(h) the use of 'real-time' remote biometric identification..." still
says *what* is being enumerated ("The following AI practices shall be prohibited:").
"""

from __future__ import annotations

from eu_ai_act_rag.schemas import Chunk, LegalUnit, Paragraph
from eu_ai_act_rag.text import split_sentences


def _group(items: list[str], max_chars: int, prefix: str = "") -> list[str]:
    groups: list[str] = []
    cur: list[str] = []
    size = len(prefix)
    for it in items:
        if cur and size + len(it) > max_chars:
            groups.append("\n".join([prefix, *cur]).strip())
            cur, size = [], len(prefix)
        cur.append(it)
        size += len(it) + 1
    if cur:
        groups.append("\n".join([prefix, *cur]).strip())
    return groups


def split_paragraph(p: Paragraph, max_chars: int) -> list[str]:
    if len(p.text) <= max_chars:
        return [p.text]
    if p.points:
        budget = max(400, max_chars - len(p.lead))
        items: list[str] = []
        for pt in p.points:  # a single oversized point is split on sentence boundaries
            items.extend(_group(split_sentences(pt) or [pt], budget) if len(pt) > budget else [pt])
        return _group(items, max_chars, prefix=p.lead)
    return _group(split_sentences(p.text) or [p.text], max_chars)


class StructureAwareChunker:
    def __init__(self, max_chars: int = 1800) -> None:
        self.max_chars = max_chars
        self.name = "structure"

    def chunk_unit(self, u: LegalUnit) -> list[Chunk]:
        out: list[Chunk] = []
        for p in u.paragraphs:
            for piece in split_paragraph(p, self.max_chars):
                out.append(
                    Chunk(
                        chunk_id=f"{u.unit_id}:{u.lang}:{self.name}:{len(out)}",
                        unit_id=u.unit_id,
                        regulation=u.regulation,
                        lang=u.lang,
                        text=piece,
                        paragraph=p.number if u.kind == "article" else None,
                        title=u.title,
                        strategy=self.name,
                    )
                )
        return out

    def chunk(self, units: list[LegalUnit]) -> list[Chunk]:
        return [c for u in units for c in self.chunk_unit(u)]
