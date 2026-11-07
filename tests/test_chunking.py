from __future__ import annotations

from eu_ai_act_rag.chunking.contextual import TemplateHeaderGenerator, apply_headers
from eu_ai_act_rag.chunking.fixed import FixedSizeChunker
from eu_ai_act_rag.chunking.structure import StructureAwareChunker, split_paragraph
from eu_ai_act_rag.schemas import LegalUnit, Paragraph


def _unit(uid: str, paras: list[Paragraph], lang: str = "en") -> LegalUnit:
    return LegalUnit(
        unit_id=uid,
        regulation="ai_act",
        kind="article",
        number=uid.split("_")[-1],
        title="Title",
        lang=lang,
        chapter="CHAPTER III - HIGH-RISK AI SYSTEMS",
        paragraphs=paras,
    )  # type: ignore[arg-type]


def test_split_carries_lead_into_every_piece() -> None:
    p = Paragraph(
        number="1",
        lead="1. The following shall be prohibited:",
        points=[f"({c}) " + "word " * 60 for c in "abcdef"],
    )
    pieces = split_paragraph(p, max_chars=700)
    assert len(pieces) > 1
    assert all(piece.startswith("1. The following shall be prohibited:") for piece in pieces)
    assert all(len(piece) <= 700 for piece in pieces)


def test_structure_chunks_keep_paragraph_metadata(units: list[LegalUnit]) -> None:
    chunks = StructureAwareChunker().chunk(units)
    art5 = [c for c in chunks if c.unit_id == "ai_act:art_5" and c.lang == "en"]
    assert art5 and all(c.paragraph is not None for c in art5)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert art5[0].citation_label == "AI Act, Art. 5(1)"


def test_fixed_chunks_majority_attribution() -> None:
    u1 = _unit("ai_act:art_1", [Paragraph(lead="alpha " * 100)])
    u2 = _unit("ai_act:art_2", [Paragraph(lead="beta " * 30)])
    chunks = FixedSizeChunker(size=50, overlap=10).chunk([u1, u2])
    assert chunks[0].unit_id == "ai_act:art_1"
    assert chunks[-1].unit_id == "ai_act:art_2"
    # a window straddling the boundary is attributed to exactly one unit
    assert all(c.unit_id in {"ai_act:art_1", "ai_act:art_2"} for c in chunks)


def test_template_headers_only_affect_index_text() -> None:
    u = _unit("ai_act:art_9", [Paragraph(number="2", lead="2. The risk management system...")])
    chunks = apply_headers([u], StructureAwareChunker().chunk([u]), TemplateHeaderGenerator())
    c = chunks[0]
    assert c.header.startswith("EU AI Act (Regulation (EU) 2024/1689) > Chapter III - High-risk AI")
    assert "Article 9: Title > paragraph 2" in c.header
    assert c.text.startswith("2. The risk management")
    assert c.index_text.startswith(c.header)
