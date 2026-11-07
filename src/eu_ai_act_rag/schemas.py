"""Core data model shared by ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Lang = Literal["en", "fr"]
UnitKind = Literal["article", "recital", "annex"]


class Paragraph(BaseModel):
    """A numbered paragraph of an article (or one block of a recital/annex).

    ``lead`` is the introductory sentence ("The following AI practices shall be prohibited:")
    and ``points`` are the enumerated items under it; ``text`` is derived from both so the
    stored corpus carries no duplicated text.
    """

    number: str | None = None
    lead: str = ""
    points: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join([self.lead, *self.points]) if self.lead else "\n".join(self.points)


class LegalUnit(BaseModel):
    """One article, recital or annex of a regulation, in one language."""

    unit_id: str  # language-independent, e.g. "ai_act:art_5", "gdpr:art_17", "ai_act:anx_III"
    regulation: str  # "ai_act" | "gdpr"
    kind: UnitKind
    number: str  # "5", "III", "42"
    title: str = ""
    lang: Lang
    chapter: str = ""  # "Chapter II - Prohibited AI practices"
    section: str = ""
    paragraphs: list[Paragraph]
    source_url: str = ""

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)

    @property
    def label(self) -> str:
        return unit_label(self.unit_id, self.lang)


class Chunk(BaseModel):
    chunk_id: str
    unit_id: str
    regulation: str
    lang: Lang
    text: str  # raw text shown to users and passed to the LLM
    header: str = ""  # contextual header prepended for indexing only
    paragraph: str | None = None
    title: str = ""
    strategy: str = ""

    @property
    def index_text(self) -> str:
        return f"{self.header}\n{self.text}" if self.header else self.text

    @property
    def citation_label(self) -> str:
        base = unit_label(self.unit_id, self.lang)
        return f"{base}({self.paragraph})" if self.paragraph else base


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    rank: int = 0
    source: str = ""  # which retriever produced the score


class Citation(BaseModel):
    marker: int  # the [n] marker used in the answer
    chunk_id: str
    unit_id: str
    label: str  # human readable, e.g. "AI Act, Art. 5(1)"
    quote: str = ""


class Answer(BaseModel):
    """Structured answer returned by the API. Citations are mandatory unless abstaining."""

    question: str
    language: Lang
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: str | None = None
    provider: str = ""
    retrieval: list[SearchHit] = Field(default_factory=list)


_REG_NAMES = {
    ("ai_act", "en"): "AI Act",
    ("ai_act", "fr"): "Règlement IA",
    ("gdpr", "en"): "GDPR",
    ("gdpr", "fr"): "RGPD",
}
_KIND_NAMES = {
    ("art", "en"): "Art.",
    ("art", "fr"): "art.",
    ("rct", "en"): "Recital",
    ("rct", "fr"): "considérant",
    ("anx", "en"): "Annex",
    ("anx", "fr"): "annexe",
}


def unit_label(unit_id: str, lang: Lang = "en") -> str:
    """'ai_act:art_5' -> 'AI Act, Art. 5' / 'Règlement IA, art. 5'."""
    reg, _, local = unit_id.partition(":")
    kind, _, num = local.partition("_")
    reg_name = _REG_NAMES.get((reg, lang), reg)
    kind_name = _KIND_NAMES.get((kind, lang), kind)
    return f"{reg_name}, {kind_name} {num}"
