"""Contextual chunk headers ("contextual retrieval", Anthropic 2024).

A paragraph such as "2. The risk management system shall comprise the following steps..."
does not say which regulation, article or obligation it belongs to. We prepend a short header
*for indexing only* (BM25 and embeddings see it; the user and the generator see the raw text):

* ``TemplateHeaderGenerator`` - deterministic, free: regulation > chapter > section > article
  number + title > paragraph. Works offline and captures most of the gain on legal text, whose
  hierarchy is explicit.
* ``LLMHeaderGenerator`` - the original recipe: an LLM reads the whole article and writes 1-2
  sentences situating the chunk. Results are cached on disk by content hash; any failure falls
  back to the template header.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from eu_ai_act_rag.llm.base import LLMProvider
from eu_ai_act_rag.schemas import Chunk, Lang, LegalUnit

log = logging.getLogger(__name__)

_REG = {
    ("ai_act", "en"): "EU AI Act (Regulation (EU) 2024/1689)",
    ("ai_act", "fr"): "Règlement européen sur l'IA (règlement (UE) 2024/1689)",
    ("gdpr", "en"): "GDPR (Regulation (EU) 2016/679)",
    ("gdpr", "fr"): "RGPD (règlement (UE) 2016/679)",
}
_KIND = {
    ("article", "en"): "Article",
    ("article", "fr"): "Article",
    ("recital", "en"): "Recital",
    ("recital", "fr"): "Considérant",
    ("annex", "en"): "Annex",
    ("annex", "fr"): "Annexe",
}
_PARA = {"en": "paragraph", "fr": "paragraphe"}


_ACRONYMS = re.compile(r"\b(ai|ia|ue|eu|gpai)\b")


def _cap(s: str) -> str:
    """'CHAPTER III - HIGH-RISK AI SYSTEMS' -> 'Chapter III - High-risk AI systems'."""
    if not s.isupper():
        return s
    head, sep, rest = s.partition(" - ")
    words = head.split()
    head = " ".join([words[0].capitalize(), *words[1:]]) if words else head
    rest = _ACRONYMS.sub(lambda m: m.group(0).upper(), rest.capitalize())
    return f"{head}{sep}{rest}"


def template_header(unit: LegalUnit, chunk: Chunk) -> str:
    parts = [_REG.get((unit.regulation, unit.lang), unit.regulation)]
    if unit.chapter:
        parts.append(_cap(unit.chapter))
    if unit.section:
        parts.append(_cap(unit.section))
    head = f"{_KIND[(unit.kind, unit.lang)]} {unit.number}"
    if unit.title:
        head += f": {unit.title}"
    parts.append(head)
    if chunk.paragraph:
        parts.append(f"{_PARA[unit.lang]} {chunk.paragraph}")
    return " > ".join(parts)


class TemplateHeaderGenerator:
    name = "template"

    def headers(self, unit: LegalUnit, chunks: list[Chunk]) -> list[str]:
        return [template_header(unit, c) for c in chunks]


_LLM_PROMPT = """<document>
{document}
</document>
Here is a chunk from the document above:
<chunk>
{chunk}
</chunk>
Write one or two short sentences, in {language}, that situate this chunk within the document \
(which regulation and article, who the obligation applies to, what topic) to improve search \
retrieval of the chunk. Answer only with the sentences."""


class LLMHeaderGenerator:
    name = "llm"

    def __init__(self, llm: LLMProvider, cache_dir: Path, max_doc_chars: int = 12000) -> None:
        self.llm = llm
        self.max_doc_chars = max_doc_chars
        self.cache_path = cache_dir / f"headers_{llm.name.replace(':', '_')}.json"
        self.cache: dict[str, str] = (
            json.loads(self.cache_path.read_text("utf-8")) if self.cache_path.exists() else {}
        )

    def _key(self, unit: LegalUnit, chunk: Chunk) -> str:
        return hashlib.sha1(f"{unit.unit_id}|{unit.lang}|{chunk.text}".encode()).hexdigest()

    def headers(self, unit: LegalUnit, chunks: list[Chunk]) -> list[str]:
        out: list[str] = []
        dirty = False
        for c in chunks:
            key = self._key(unit, c)
            if key not in self.cache:
                base = template_header(unit, c)
                try:
                    gen = self.llm.complete(
                        "You write retrieval context for legal text chunks.",
                        _LLM_PROMPT.format(
                            document=f"{base}\n{unit.text}"[: self.max_doc_chars],
                            chunk=c.text,
                            language="French" if unit.lang == "fr" else "English",
                        ),
                        max_tokens=120,
                    ).strip()
                    self.cache[key] = f"{base}. {gen}" if gen else base
                except Exception as exc:
                    log.warning("LLM header failed for %s: %s", c.chunk_id, exc)
                    self.cache[key] = base
                dirty = True
            out.append(self.cache[key])
        if dirty:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), "utf-8")
        return out


def apply_headers(
    units: list[LegalUnit],
    chunks: list[Chunk],
    generator: TemplateHeaderGenerator | LLMHeaderGenerator,
) -> list[Chunk]:
    """Return copies of ``chunks`` with ``header`` filled. Chunks whose unit is unknown (or
    that straddle units, like fixed-size windows) get the header of their majority unit."""
    by_key = {(u.unit_id, u.lang): u for u in units}
    groups: dict[tuple[str, Lang], list[Chunk]] = {}
    for c in chunks:
        groups.setdefault((c.unit_id, c.lang), []).append(c)
    out: dict[str, Chunk] = {}
    for key, group in groups.items():
        unit = by_key.get(key)
        if unit is None:
            out.update({c.chunk_id: c for c in group})
            continue
        for c, h in zip(group, generator.headers(unit, group), strict=True):
            out[c.chunk_id] = c.model_copy(update={"header": h})
    return [out[c.chunk_id] for c in chunks]
