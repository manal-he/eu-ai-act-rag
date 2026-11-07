"""Text normalisation, tokenisation and a tiny FR/EN language detector.

Kept dependency-free on purpose: BM25 and the offline pipeline must run anywhere.
"""

from __future__ import annotations

import re
import unicodedata

from eu_ai_act_rag.schemas import Lang

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")

EN_STOP = frozenset(
    """a an and are as at be been by for from has have in into is it its of on or such that the
    their them these this those to was were which with shall may any other where when than
    not no all under referred point points paragraph paragraphs article articles regulation
    what who how does do can should must i my our we you your if""".split()
)
FR_STOP = frozenset(
    """a au aux avec ce ces cet cette dans de des du elle en est et il ils la le les leur leurs
    lorsque ou par pas pour qu que qui sa se ses son sont sur un une y l d s n c j m t
    point points paragraphe article articles reglement vise visee visees vises present
    quel quelle quels quelles comment peut doit doivent je mon ma mes nous vous votre si""".split()
)
STOPWORDS = EN_STOP | FR_STOP

_FR_MARKERS = frozenset(
    """le la les des du une est et que qui pour dans sur au aux quelles quels quelle quel
    comment peut doit sont ce cette ou lorsque""".split()
)
_EN_MARKERS = frozenset(
    """the of and is are what which who how does do can for with under to an a must should
    when""".split()
)


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text.replace("\xa0", " ").replace(" ", " ")).strip()


def fold_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _stem(tok: str) -> str:
    # Deliberately minimal plural folding (EN/FR) - a real stemmer is a roadmap item.
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok[-1] in "sx" and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    toks = _TOKEN.findall(fold_accents(text.lower()))
    if not keep_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [_stem(t) for t in toks]


def detect_language(text: str, default: Lang = "en") -> Lang:
    """Stop-word vote between French and English. Good enough for routing a question."""
    words = re.findall(r"[a-z']+", fold_accents(text.lower()))
    fr = sum(1 for w in words if w in _FR_MARKERS or w.startswith(("l'", "d'", "qu'")))
    en = sum(1 for w in words if w in _EN_MARKERS)
    if any(ch in text for ch in "éèêàçùôî"):
        fr += 2
    if fr == en:
        return default
    return "fr" if fr > en else "en"


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;:])\s+(?=[A-Z(\d«À-Ý])", normalize_ws(text))
    return [p for p in parts if len(p) > 20]
