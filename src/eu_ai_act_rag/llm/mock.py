"""Deterministic, offline "LLM": an extractive reader over the numbered context.

It consumes exactly the same prompt as the real providers (see `generation.prompts`), picks the
sentences with the highest lexical overlap with the question and cites them with ``[n]``
markers. It never invents text, which makes it a useful lower bound and lets the whole stack
(API, streaming, citations, abstention, UI) run with zero keys. It is *not* a language model:
answers are verbatim extracts, not syntheses.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from eu_ai_act_rag.generation.prompts import ABSTAIN_SENTINEL, parse_prompt
from eu_ai_act_rag.text import detect_language, split_sentences, tokenize

_INTRO = {"en": "Based on the retrieved provisions:", "fr": "D'après les dispositions retrouvées :"}


class MockLLM:
    name = "mock"

    def __init__(self, max_sentences: int = 3, min_overlap: float = 0.34) -> None:
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        question, passages = parse_prompt(prompt)
        if not passages:
            return ABSTAIN_SENTINEL
        q_terms = set(tokenize(question))
        if not q_terms:
            return ABSTAIN_SENTINEL
        scored: list[tuple[float, int, int, str]] = []
        for order, (marker, label, text) in enumerate(passages):
            # the passage label carries the article title ("Right to erasure"), which is often
            # the only place the question's key term appears
            label_terms = set(tokenize(label))
            for sent in split_sentences(text) or [text]:
                s_terms = set(tokenize(sent)) | label_terms
                if not s_terms:
                    continue
                overlap = len(q_terms & s_terms) / len(q_terms)
                # small prior for higher-ranked passages keeps answers anchored on top hits
                scored.append((overlap - 0.01 * order, order, marker, sent))
        scored.sort(key=lambda t: (-t[0], t[1]))
        if not scored or scored[0][0] + 0.01 * scored[0][1] < self.min_overlap:
            return ABSTAIN_SENTINEL
        picked: list[tuple[int, int, str]] = []
        for score, order, marker, sent in scored:
            if score + 0.01 * order < self.min_overlap or len(picked) >= self.max_sentences:
                break
            if all(sent != p[2] for p in picked):
                picked.append((order, marker, sent))
        picked.sort()
        lang = detect_language(question)
        body = "\n".join(f"- {' '.join(s.split())} [{m}]" for _, m, s in picked)
        return f"{_INTRO[lang]}\n{body}"

    def stream(self, system: str, prompt: str, max_tokens: int = 1024) -> Iterator[str]:
        text = self.complete(system, prompt, max_tokens)
        yield from re.findall(r"\S+\s*", text)
