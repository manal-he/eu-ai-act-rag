"""Answer-quality judges: faithfulness (is every claim supported by the cited context?) and
correctness (does the answer agree with the reference answer?).

``LLMJudge`` uses any `LLMProvider` with a fixed rubric and returns a 0-1 score per criterion.
``LexicalJudge`` is a deterministic offline proxy (token overlap) so the generation eval can
run in CI; its numbers are *not* comparable to an LLM judge and are labelled as such.

Calibration: an LLM judge is a measurement instrument and must be calibrated against human
labels before its numbers are trusted - see evals/calibration/README.md and
`agreement()` below (Cohen's kappa on binarised scores + Spearman rank correlation).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from eu_ai_act_rag.llm.base import LLMProvider
from eu_ai_act_rag.text import split_sentences, tokenize


@dataclass
class Judgement:
    faithfulness: float
    correctness: float
    rationale: str = ""


class Judge(Protocol):
    name: str

    def judge(
        self, question: str, answer: str, contexts: list[str], reference: str
    ) -> Judgement: ...


def _f1(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    if not inter:
        return 0.0
    p, r = inter / len(sa), inter / len(sb)
    return 2 * p * r / (p + r)


class LexicalJudge:
    """Faithfulness = share of answer sentences whose content words are >= `support`%
    present in the context. Correctness = token F1 against the reference answer."""

    name = "lexical-proxy"

    def __init__(self, support: float = 0.6) -> None:
        self.support = support

    def judge(self, question: str, answer: str, contexts: list[str], reference: str) -> Judgement:
        ctx = set(tokenize(" ".join(contexts)))
        clean = re.sub(r"\[\d+\]", "", answer)
        sents = [s for s in (split_sentences(clean) or [clean]) if tokenize(s)]
        supported = 0
        for s in sents:
            toks = set(tokenize(s))
            if toks and len(toks & ctx) / len(toks) >= self.support:
                supported += 1
        faith = supported / len(sents) if sents else 0.0
        return Judgement(faith, _f1(tokenize(clean), tokenize(reference)))


_RUBRIC = """You are grading an answer produced by a legal question-answering assistant about \
the EU AI Act and the GDPR.

<question>{question}</question>
<context>{context}</context>
<reference_answer>{reference}</reference_answer>
<answer>{answer}</answer>

Score two criteria from 0 to 1:
- faithfulness: fraction of the answer's claims that are directly supported by the context \
(ignore the reference). Unsupported or contradicted claims lower the score.
- correctness: agreement of the answer with the reference answer on the legal substance \
(who is obliged, what, conditions, numbers, dates). Missing key elements lowers the score; \
extra correct detail does not.
Return only JSON: {{"faithfulness": <float>, "correctness": <float>, "rationale": "<short>"}}"""


def _clip(v: object) -> float:
    return max(0.0, min(1.0, float(str(v))))


class LLMJudge:
    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm
        self.name = f"llm-judge:{llm.name}"

    def judge(self, question: str, answer: str, contexts: list[str], reference: str) -> Judgement:
        raw = self.llm.complete(
            "You are a strict, consistent grader. Output JSON only.",
            _RUBRIC.format(
                question=question,
                context="\n---\n".join(contexts),
                reference=reference,
                answer=answer,
            ),
            max_tokens=300,
        )
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return Judgement(0.0, 0.0, f"unparseable judge output: {raw[:80]}")
        data = json.loads(m.group(0))
        return Judgement(
            _clip(data.get("faithfulness", 0)),
            _clip(data.get("correctness", 0)),
            str(data.get("rationale", "")),
        )


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((list(a).count(lbl) / n) * (list(b).count(lbl) / n) for lbl in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def agreement(
    judge_scores: Sequence[float], human_scores: Sequence[float], threshold: float = 0.5
) -> dict[str, float]:
    """Judge-vs-human agreement on one criterion."""
    jb = [int(s >= threshold) for s in judge_scores]
    hb = [int(s >= threshold) for s in human_scores]
    return {
        "n": float(len(jb)),
        "cohen_kappa": cohen_kappa(jb, hb),
        "spearman": spearman(judge_scores, human_scores),
        "raw_agreement": sum(x == y for x, y in zip(jb, hb, strict=True)) / len(jb),
    }
