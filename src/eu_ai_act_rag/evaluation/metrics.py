"""Retrieval metrics computed at the *legal unit* level (article / annex / recital).

Chunks are mapped to their unit and de-duplicated in rank order, so a strategy that returns
five paragraphs of Article 5 is not rewarded five times, and fixed-size and structure-aware
chunking are compared on the same footing (what a lawyer cares about: the right article).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from eu_ai_act_rag.schemas import SearchHit


def unit_ranking(hits: Sequence[SearchHit]) -> list[str]:
    seen: list[str] = []
    for h in hits:
        if h.chunk.unit_id not in seen:
            seen.append(h.chunk.unit_id)
    return seen


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & set(gold)) / len(set(gold))


def hit_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & set(gold) else 0.0


def mrr_at_k(ranked: Sequence[str], gold: Sequence[str], k: int = 10) -> float:
    for i, u in enumerate(ranked[:k], 1):
        if u in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int = 10) -> float:
    """Binary relevance nDCG."""
    gold_set = set(gold)
    dcg = sum(1.0 / math.log2(i + 1) for i, u in enumerate(ranked[:k], 1) if u in gold_set)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))
    return s[idx]
