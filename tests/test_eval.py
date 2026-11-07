from __future__ import annotations

import re
from pathlib import Path

import pytest

from eu_ai_act_rag.config import PROJECT_ROOT
from eu_ai_act_rag.embeddings.providers import HashingEmbedder
from eu_ai_act_rag.evaluation.ablation import (
    OFFLINE_STEPS,
    answerable,
    load_golden,
    run_ablation,
    write_report,
)
from eu_ai_act_rag.evaluation.judge import LexicalJudge, agreement, cohen_kappa, spearman
from eu_ai_act_rag.evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from eu_ai_act_rag.schemas import LegalUnit

GOLDEN = PROJECT_ROOT / "evals" / "golden" / "golden_v1.jsonl"
UNIT_ID = re.compile(r"^(ai_act|gdpr):(art_\d+|rct_\d+|anx_[IVXL]+)$")


def test_golden_set_is_well_formed() -> None:
    items = load_golden(GOLDEN)
    assert 60 <= len(items) <= 100
    assert len({i.id for i in items}) == len(items)
    assert {i.lang for i in items} == {"en", "fr"}
    assert sum(i.lang == "fr" for i in items) >= 0.3 * len(items)
    for i in items:
        assert all(UNIT_ID.match(u) for u in i.gold), i.id
        assert i.reference and i.question
    assert any(not i.gold for i in items), "needs unanswerable items to test abstention"


def test_metrics() -> None:
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, ["b", "z"], 2) == 0.5
    assert mrr_at_k(ranked, ["c"]) == pytest.approx(1 / 3)
    assert ndcg_at_k(ranked, ["a"]) == 1.0
    assert 0 < ndcg_at_k(ranked, ["d"]) < 1


def test_offline_ablation_runs(units: list[LegalUnit], tmp_path: Path) -> None:
    golden = load_golden(GOLDEN)
    assert len(answerable(golden, units)) >= 40
    results = run_ablation(
        OFFLINE_STEPS,
        units,
        golden,
        embedder_factory=HashingEmbedder,
        reranker_factory=None,
        log=lambda _: None,
    )
    assert all(not r.skipped for r in results)
    for r in results:
        assert all(0.0 <= v <= 1.0 for v in r.metrics.values())
    write_report(results, tmp_path / "r.md", title="t", setup={}, chart_name="x.png")
    assert "| Step |" in (tmp_path / "r.md").read_text(encoding="utf-8")


def test_judge_calibration_stats() -> None:
    assert cohen_kappa([1, 0, 1, 1], [1, 0, 1, 1]) == 1.0
    assert spearman([0.1, 0.5, 0.9], [1, 2, 3]) == pytest.approx(1.0)
    stats = agreement([0.9, 0.2, 0.7, 0.4], [1.0, 0.0, 0.0, 0.0])
    assert stats["raw_agreement"] == 0.75


def test_lexical_judge() -> None:
    j = LexicalJudge().judge(
        "q",
        "The controller notifies within 72 hours [1].",
        ["The controller shall notify within 72 hours."],
        "Within 72 hours.",
    )
    assert j.faithfulness == 1.0 and j.correctness > 0
