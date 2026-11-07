"""Ablation runner: each step changes one retrieval decision and is scored on the golden set."""

from __future__ import annotations

import json
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eu_ai_act_rag.embeddings.providers import Embedder
from eu_ai_act_rag.evaluation.judge import Judge
from eu_ai_act_rag.evaluation.metrics import (
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    unit_ranking,
)
from eu_ai_act_rag.generation.answerer import Answerer
from eu_ai_act_rag.pipeline import RetrievalConfig, RetrievalPipeline, build_pipeline
from eu_ai_act_rag.retrieval.rerank import Reranker
from eu_ai_act_rag.schemas import Lang, LegalUnit

DEPTH = 20  # chunks retrieved per query for scoring (unit ranking is derived from them)


@dataclass
class GoldenItem:
    id: str
    lang: Lang
    question: str
    gold: list[str]
    reference: str
    category: str = ""


def load_golden(path: Path) -> list[GoldenItem]:
    with path.open(encoding="utf-8") as fh:
        return [GoldenItem(**json.loads(line)) for line in fh if line.strip()]


@dataclass(frozen=True)
class Step:
    name: str
    cfg: RetrievalConfig
    needs_dense: bool = False
    needs_reranker: bool = False


def _c(**kw: Any) -> RetrievalConfig:
    return RetrievalConfig(**kw)


OFFLINE_STEPS: list[Step] = [
    Step("A. Fixed 180w + BM25", _c(chunking="fixed", headers="none", retriever="bm25")),
    Step("B. Structure-aware + BM25", _c(chunking="structure", headers="none", retriever="bm25")),
    Step(
        "C. B + contextual headers (template)",
        _c(chunking="structure", headers="template", retriever="bm25"),
    ),
    Step(
        "D. C + hybrid RRF (BM25 + hashing vectors)",
        _c(chunking="structure", headers="template", retriever="hybrid"),
        needs_dense=True,
    ),
]

FULL_STEPS: list[Step] = [
    Step(
        "1. Fixed 180w + dense",
        _c(chunking="fixed", headers="none", retriever="dense"),
        needs_dense=True,
    ),
    Step(
        "2. Structure-aware + dense",
        _c(chunking="structure", headers="none", retriever="dense"),
        needs_dense=True,
    ),
    Step(
        "3. + hybrid RRF (BM25 + dense)",
        _c(chunking="structure", headers="none", retriever="hybrid"),
        needs_dense=True,
    ),
    Step(
        "4. + cross-encoder reranker",
        _c(chunking="structure", headers="none", retriever="hybrid", reranker="local"),
        needs_dense=True,
        needs_reranker=True,
    ),
    Step(
        "5. + contextual headers (template)",
        _c(chunking="structure", headers="template", retriever="hybrid", reranker="local"),
        needs_dense=True,
        needs_reranker=True,
    ),
    # controls - isolate single factors that the cumulative chain conflates
    Step("ctrl: BM25 only (structure)", _c(chunking="structure", headers="none", retriever="bm25")),
    Step(
        "ctrl: hybrid weighted a=0.5 (structure)",
        _c(chunking="structure", headers="none", retriever="hybrid", fusion="weighted"),
        needs_dense=True,
    ),
    Step(
        "ctrl: 5 without reranker",
        _c(chunking="structure", headers="template", retriever="hybrid"),
        needs_dense=True,
    ),
]


@dataclass
class StepResult:
    name: str
    config: dict[str, Any]
    n_queries: int
    n_chunks: int
    index_seconds: float
    metrics: dict[str, float]
    by_lang: dict[str, dict[str, float]]
    latency_ms: dict[str, float]
    skipped: str = ""
    per_query: list[dict[str, Any]] = field(default_factory=list)


def answerable(golden: list[GoldenItem], units: list[LegalUnit]) -> list[GoldenItem]:
    have = {(u.unit_id, u.lang) for u in units}
    return [g for g in golden if g.gold and all((u, g.lang) in have for u in g.gold)]


def _score(items: list[dict[str, Any]]) -> dict[str, float]:
    keys = ["recall@1", "recall@5", "recall@10", "hit@5", "mrr@10", "ndcg@10"]
    return {k: round(statistics.fmean(i[k] for i in items), 4) if items else 0.0 for k in keys}


def evaluate_pipeline(
    pipeline: RetrievalPipeline, items: list[GoldenItem]
) -> tuple[list[dict[str, Any]], list[float]]:
    rows, lat = [], []
    for g in items:
        t0 = time.perf_counter()
        hits = pipeline.search(g.question, DEPTH, g.lang)
        lat.append((time.perf_counter() - t0) * 1e3)
        ranked = unit_ranking(hits)
        rows.append(
            {
                "id": g.id,
                "lang": g.lang,
                "gold": g.gold,
                "ranked": ranked[:10],
                "recall@1": recall_at_k(ranked, g.gold, 1),
                "recall@5": recall_at_k(ranked, g.gold, 5),
                "recall@10": recall_at_k(ranked, g.gold, 10),
                "hit@5": hit_at_k(ranked, g.gold, 5),
                "mrr@10": mrr_at_k(ranked, g.gold, 10),
                "ndcg@10": ndcg_at_k(ranked, g.gold, 10),
            }
        )
    return rows, lat


def run_ablation(
    steps: list[Step],
    units: list[LegalUnit],
    golden: list[GoldenItem],
    *,
    embedder_factory: Callable[[], Embedder] | None,
    reranker_factory: Callable[[], Reranker] | None,
    log: Callable[[str], None] = print,
) -> list[StepResult]:
    items = answerable(golden, units)
    embedder = (
        embedder_factory() if embedder_factory and any(s.needs_dense for s in steps) else None
    )
    reranker = (
        reranker_factory() if reranker_factory and any(s.needs_reranker for s in steps) else None
    )
    results: list[StepResult] = []
    for step in steps:
        if (step.needs_dense and embedder is None) or (step.needs_reranker and reranker is None):
            results.append(
                StepResult(
                    step.name,
                    asdict(step.cfg),
                    0,
                    0,
                    0.0,
                    {},
                    {},
                    {},
                    skipped="requires an embedder/reranker not enabled",
                )
            )
            log(f"skip  {step.name}")
            continue
        t0 = time.perf_counter()
        pipeline, chunks = build_pipeline(
            units,
            step.cfg,
            embedder=embedder if step.needs_dense else None,
            reranker=reranker if step.needs_reranker else None,
        )
        index_s = time.perf_counter() - t0
        rows, lat = evaluate_pipeline(pipeline, items)
        by_lang = {lang: _score([r for r in rows if r["lang"] == lang]) for lang in ("en", "fr")}
        res = StepResult(
            step.name,
            asdict(step.cfg),
            len(rows),
            len(chunks),
            round(index_s, 2),
            _score(rows),
            by_lang,
            {"p50": round(percentile(lat, 50), 1), "p95": round(percentile(lat, 95), 1)},
            per_query=rows,
        )
        results.append(res)
        log(f"done  {step.name}: {res.metrics} latency={res.latency_ms}")
    return results


@dataclass
class GenerationResult:
    step: str
    llm: str
    judge: str
    n_answerable: int
    n_unanswerable: int
    faithfulness: float
    correctness: float
    citation_precision: float
    answer_rate_answerable: float
    abstain_rate_unanswerable: float
    p50_ms: float


def run_generation(
    pipeline: RetrievalPipeline,
    answerer: Answerer,
    judge: Judge,
    golden: list[GoldenItem],
    units: list[LegalUnit],
    step_name: str,
    top_k: int = 6,
) -> tuple[GenerationResult, list[dict[str, Any]]]:
    ans_items = answerable(golden, units)
    unans = [g for g in golden if not g.gold]
    rows: list[dict[str, Any]] = []
    lat: list[float] = []
    for g in [*ans_items, *unans]:
        t0 = time.perf_counter()
        hits = pipeline.search(g.question, top_k, g.lang)
        a = answerer.answer(g.question, g.lang, hits)
        lat.append((time.perf_counter() - t0) * 1e3)
        row: dict[str, Any] = {
            "id": g.id,
            "lang": g.lang,
            "answerable": bool(g.gold),
            "abstained": a.abstained,
            "answer": a.answer,
            "cited": [c.unit_id for c in a.citations],
        }
        if g.gold and not a.abstained:
            cited_text = [h.chunk.text for h in hits]
            j = judge.judge(g.question, a.answer, cited_text, g.reference)
            row.update(
                faithfulness=j.faithfulness,
                correctness=j.correctness,
                citation_precision=(
                    sum(u in g.gold for u in row["cited"]) / len(row["cited"])
                    if row["cited"]
                    else 0.0
                ),
            )
        rows.append(row)
    judged = [r for r in rows if "faithfulness" in r]

    def mean(k: str) -> float:
        return round(statistics.fmean(r[k] for r in judged), 4) if judged else 0.0

    ans_rows = [r for r in rows if r["answerable"]]
    un_rows = [r for r in rows if not r["answerable"]]
    res = GenerationResult(
        step=step_name,
        llm=answerer.llm.name,
        judge=judge.name,
        n_answerable=len(ans_rows),
        n_unanswerable=len(un_rows),
        faithfulness=mean("faithfulness"),
        correctness=mean("correctness"),
        citation_precision=mean("citation_precision"),
        answer_rate_answerable=round(
            sum(not r["abstained"] for r in ans_rows) / max(len(ans_rows), 1), 4
        ),
        abstain_rate_unanswerable=round(
            sum(r["abstained"] for r in un_rows) / max(len(un_rows), 1), 4
        ),
        p50_ms=round(percentile(lat, 50), 1),
    )
    return res, rows


# --------------------------------------------------------------------------- reporting


def write_chart(results: list[StepResult], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    done = [r for r in results if not r.skipped]
    metrics = [("recall@5", "Recall@5"), ("mrr@10", "MRR@10"), ("ndcg@10", "nDCG@10")]
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(10, 0.9 + 0.62 * len(done)), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    bar_h = 0.24
    for j, ((key, label), color) in enumerate(zip(metrics, colors, strict=True)):
        ys = [i + (j - 1) * (bar_h + 0.02) for i in range(len(done))]
        vals = [r.metrics[key] for r in done]
        ax.barh(ys, vals, height=bar_h, color=color, label=label, edgecolor=surface, linewidth=1)
        if key == "recall@5":  # selective direct labels: one metric only
            for y, v in zip(ys, vals, strict=True):
                ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=7.5, color=ink2)
    ax.set_yticks(range(len(done)))
    ax.set_yticklabels([r.name for r in done], fontsize=8.5, color=ink)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("score (unit-level, higher is better)", fontsize=8.5, color=ink2)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=8)
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title, fontsize=10, color=ink, loc="left", pad=18)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def write_report(
    results: list[StepResult],
    path: Path,
    *,
    title: str,
    setup: dict[str, str],
    chart_name: str,
    generation: GenerationResult | None = None,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"_Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC} by "
        "`evals/run_ablation.py` - numbers below are produced by this run, not edited._",
        "",
        "## Setup",
        "",
    ]
    lines += [f"- **{k}**: {v}" for k, v in setup.items()]
    lines += [f"- **python / platform**: {platform.python_version()} / {platform.platform()}", ""]
    lines += [
        "## Retrieval (unit level, language-filtered to the question language)",
        "",
        f"![ablation chart]({chart_name})",
        "",
        "| Step | Chunks | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if r.skipped:
            lines.append(f"| {r.name} | - | skipped: {r.skipped} | | | | | | |")
            continue
        m = r.metrics
        lines.append(
            f"| {r.name} | {r.n_chunks} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | "
            f"{m['recall@10']:.3f} | {m['mrr@10']:.3f} | {m['ndcg@10']:.3f} | "
            f"{r.latency_ms['p50']:.1f} | {r.latency_ms['p95']:.1f} |"
        )
    lines += [
        "",
        "### By question language (Recall@5 / MRR@10)",
        "",
        "| Step | EN R@5 | EN MRR | FR R@5 | FR MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        if r.skipped:
            continue
        en, fr = r.by_lang["en"], r.by_lang["fr"]
        lines.append(
            f"| {r.name} | {en['recall@5']:.3f} | {en['mrr@10']:.3f} | "
            f"{fr['recall@5']:.3f} | {fr['mrr@10']:.3f} |"
        )
    n = next((r.n_queries for r in results if not r.skipped), 0)
    lines += [
        "",
        f"Queries scored: **{n}** answerable golden items whose gold units are all "
        "present in the indexed corpus (unanswerable items are excluded from retrieval "
        "metrics). Latency = retrieval (+ rerank) per query on this machine, embeddings of "
        "documents cached, query embedding included.",
        "",
    ]
    if generation is not None:
        g = generation
        lines += [
            "## Generation (end-to-end)",
            "",
            f"Config: retrieval = `{g.step}`, LLM = `{g.llm}`, judge = `{g.judge}`.",
            "",
            "| Answerable | Unanswerable | Answer rate (answerable) | Abstain rate "
            "(unanswerable) | Faithfulness | Correctness | Citation precision | p50 ms |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {g.n_answerable} | {g.n_unanswerable} | {g.answer_rate_answerable:.3f} | "
            f"{g.abstain_rate_unanswerable:.3f} | {g.faithfulness:.3f} | {g.correctness:.3f} | "
            f"{g.citation_precision:.3f} | {g.p50_ms:.1f} |",
            "",
        ]
        if g.judge == "lexical-proxy":
            lines += [
                "> The lexical judge is a token-overlap proxy used so this runs offline. "
                "An extractive answerer is faithful by construction, so faithfulness here "
                "is a sanity check, not a quality claim. Use `--judge llm` for real "
                "scores, after calibrating the judge (evals/calibration/README.md).",
                "",
            ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def dump_json(results: list[StepResult], path: Path, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": [asdict(r) for r in results], **(extra or {})}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
