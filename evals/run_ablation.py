"""Run the retrieval ablation (and optionally the generation eval) and write a report.

Offline (zero keys, committed fixture, BM25 + hashing vectors):
    python evals/run_ablation.py --profile offline

Full (downloaded corpus, multilingual-e5 + cross-encoder; needs `pip install -e .[local]`):
    python evals/run_ablation.py --profile full --corpus data/processed/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from eu_ai_act_rag.config import PROJECT_ROOT, Settings
from eu_ai_act_rag.embeddings.providers import build_embedder
from eu_ai_act_rag.evaluation.ablation import (
    FULL_STEPS,
    OFFLINE_STEPS,
    answerable,
    dump_json,
    load_golden,
    run_ablation,
    run_generation,
    write_chart,
    write_report,
)
from eu_ai_act_rag.evaluation.judge import LexicalJudge, LLMJudge
from eu_ai_act_rag.generation.answerer import Answerer
from eu_ai_act_rag.ingest.store import read_units
from eu_ai_act_rag.llm.providers import build_llm
from eu_ai_act_rag.pipeline import build_pipeline
from eu_ai_act_rag.retrieval.rerank import build_reranker

GOLDEN = PROJECT_ROOT / "evals" / "golden" / "golden_v1.jsonl"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--profile", choices=["offline", "full"], default="offline")
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--golden", type=Path, default=GOLDEN)
    p.add_argument("--embedder", default=None, help="hashing | local | vertex | openai")
    p.add_argument("--reranker", default=None, help="local | cohere | vertex")
    p.add_argument("--generation", action="store_true", help="also run the end-to-end eval")
    p.add_argument("--llm", default="mock", help="generation LLM provider")
    p.add_argument("--judge", choices=["lexical", "llm"], default="lexical")
    p.add_argument("--judge-llm", default="anthropic")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    settings = Settings()
    offline = args.profile == "offline"
    corpus = args.corpus or (
        settings.corpus_path if offline else PROJECT_ROOT / "data" / "processed" / "corpus.jsonl"
    )
    embedder_name = args.embedder or ("hashing" if offline else "local")
    reranker_name = args.reranker or ("none" if offline else "local")
    steps = OFFLINE_STEPS if offline else FULL_STEPS
    out = args.out or PROJECT_ROOT / "evals" / "results" / args.profile

    units = read_units(corpus)
    golden = load_golden(args.golden)
    items = answerable(golden, units)
    print(f"{len(units)} units, {len(golden)} golden items, {len(items)} answerable in corpus")

    results = run_ablation(
        steps,
        units,
        golden,
        embedder_factory=lambda: build_embedder(settings, embedder_name),
        reranker_factory=(
            None if reranker_name == "none" else lambda: build_reranker(settings, reranker_name)
        ),
    )

    gen = None
    if args.generation:
        best = max((r for r in results if not r.skipped), key=lambda r: r.metrics["mrr@10"])
        step = next(s for s in steps if s.name == best.name)
        pipeline, _ = build_pipeline(
            units,
            step.cfg,
            embedder=build_embedder(settings, embedder_name) if step.needs_dense else None,
            reranker=build_reranker(settings, reranker_name) if step.needs_reranker else None,
        )
        judge = (
            LexicalJudge()
            if args.judge == "lexical"
            else LLMJudge(build_llm(settings, args.judge_llm))
        )
        gen, rows = run_generation(
            pipeline, Answerer(build_llm(settings, args.llm)), judge, golden, units, best.name
        )
        (out / "generation_rows.json").parent.mkdir(parents=True, exist_ok=True)
        (out / "generation_rows.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print("generation:", asdict(gen))

    shown = (
        corpus.relative_to(PROJECT_ROOT) if corpus.is_relative_to(PROJECT_ROOT) else corpus
    ).as_posix()
    setup = {
        "profile": args.profile,
        "corpus": f"`{shown}` - {len(units)} units (FR+EN)",
        "golden set": f"`{args.golden.name}` - {len(golden)} items (v1, single author, unreviewed)",
        "embedder": embedder_name
        + (
            " (feature hashing: a lexical stand-in, NOT semantic)"
            if embedder_name == "hashing"
            else f" ({settings.local_embedding_model})"
            if embedder_name == "local"
            else ""
        ),
        "reranker": reranker_name
        + (f" ({settings.reranker_model})" if reranker_name == "local" else ""),
        "fusion": "RRF k=60 over top-40 per retriever (weighted: min-max, alpha=0.5)",
    }
    title = f"Retrieval ablation - {args.profile} profile"
    write_chart(results, out / "ablation.png", title)
    write_report(
        results,
        out / "report.md",
        title=title,
        setup=setup,
        chart_name="ablation.png",
        generation=gen,
    )
    dump_json(
        results, out / "results.json", {"setup": setup, "generation": asdict(gen) if gen else None}
    )
    print(f"Report written to {out / 'report.md'}")


if __name__ == "__main__":
    main()
