"""Command line: `euai ingest | make-fixture | search | ask`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from eu_ai_act_rag.config import PROJECT_ROOT, get_settings

PROCESSED = PROJECT_ROOT / "data" / "processed" / "corpus.jsonl"
FIXTURE = PROJECT_ROOT / "data" / "fixtures" / "corpus_fixture.jsonl"

# Units kept in the committed offline fixture: the most-asked-about provisions, chosen so that
# a meaningful subset of the golden set is answerable offline. Everything else is downloaded.
FIXTURE_UNITS: tuple[str, ...] = (
    *(
        f"ai_act:art_{n}"
        for n in (2, 3, 4, 5, 6, 9, 10, 12, 13, 14, 26, 27, 50, 51, 53, 55, 73, 99, 113)
    ),
    "ai_act:anx_III",
    *(f"gdpr:art_{n}" for n in (4, 5, 6, 7, 9, 12, 15, 17, 20, 22, 25, 32, 33, 35, 37, 83)),
)


def cmd_ingest(args: argparse.Namespace) -> int:
    from eu_ai_act_rag.ingest.download import download_all
    from eu_ai_act_rag.ingest.parser import parse_file
    from eu_ai_act_rag.ingest.store import write_units

    settings = get_settings()
    paths = download_all(settings.raw_dir, force=args.force)
    units = [u for p in paths for u in parse_file(p)]
    n = write_units(units, Path(args.out))
    print(f"Wrote {n} units to {args.out}")
    print("Set CORPUS_PATH to this file (see .env.example) to serve the full corpus.")
    return 0


def cmd_make_fixture(args: argparse.Namespace) -> int:
    from eu_ai_act_rag.ingest.store import read_units, write_units

    wanted = set(FIXTURE_UNITS)
    units = [u for u in read_units(Path(args.src)) if u.unit_id in wanted]
    n = write_units(units, Path(args.out))
    print(f"Wrote {n} units ({len(wanted)} ids x languages) to {args.out}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from eu_ai_act_rag.pipeline import build_service

    svc = build_service(get_settings())
    lang, hits = svc.search(args.query, args.k)
    print(f"[lang={lang}] {svc.pipeline.retriever.name}")
    for h in hits:
        print(f"{h.rank:>2}. {h.score:8.4f}  {h.chunk.citation_label:<28} {h.chunk.text[:90]!r}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from eu_ai_act_rag.pipeline import build_service

    svc = build_service(get_settings())
    ans = svc.ask(args.question)
    print(json.dumps(ans.model_dump(exclude={"retrieval"}), ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="euai", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="download EUR-Lex FR/EN texts and parse them")
    s.add_argument("--out", default=str(PROCESSED))
    s.add_argument("--force", action="store_true", help="re-download even if cached")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("make-fixture", help="extract the small committed fixture")
    s.add_argument("--src", default=str(PROCESSED))
    s.add_argument("--out", default=str(FIXTURE))
    s.set_defaults(fn=cmd_make_fixture)

    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=6)
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("ask")
    s.add_argument("question")
    s.set_defaults(fn=cmd_ask)

    args = p.parse_args(argv)
    rc: int = args.fn(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
