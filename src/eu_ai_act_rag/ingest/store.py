"""JSONL persistence for parsed legal units."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from eu_ai_act_rag.schemas import LegalUnit


def write_units(units: Iterable[LegalUnit], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for u in units:
            fh.write(u.model_dump_json() + "\n")
            n += 1
    return n


def read_units(path: Path) -> list[LegalUnit]:
    with path.open(encoding="utf-8") as fh:
        return [LegalUnit.model_validate_json(line) for line in fh if line.strip()]
