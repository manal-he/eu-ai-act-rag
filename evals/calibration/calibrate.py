"""Agreement between human labels and judge scores.

python evals/calibration/calibrate.py human_labels.jsonl judged_rows.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from eu_ai_act_rag.evaluation.judge import agreement


def main(labels_path: str, judged_path: str) -> None:
    human = {}
    for line in Path(labels_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            human[row["id"]] = row
    judged = {r["id"]: r for r in json.loads(Path(judged_path).read_text(encoding="utf-8"))}
    ids = [i for i in human if i in judged and "faithfulness" in judged[i]]
    if not ids:
        sys.exit("no overlapping judged rows")
    for crit in ("faithfulness", "correctness"):
        stats = agreement([judged[i][crit] for i in ids], [human[i][crit] for i in ids])
        print(crit, {k: round(v, 3) for k, v in stats.items()})


if __name__ == "__main__":
    main(*sys.argv[1:3])
