#!/usr/bin/env python3
"""
check_results.py — compare a reproduction run against the published numbers
==========================================================================

Every value in `expected_results.csv` was produced by the code in this
repository on the published dataset. A reproduction should land on the same
numbers: the models are seeded and the splits are deterministic, so the only
expected source of drift is a different library version or CPU arithmetic. The
tolerances in that file are wider than any drift seen in practice and narrower
than every effect the paper reports.

    uv run python scripts/check_results.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
EXPECTED = Path(__file__).with_name("expected_results.csv")
RESULTS = Path(os.environ.get("MQSS_RESULTS_DIR", REPO / "experiments" / "results"))

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def read_value(source: str, selector: str) -> float | None:
    """Pull one number out of a result file. `selector` is format specific."""
    path = RESULTS / source
    if not path.exists():
        return None
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        for key in selector.split("."):
            if isinstance(data, list):
                data = data[int(key)]
            elif key in data:
                data = data[key]
            else:
                return None
        return float(data)
    frame = pd.read_csv(path)
    column, *filters = selector.split("|")
    for f in filters:
        col, val = f.split("=", 1)
        frame = frame[frame[col].astype(str) == val]
    if frame.empty or column not in frame:
        return None
    return float(frame[column].iloc[0])


def main() -> int:
    if not EXPECTED.exists():
        print(f"  {YELLOW}no expected_results.csv alongside this script{RESET}")
        print("  Nothing to compare against; the run's own output is in")
        print(f"  {RESULTS}")
        return 0

    expected = pd.read_csv(EXPECTED)
    ok = bad = missing = 0
    width = max(len(str(r.label)) for r in expected.itertuples())

    for row in expected.itertuples():
        got = read_value(row.source, row.selector)
        label = str(row.label).ljust(width)
        if got is None:
            print(f"  {YELLOW}skip{RESET}  {label}  not produced (step skipped?)")
            missing += 1
            continue
        delta = abs(got - row.value)
        if delta <= row.tolerance:
            print(f"  {GREEN}ok{RESET}    {label}  {got:.4f}")
            ok += 1
        else:
            print(f"  {RED}DIFF{RESET}  {label}  got {got:.4f}, published {row.value:.4f} "
                  f"(delta {delta:.4f} > {row.tolerance})")
            bad += 1

    print(f"\n  {ok} match, {bad} differ, {missing} not run")
    if bad:
        print(f"\n  {RED}Results differ from the published numbers.{RESET} Check the library")
        print("  versions first (uv.lock pins them), then that data/ holds the published")
        print("  dataset and not a modified copy.")
        return 1
    if not missing:
        print(f"\n  {GREEN}Everything reproduces.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
