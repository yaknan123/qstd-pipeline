#!/usr/bin/env python3
"""
compare_pools.py
================

Side-by-side table of the per-device tiers across analysis sets, so a new set
(e.g. AVAIL, "rows where the data exist") can be read against the pool A
numbers it is meant to be compared with.

Reads, for each pool, what train_tiers.py and tier2a_sensor_only.py wrote:
    results/                    the default pool (A)
    results/pool_<name>/        any other pool
and writes results/pool_comparison.csv. Nothing is trained here.

    python experiments/compare_pools.py                 # A vs AVAIL
    python experiments/compare_pools.py --pools A B AVAIL
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import common as C


def pool_dir(pool: str) -> Path:
    return C.BASE_RESULTS_DIR if pool == C.DEFAULT_POOL else C.BASE_RESULTS_DIR / f"pool_{pool}"


def tier_rows(pool: str) -> list[dict]:
    """One row per (device, model) from task3 and, for the superconducting device, task5."""
    d = pool_dir(pool)
    rows = []
    t3 = d / "task3_train_tiers.json"
    if t3.exists():
        payload = json.loads(t3.read_text())
        for r in payload.get("per_device", []):
            for key, name in (("tier2", r["tier2_name"]), ("tier3", "Tier 3")):
                m = r[key]
                rows.append({
                    "pool": pool, "device": r["display"], "model": name,
                    "n_train": m["n_train"], "n_test": m["n_test"],
                    "n_features": m["n_features"],
                    "R2_holdout": m["r2_holdout"], "R2_cv": m["r2_cv_mean"],
                    "R2_cv_std": m["r2_cv_std"], "MAE": m["mae_holdout"],
                })
    t5 = d / "task5_sensor_only.json"
    if t5.exists():
        # Only the sensor-only the superconducting device model is new here; its Tier 2b and Tier 3
        # rows duplicate task3 on the same split.
        m = json.loads(t5.read_text()).get("tier2a")
        if m:
            rows.append({"pool": pool, "device": "the superconducting device", "model": "Tier 2a (sensor only)",
                         "n_train": m["n_train"], "n_test": m["n_test"],
                         "n_features": m["n_features"],
                         "R2_holdout": m["r2_holdout"], "R2_cv": m["r2_cv_mean"],
                         "R2_cv_std": m["r2_cv_std"], "MAE": m["mae_holdout"]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", nargs="+", default=[C.DEFAULT_POOL, C.AVAIL_POOL],
                    choices=C.POOL_CHOICES)
    args = ap.parse_args()

    rows = [r for p in args.pools for r in tier_rows(p)]
    if not rows:
        print("no results found; run train_tiers.py --pool <name> first")
        return 1
    df = pd.DataFrame(rows)

    # DeltaR2 of each tier over the Tier 3 trained on the same rows and split.
    base = (df[df.model == "Tier 3"]
            .set_index(["pool", "device"])[["R2_holdout", "R2_cv"]])
    df["dR2_holdout_vs_tier3"] = [
        r.R2_holdout - base.loc[(r.pool, r.device), "R2_holdout"]
        if (r.pool, r.device) in base.index else float("nan") for r in df.itertuples()]
    df["dR2_cv_vs_tier3"] = [
        r.R2_cv - base.loc[(r.pool, r.device), "R2_cv"]
        if (r.pool, r.device) in base.index else float("nan") for r in df.itertuples()]

    missing = [p for p in args.pools if p not in set(df.pool)]
    if missing:
        print(f"  note: no results yet for pool(s) {missing}")

    df = df.sort_values(["device", "model", "pool"],
                        key=lambda s: s.map({p: i for i, p in enumerate(args.pools)})
                        if s.name == "pool" else s).reset_index(drop=True)
    out = df.round(4)
    print(out.to_string(index=False))
    path = C.BASE_RESULTS_DIR / "pool_comparison.csv"
    out.to_csv(path, index=False)
    print(f"\n  wrote {path}")
    print("  dR2 within CV std (R2_cv_std) is not distinguishable from 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
