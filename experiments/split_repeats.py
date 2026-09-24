#!/usr/bin/env python3
"""
split_repeats.py — every DeltaR^2 in the paper, as mean +/- sd over five splits
==============================================================================

WHY
---
The paper's central question is whether system telemetry improves fidelity
prediction, and the answer is a difference of a few thousandths of R^2. A single
80/20 group split does not resolve a difference that small: which jobs land in
the holdout moves a DeltaR^2 by about as much as the DeltaR^2 is. Reporting one
split therefore reports split luck.

This script runs the same comparison over the five splits in
`common.SPLIT_SEEDS`, with everything else held fixed (same pool, same coverage
filter, same seeded model), and reports the mean and the standard deviation. An
interval that contains zero says plainly that the effect is not resolved.

It replaces single-split point estimates for:
  Table III  per-device Tier 2 vs Tier 3
  Table V    the superconducting device feature-category decomposition (sensor, then calibration)
  Table VII  the sensor alignment window ablation

WHAT IT DOES NOT DO
-------------------
No cross-validation. CV measures variation inside the training part; this
measures variation of the reported holdout number itself, which is the quantity
the paper needs. The two are complementary and both are reported.

USAGE
-----
    python experiments/split_repeats.py                      # pools A, B, AVAIL
    python experiments/split_repeats.py --pools A            # one set
    python experiments/split_repeats.py --window 2=<parquet> --window 10=<parquet>

Writes to the results directory:
    task9_split_repeats.csv    one row per (pool, device, tier, split)
    task9_split_summary.csv    mean +/- sd per (pool, device, tier)
    task9_split_repeats.json   the same, plus the settings used
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MIN_DEVICE_ROWS = 1000


def device_features(device: str, columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    sensors_by_dev, calib_by_dev = C.resolve_device_columns(columns)
    sensors = sensors_by_dev.get(device, [])
    calib = calib_by_dev.get(device, [])
    return list(C.CIRCUIT_FEATURES) + sensors + calib, sensors, calib


def evaluate(device: str, pool: str | None, dataset: Path, window: int | None = None) -> list[dict]:
    """Tier 3 / 2a / 2b on one device and pool, once per split seed."""
    columns = C.dataset_columns(dataset)
    features, sensors, calib = device_features(device, columns)
    load_cols = features + [C.TARGET_COL, C.DEVICE_COL, C.GROUP_COL, C.TIME_COL]
    if pool == C.AVAIL_POOL:
        load_cols.append(C.ROW_ID_COL)

    df = C.load_dataset(load_cols, dataset=dataset)
    df = df[df[C.DEVICE_COL] == device].reset_index(drop=True)
    df = C.apply_pool(df, device, pool)
    df = C.prepare_frame(df, features)
    if len(df) < MIN_DEVICE_ROWS:
        print(f"    only {len(df):,} rows; skipping")
        return []

    rows = []
    for seed in C.SPLIT_SEEDS:
        train_idx, test_idx = C.group_holdout_split(df, seed=seed)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # The coverage filter is refit per split, exactly as the single-split
        # scripts do, so a feature is never kept on the strength of holdout rows.
        kept, _ = C.coverage_filter(train_df, features)
        kept_sensors = [f for f in kept if f in set(sensors)]
        kept_calib = [f for f in kept if f in set(calib)]

        tiers = {"Tier 3": list(C.CIRCUIT_FEATURES)}
        if kept_sensors:
            tiers["Tier 2a"] = list(C.CIRCUIT_FEATURES) + kept_sensors
        if kept_calib:
            tiers["Tier 2b"] = list(C.CIRCUIT_FEATURES) + kept_sensors + kept_calib

        scores = {tier: C.fit_and_score_lgbm(train_df, test_df, feats, run_cv=False)
                  for tier, feats in tiers.items()}
        base = scores["Tier 3"].r2_holdout
        for tier, m in scores.items():
            rows.append({
                "pool": pool or "all", "device": C.DEVICE_DISPLAY.get(device, device),
                "window_min": window, "tier": tier, "split_seed": seed,
                "n_features": len(tiers[tier]), "n_train": len(train_df),
                "n_holdout": len(test_df), "r2": m.r2_holdout, "mae": m.mae_holdout, "rmse": m.rmse_holdout,
                "delta_r2_vs_tier3": m.r2_holdout - base,
            })
        top = "Tier 2b" if "Tier 2b" in scores else "Tier 2a"
        print(f"    seed {seed:>6}: Tier 3 {base:.4f}  {top} {scores[top].r2_holdout:.4f}  "
              f"delta {scores[top].r2_holdout - base:+.4f}  holdout {len(test_df):,}",
              flush=True)
    return rows


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["pool", "device", "window_min", "tier"], dropna=False)
    out = grouped.agg(
        n_splits=("r2", "size"),
        n_features=("n_features", "median"),
        r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
        mae_mean=("mae", "mean"),
        delta_mean=("delta_r2_vs_tier3", "mean"),
        delta_sd=("delta_r2_vs_tier3", "std"),
        delta_min=("delta_r2_vs_tier3", "min"),
        delta_max=("delta_r2_vs_tier3", "max"),
    ).reset_index()
    # The question the paper asks of every one of these numbers.
    out["resolved"] = np.where(
        out["tier"] == "Tier 3", "",
        np.where((out["delta_mean"] - out["delta_sd"] > 0)
                 | (out["delta_mean"] + out["delta_sd"] < 0), "yes", "no"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", nargs="+", default=["A", "B", C.AVAIL_POOL])
    ap.add_argument("--dataset", default=str(C.DEFAULT_DATASET))
    ap.add_argument("--window", action="append", default=[],
                    metavar="MIN=PARQUET",
                    help="Extra alignment window, e.g. --window 2=data/windows/window_02/...")
    args = ap.parse_args()

    C.set_all_seeds()
    print(f"  dataset  : {args.dataset}")
    print(f"  splits   : {len(C.SPLIT_SEEDS)} x 80/20 grouped by {C.GROUP_COL} "
          f"(seeds {', '.join(str(s) for s in C.SPLIT_SEEDS)})")
    print(f"  pools    : {', '.join(args.pools)}")

    windows: list[tuple[int | None, Path]] = [(None, Path(args.dataset))]
    for spec in args.window:
        minutes, _, path = spec.partition("=")
        windows.append((int(minutes), Path(path)))

    rows: list[dict] = []
    for pool in args.pools:
        for device in (C.QEXA, C.MARMOT):
            for window, path in windows:
                tag = "" if window is None else f", window {window} min"
                print(f"\n  {C.DEVICE_DISPLAY.get(device, device)}, pool {pool}{tag}")
                rows += evaluate(device, pool, path, window)

    if not rows:
        print("  nothing evaluated")
        return 1

    frame = pd.DataFrame(rows)
    summary = summarise(frame)

    out_dir = C.BASE_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "task9_split_repeats.csv", index=False)
    summary.to_csv(out_dir / "task9_split_summary.csv", index=False)
    C.save_results("task9_split_repeats",
                   {"split_seeds": list(C.SPLIT_SEEDS), "pools": args.pools,
                    "dataset": args.dataset,
                    "summary": summary.to_dict(orient="records")})

    print("\n" + "=" * 74)
    print("  DeltaR^2 vs Tier 3, mean +/- sd over "
          f"{len(C.SPLIT_SEEDS)} group splits")
    print("=" * 74)
    show = summary[summary["tier"] != "Tier 3"].copy()
    show["delta"] = show.apply(
        lambda r: f"{r.delta_mean:+.4f} +/- {r.delta_sd:.4f}", axis=1)
    show["range"] = show.apply(
        lambda r: f"[{r.delta_min:+.4f}, {r.delta_max:+.4f}]", axis=1)
    cols = ["pool", "device", "window_min", "tier", "r2_mean", "delta", "range", "resolved"]
    print(show[cols].to_string(index=False))
    print("\n  'resolved' = does mean +/- 1 sd exclude zero?")
    print(f"  wrote {out_dir / 'task9_split_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
