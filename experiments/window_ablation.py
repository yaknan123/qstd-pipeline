#!/usr/bin/env python3
"""
window_ablation.py
==================

Task 7 — alignment window-size ablation (PAPER_OBJECTIVES.md Section 3.5).
Numbers feed Appendix B, Table VII.

THE EXPERIMENT
--------------
For each lookback window length w in {2, 5, 10, 20} minutes:

  1. Re-run the the monitoring system join with a lookback window of length w      <- prepare_windows.sh
  2. Re-aggregate sensor features (mean, min, max, sd)            <- same script
  3. Retrain the superconducting device Tier 2b  (LightGBM)                            <- this script
  4. Retrain the trapped-ion device Tier 2a (LightGBM)                            <- this script

Eight training runs total, plus Tier 3 once per device. Tier 3 uses no sensor
features, so it is invariant across windows: it is trained a single time per
device and reused as the baseline for every window's DeltaR^2.

TWO-STAGE DESIGN
----------------
Steps 1-2 need the monitoring system access and run on the LRZ node; steps 3-4 only need the
resulting parquets. So this script does not shell out to the extractor. It
takes one already-built dataset per window:

    python experiments/window_ablation.py \
        --window 5=data/stage10_cleaned_final_dataset.parquet \
        --window 2=data/windows/window_02/stage10_cleaned_final_dataset.parquet \
        --window 10=data/windows/window_10/stage10_cleaned_final_dataset.parquet \
        --window 20=data/windows/window_20/stage10_cleaned_final_dataset.parquet

The per-window extractions run on LRZ
(sensor_calibration_query_pipeline/scripts/run_window_extractions.sh);
experiments/prepare_windows.sh turns them into these stage 10 parquets.
Every window trains on the same month pool (--pool, default A). Windows
whose dataset is absent are reported as pending rather than silently skipped,
so a partial table is never mistaken for a complete one.

COMPARABILITY
-------------
Every window uses the same seed, the same GroupShuffleSplit and the same
coverage threshold. Only the sensor values change between runs, which is the
point of the ablation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import common as C

DEFAULT_WINDOWS = (2, 5, 10, 20)
MIN_ROWS = 500


def parse_window_arg(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--window expects MINUTES=PATH, got {value!r}"
        )
    minutes, path = value.split("=", 1)
    return int(minutes), Path(path)


def tier_for(device: str) -> str:
    return "Tier 2b" if device == C.QEXA else "Tier 2a"


def run_one(device, dataset, limit, run_cv, tier3_cache, pool=None):
    """Train this device's Tier 2 on one window's dataset.

    Tier 3 is cached per device: it uses no sensor features, so re-fitting it
    for every window would burn compute to reproduce the same number.
    """
    label = C.DEVICE_DISPLAY.get(device, device)
    columns = C.dataset_columns(dataset)
    sensor_cols, calib_cols = C.resolve_device_columns(columns)
    sensors = sensor_cols.get(device, [])
    calib = calib_cols.get(device, [])
    features = list(C.CIRCUIT_FEATURES) + sensors + calib
    C.assert_no_cross_device_leakage(device, features, sensor_cols, calib_cols)

    df = C.load_dataset(
        features + [C.TARGET_COL, C.DEVICE_COL, C.GROUP_COL, C.TIME_COL],
        dataset=dataset,
        limit=limit,
    )
    df = df[df[C.DEVICE_COL] == device].reset_index(drop=True)
    df = C.apply_pool(df, device, pool)
    df = C.prepare_frame(df, features)
    if len(df) < MIN_ROWS:
        print(f"    {label}: only {len(df):,} rows; skipping")
        return None

    train_idx, test_idx = C.group_holdout_split(df)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    kept, _ = C.coverage_filter(train_df, features)
    kept_sensors = [f for f in kept if f in set(sensors)]
    kept_calib = [f for f in kept if f in set(calib)]

    m2 = C.fit_and_score_lgbm(train_df, test_df, kept, run_cv=run_cv)
    print(f"    {label} {tier_for(device)}: {m2.summary()}")

    if device not in tier3_cache:
        m3 = C.fit_and_score_lgbm(
            train_df, test_df, C.tier3_feature_columns(), run_cv=run_cv
        )
        tier3_cache[device] = m3
        print(f"    {label} Tier 3 (window-invariant): {m3.summary()}")
    m3 = tier3_cache[device]

    return {
        "device": device,
        "display": label,
        "tier": tier_for(device),
        "n_sensor_features": len(kept_sensors),
        "n_calib_features": len(kept_calib),
        "tier2": m2.to_dict(),
        "tier3": m3.to_dict(),
        "delta_r2": C.delta_r2(m2.r2_holdout, m3.r2_holdout),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window",
        action="append",
        type=parse_window_arg,
        metavar="MINUTES=PATH",
        help="A window length and the dataset built with it; repeatable",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Read only the first N rows (smoke tests only)")
    parser.add_argument("--no-cv", action="store_true", help="Skip GroupKFold CV")
    parser.add_argument("--pool", choices=C.POOL_CHOICES, default=C.DEFAULT_POOL,
                        action=C.PoolAction,
                        help="Month pool (see common.MONTH_POOLS) or AVAIL; default A")
    args = parser.parse_args()

    C.set_all_seeds()

    windows = dict(args.window or [])
    if not windows:
        parser.error(
            "no windows given. Pass --window MINUTES=PATH for each of "
            f"{DEFAULT_WINDOWS}; see experiments/prepare_windows.sh"
        )

    print("=" * 74)
    print("TASK 7 — alignment window-size ablation")
    print("=" * 74)

    missing = [w for w in DEFAULT_WINDOWS if w not in windows]
    if missing:
        print(f"  PENDING: no dataset supplied for window(s) {missing} minutes. "
              f"The Table VII row(s) will be incomplete.")

    rows = []
    results = []
    tier3_cache: dict[str, C.RegressionMetrics] = {}
    run_cv = not args.no_cv

    for minutes in sorted(windows):
        dataset = windows[minutes]
        print(f"\n  window = {minutes} min  ({dataset})")
        if not dataset.exists():
            print(f"    PENDING: dataset not found; run prepare_windows.sh first")
            continue
        C.warn_if_alignment_stale(C.dataset_columns(dataset))

        for device in C.PAPER_DEVICES:
            result = run_one(device, dataset, args.limit, run_cv, tier3_cache, args.pool)
            if not result:
                continue
            result["window_min"] = minutes
            results.append(result)
            rows.append(
                {
                    "window_min": minutes,
                    "device": result["display"],
                    "tier": result["tier"],
                    "n_features": result["tier2"]["n_features"],
                    "R2_holdout": round(result["tier2"]["r2_holdout"], 4),
                    "R2_tier3": round(result["tier3"]["r2_holdout"], 4),
                    "delta_R2_vs_tier3": round(result["delta_r2"], 4),
                    "MAE": round(result["tier2"]["mae_holdout"], 4),
                    "RMSE": round(result["tier2"]["rmse_holdout"], 4),
                }
            )

    if not rows:
        print("\n  no windows produced results; nothing written")
        return 1

    table = pd.DataFrame(rows).sort_values(["device", "window_min"])
    print("\n" + "=" * 74)
    print("  WINDOW ABLATION SUMMARY (Appendix B, Table VII)")
    print("=" * 74)
    print(table.to_string(index=False))
    C.save_table("task7_window_ablation", table)

    best = {}
    for device in table["device"].unique():
        sub = table[table["device"] == device]
        row = sub.loc[sub["delta_R2_vs_tier3"].idxmax()]
        best[device] = {
            "window_min": int(row["window_min"]),
            "delta_r2": float(row["delta_R2_vs_tier3"]),
        }
        print(f"\n  {device}: largest DeltaR^2 at w={int(row['window_min'])} min "
              f"({row['delta_R2_vs_tier3']:+.4f})")

    C.save_results(
        "task7_window_ablation",
        {
            "pool": args.pool,
            "windows_requested": list(DEFAULT_WINDOWS),
            "windows_run": sorted(windows),
            "windows_missing": missing,
            "best_window_per_device": best,
            "results": results,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
