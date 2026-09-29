#!/usr/bin/env python3
"""
forward_split.py — does the telemetry still help when the test set is the future?
================================================================================

HOST: wherever the dataset is. No database, no monitoring system.

THE QUESTION
------------
Sensor and calibration values drift slowly, so they carry information about
*when* a circuit ran. Fidelity also drifts. A random split lets a model exploit
that: it can recognise the period a holdout circuit belongs to, because circuits
from the same days are in the training set, and predict the fidelity typical of
that period without learning anything physical.

A forward split removes exactly that shortcut. Train on the earlier months, test
on the later ones, so nothing in training shares a period with anything in test.

    if the telemetry contribution is period identification, it should
    collapse here, and may go negative.

    if it survives, the contribution is something the sensors measure
    about the machine, not about the calendar.

This is the experiment that settles the interpretation the paper argues for, and
it is the one a reviewer is most likely to ask for.

WHAT IT REPORTS
---------------
For each device and each analysis set, Tier 3 (circuit only) against Tier 2a
(+ sensors) and Tier 2b (+ calibration), under two splits:

    random    the usual 80/20 grouped by job          -- the paper's setting
    forward   trained on the earliest 80% of months, tested on the latest 20%

The number to read is how DeltaR^2 changes between the two.

Absolute R^2 will fall under the forward split for reasons that have nothing to
do with telemetry: later months contain different workloads. That is expected and
is not the point; the comparison is DeltaR^2 against DeltaR^2.

USAGE
-----
    python experiments/forward_split.py
    python experiments/forward_split.py --pools A AVAIL --holdout-frac 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MIN_ROWS = 1000


def device_features(device: str, columns: list[str]):
    sensors_by_dev, calib_by_dev = C.resolve_device_columns(columns)
    sensors = sensors_by_dev.get(device, [])
    calib = calib_by_dev.get(device, [])
    return list(C.CIRCUIT_FEATURES) + sensors + calib, sensors, calib


def forward_indices(df: pd.DataFrame, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Train on the earliest months, test on the latest, splitting whole jobs.

    The cut is placed on the completion time, and every circuit of a job stays on
    one side, so a job straddling the boundary cannot leak.
    """
    t = pd.to_datetime(df[C.TIME_COL], utc=True, errors="coerce")
    job_time = t.groupby(df[C.GROUP_COL]).transform("min")
    cut = job_time.quantile(1 - frac)
    test = (job_time > cut).to_numpy()
    return np.where(~test)[0], np.where(test)[0]


def evaluate(device: str, pool: str, holdout_frac: float) -> list[dict]:
    columns = C.dataset_columns(C.DEFAULT_DATASET)
    features, sensors, calib = device_features(device, columns)
    load_cols = features + [C.TARGET_COL, C.DEVICE_COL, C.GROUP_COL, C.TIME_COL]
    if pool == C.AVAIL_POOL:
        load_cols.append(C.ROW_ID_COL)

    df = C.load_dataset(load_cols, dataset=C.DEFAULT_DATASET)
    df = df[df[C.DEVICE_COL] == device].reset_index(drop=True)
    df = C.apply_pool(df, device, pool)
    df = C.prepare_frame(df, features)
    if len(df) < MIN_ROWS:
        print(f"    only {len(df):,} rows; skipping")
        return []

    out = []
    for split in ("random", "forward"):
        if split == "random":
            tr, te = C.group_holdout_split(df)
        else:
            tr, te = forward_indices(df, holdout_frac)
        if len(te) < MIN_ROWS or len(tr) < MIN_ROWS:
            print(f"    {split}: too few rows either side; skipping")
            continue
        train_df, test_df = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
        kept, _ = C.coverage_filter(train_df, features)
        kept_sensors = [f for f in kept if f in set(sensors)]
        kept_calib = [f for f in kept if f in set(calib)]

        tiers = {"Tier 3": list(C.CIRCUIT_FEATURES)}
        if kept_sensors:
            tiers["Tier 2a"] = list(C.CIRCUIT_FEATURES) + kept_sensors
        if kept_calib:
            tiers["Tier 2b"] = list(C.CIRCUIT_FEATURES) + kept_sensors + kept_calib

        scores = {t: C.fit_and_score_lgbm(train_df, test_df, f, run_cv=False)
                  for t, f in tiers.items()}
        base = scores["Tier 3"].r2_holdout
        t_train = pd.to_datetime(train_df[C.TIME_COL], utc=True, errors="coerce")
        t_test = pd.to_datetime(test_df[C.TIME_COL], utc=True, errors="coerce")
        for tier, m in scores.items():
            out.append({
                "pool": pool, "device": C.DEVICE_DISPLAY.get(device, device),
                "split": split, "tier": tier, "n_features": len(tiers[tier]),
                "n_train": len(train_df), "n_test": len(test_df),
                "train_last_month": t_train.max().strftime("%Y-%m"),
                "test_first_month": t_test.min().strftime("%Y-%m"),
                "r2": m.r2_holdout, "mae": m.mae_holdout,
                "delta_r2_vs_tier3": m.r2_holdout - base,
            })
        top = "Tier 2b" if "Tier 2b" in scores else "Tier 2a"
        print(f"    {split:<8}: Tier 3 {base:.4f}  {top} {scores[top].r2_holdout:.4f}  "
              f"delta {scores[top].r2_holdout - base:+.4f}  "
              f"(train to {t_train.max():%Y-%m}, test from {t_test.min():%Y-%m})", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", nargs="+", default=["A", "B", C.AVAIL_POOL])
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    args = ap.parse_args()

    C.set_all_seeds()
    print(f"  dataset : {C.DEFAULT_DATASET}")
    print(f"  forward : latest {args.holdout_frac:.0%} of the period is the test set\n")

    rows: list[dict] = []
    for pool in args.pools:
        for device in (C.SC, C.ION):
            print(f"  {C.DEVICE_DISPLAY.get(device, device)}, pool {pool}")
            rows += evaluate(device, pool, args.holdout_frac)

    if not rows:
        print("  nothing evaluated")
        return 1

    frame = pd.DataFrame(rows)
    out_dir = C.BASE_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "task10_forward_split.csv", index=False)

    print("\n" + "=" * 78)
    print("  DeltaR^2 vs Tier 3: random split against forward-in-time split")
    print("=" * 78)
    d = frame[frame.tier != "Tier 3"]
    wide = d.pivot_table(index=["pool", "device", "tier"], columns="split",
                         values="delta_r2_vs_tier3")
    if {"random", "forward"} <= set(wide.columns):
        wide["change"] = wide["forward"] - wide["random"]
    print(wide.round(4).to_string())
    print("\n  A contribution that is period identification should shrink or turn")
    print("  negative in the 'forward' column.")
    C.save_results("task10_forward_split", {"rows": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
