#!/usr/bin/env python3
"""
lodo.py
=======

Task 6 — leave-one-device-out evaluation (PAPER_OBJECTIVES.md Section 3.4).

WHAT IT TRAINS
--------------
  Q-Exa -> Marmot : train Tier 1 LightGBM on Q-Exa circuits only (~1.28M rows),
                    test on all Marmot circuits
  Marmot -> Q-Exa : train on Marmot only (~100K rows), test on all Q-Exa

Both use the 22 device-agnostic Tier 1 features (21 circuit + device ID) with
the same hyperparameters and seed as the pooled Tier 1 model, and both are
compared against the pooled Tier 1 R^2.

WHY THE DEVICE ID STAYS IN
--------------------------
Section 3.4 specifies the same 23 features as pooled Tier 1, so `device_encoded`
is kept for comparability. It is constant within the training set and takes an
unseen value at test time, which is exactly the transfer condition being
measured: LightGBM never finds a useful split on a constant column, so the
model is effectively forced to rely on circuit structure alone. The encoding is
built from the pooled frame (not per-split) so that "Q-Exa" maps to the same
integer in both directions.

READING THE RESULT
------------------
A large drop against pooled Tier 1 means circuit-structure-to-fidelity
behaviour does not transfer between a superconducting and an ion-trap system,
which is a statement about the devices, not a model defect. The gap is the
number the paper reports.

USAGE
-----
    python experiments/lodo.py
    python experiments/lodo.py --limit 80000 --no-cv    # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import common as C

MIN_ROWS = 500


def run_direction(df, train_device, test_device, run_cv):
    """Train on one device, test on the other. No holdout split is needed:
    the entire held-out device *is* the test set."""
    train_df = df[df[C.DEVICE_COL] == train_device].reset_index(drop=True)
    test_df = df[df[C.DEVICE_COL] == test_device].reset_index(drop=True)

    tr_label = C.DEVICE_DISPLAY.get(train_device, train_device)
    te_label = C.DEVICE_DISPLAY.get(test_device, test_device)
    print("\n" + "-" * 74)
    print(f"  {tr_label} -> {te_label}")
    print("-" * 74)

    if len(train_df) < MIN_ROWS or len(test_df) < MIN_ROWS:
        print(f"  insufficient rows (train={len(train_df):,}, test={len(test_df):,}); "
              f"skipping")
        return None

    features = [f for f in C.tier1_feature_columns() if f in df.columns]
    print(f"  train={len(train_df):,}  test={len(test_df):,}  features={len(features)}")

    metrics = C.fit_and_score_lgbm(train_df, test_df, features, run_cv=run_cv)
    print("   ", metrics.summary())

    y_test = test_df[C.TARGET_COL].astype("float32").to_numpy()
    return {
        "train_device": train_device,
        "test_device": test_device,
        "train_display": tr_label,
        "test_display": te_label,
        "metrics": metrics.to_dict(),
        "test_target_mean": float(y_test.mean()),
        "test_target_std": float(y_test.std()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(parser)
    args = parser.parse_args()

    C.set_all_seeds()
    C.warn_if_alignment_stale(C.dataset_columns(args.dataset))
    if args.limit:
        print(f"  !! --limit {args.limit} set: smoke-test run, NOT paper numbers\n")

    cols = list(C.CIRCUIT_FEATURES) + [C.TARGET_COL, C.DEVICE_COL, C.GROUP_COL]
    df = C.load_dataset(cols, dataset=args.dataset, limit=args.limit)
    df = C.prepare_frame(df, C.CIRCUIT_FEATURES)
    df, mapping = C.add_device_encoding(df)

    print("=" * 74)
    print("TASK 6 — leave-one-device-out")
    print("=" * 74)
    print(f"  rows: {len(df):,}   device encoding: {mapping}")
    for device in sorted(df[C.DEVICE_COL].unique()):
        print(f"    {C.DEVICE_DISPLAY.get(device, device):8s} "
              f"{int((df[C.DEVICE_COL] == device).sum()):>10,} rows")

    run_cv = not args.no_cv
    results = [
        r
        for r in (
            run_direction(df, C.QEXA, C.MARMOT, run_cv),
            run_direction(df, C.MARMOT, C.QEXA, run_cv),
        )
        if r
    ]

    # Pooled Tier 1 on the same frame, as the comparison baseline.
    print("\n" + "-" * 74)
    print("  POOLED TIER 1 (baseline for comparison)")
    print("-" * 74)
    features = [f for f in C.tier1_feature_columns() if f in df.columns]
    train_idx, test_idx = C.group_holdout_split(df)
    pooled = C.fit_and_score_lgbm(
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
        features,
        run_cv=run_cv,
    )
    print("   ", pooled.summary())

    table = pd.DataFrame(
        [
            {
                "configuration": f"{r['train_display']} -> {r['test_display']}",
                "n_train": r["metrics"]["n_train"],
                "n_test": r["metrics"]["n_test"],
                "R2": round(r["metrics"]["r2_holdout"], 4),
                "MAE": round(r["metrics"]["mae_holdout"], 4),
                "RMSE": round(r["metrics"]["rmse_holdout"], 4),
                "delta_vs_pooled": round(
                    r["metrics"]["r2_holdout"] - pooled.r2_holdout, 4
                ),
            }
            for r in results
        ]
        + [
            {
                "configuration": "Pooled Tier 1 (80/20 holdout)",
                "n_train": pooled.n_train,
                "n_test": pooled.n_test,
                "R2": round(pooled.r2_holdout, 4),
                "MAE": round(pooled.mae_holdout, 4),
                "RMSE": round(pooled.rmse_holdout, 4),
                "delta_vs_pooled": 0.0,
            }
        ]
    )

    print("\n" + "=" * 74)
    print("  LODO SUMMARY")
    print("=" * 74)
    print(table.to_string(index=False))
    C.save_table("task6_lodo", table)

    C.save_results(
        "task6_lodo",
        {
            "device_encoding": mapping,
            "pooled_tier1": pooled.to_dict(),
            "directions": results,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
