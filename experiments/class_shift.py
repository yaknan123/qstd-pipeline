#!/usr/bin/env python3
"""
class_shift.py
==============

Task 8 — class-shift analysis (PAPER_OBJECTIVES.md Section 3.6).
Numbers feed the "Contribution of Time-Aligned System Data" paragraph in
Section VI.

WHAT IT DOES
------------
For every the superconducting device holdout circuit, predicts fidelity twice:

  Tier 3   circuit features only          (21 features)
  Tier 2b  circuit + sensor + calibration (the full per-device model)

then discretises both predictions at the fixed class thresholds

  good   H < 0.3
  medium 0.3 <= H < 0.6
  poor   H >= 0.6

and reports where circuits move.

WHAT IT REPORTS
---------------
  * 3x3 transition matrix (Tier 3 class -> Tier 2b class), raw counts and
    row-normalised
  * percentage of circuits that change class at all
  * percentage moving from poor to medium/good  (pessimistic -> better)
  * percentage moving from good to medium/poor  (optimistic -> worse)

Both models share one training pool and one holdout, so a class change is
attributable to the added feature categories rather than to a different split.

A note on direction: a move is not automatically an improvement. The script
also reports accuracy against the true class for both tiers, so the shift can
be read as "Tier 2b corrects Tier 3" rather than merely "Tier 2b disagrees".

USAGE
-----
    python experiments/class_shift.py
    python experiments/class_shift.py --device trapped_ion_20q    # the trapped-ion device (sensors only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import common as C


def transition_matrix(from_cls: np.ndarray, to_cls: np.ndarray) -> pd.DataFrame:
    """3x3 raw-count transition matrix, rows = source class."""
    matrix = np.zeros((3, 3), dtype=int)
    for src, dst in zip(from_cls, to_cls):
        matrix[src, dst] += 1
    return pd.DataFrame(
        matrix,
        index=[f"tier3_{n}" for n in C.CLASS_NAMES],
        columns=[f"tier2b_{n}" for n in C.CLASS_NAMES],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(parser)
    parser.add_argument("--device", default=C.QEXA, help="Device to analyse")
    args = parser.parse_args()

    C.set_all_seeds()
    device = args.device
    label = C.DEVICE_DISPLAY.get(device, device)
    columns = C.dataset_columns(args.dataset)
    C.warn_if_alignment_stale(columns)
    if args.limit:
        print(f"  !! --limit {args.limit} set: smoke-test run, NOT paper numbers\n")

    sensor_cols, calib_cols = C.resolve_device_columns(columns)
    sensors = sensor_cols.get(device, [])
    calib = calib_cols.get(device, [])
    all_features = list(C.CIRCUIT_FEATURES) + sensors + calib
    C.assert_no_cross_device_leakage(device, all_features, sensor_cols, calib_cols)

    print("=" * 74)
    print(f"TASK 8 — class-shift analysis, {label} holdout")
    print("=" * 74)

    df = C.load_dataset(
        all_features + [C.TARGET_COL, C.DEVICE_COL, C.GROUP_COL, C.TIME_COL],
        dataset=args.dataset,
        limit=args.limit,
    )
    df = df[df[C.DEVICE_COL] == device].reset_index(drop=True)
    df = C.apply_pool(df, device, args.pool)
    df = C.prepare_frame(df, all_features)
    if len(df) < 500:
        print(f"  only {len(df):,} rows for {device}; nothing to analyse")
        return 1

    train_idx, test_idx = C.group_holdout_split(df)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    kept, _ = C.coverage_filter(train_df, all_features)
    f_tier3 = C.tier3_feature_columns()
    f_tier2b = f_tier3 + [f for f in kept if f not in set(C.CIRCUIT_FEATURES)]
    print(f"  train={len(train_df):,}  holdout={len(test_df):,}")
    print(f"  Tier 3: {len(f_tier3)} features   Tier 2b: {len(f_tier2b)} features")

    holder3: dict = {}
    holder2b: dict = {}
    print("\n  fitting Tier 3 ...")
    m3 = C.fit_and_score_lgbm(
        train_df, test_df, f_tier3, run_cv=False, model_out=holder3
    )
    print("   ", m3.summary())
    print("  fitting Tier 2b ...")
    m2b = C.fit_and_score_lgbm(
        train_df, test_df, f_tier2b, run_cv=False, model_out=holder2b
    )
    print("   ", m2b.summary())

    pred3 = holder3["model"].predict(test_df[f_tier3])
    pred2b = holder2b["model"].predict(test_df[f_tier2b])
    cls3 = C.hellinger_class(pred3)
    cls2b = C.hellinger_class(pred2b)
    cls_true = C.hellinger_class(test_df[C.TARGET_COL].to_numpy())

    counts = transition_matrix(cls3, cls2b)
    row_totals = counts.sum(axis=1).replace(0, np.nan)
    normalised = (counts.div(row_totals, axis=0) * 100).round(2)

    n = len(cls3)
    changed = int((cls3 != cls2b).sum())
    # class index: 0 good, 1 medium, 2 poor
    poor_to_better = int(((cls3 == 2) & (cls2b < 2)).sum())
    good_to_worse = int(((cls3 == 0) & (cls2b > 0)).sum())

    acc3 = float((cls3 == cls_true).mean())
    acc2b = float((cls2b == cls_true).mean())

    print("\n" + "-" * 74)
    print("  TRANSITION MATRIX — raw counts (rows: Tier 3, cols: Tier 2b)")
    print("-" * 74)
    print(counts.to_string())
    print("\n  row-normalised (%)")
    print(normalised.to_string())

    print("\n" + "-" * 74)
    print("  CLASS-SHIFT SUMMARY")
    print("-" * 74)
    print(f"  holdout circuits              : {n:,}")
    print(f"  changed class                 : {changed:,} ({100.0 * changed / n:.2f}%)")
    print(f"  poor -> medium/good           : {poor_to_better:,} "
          f"({100.0 * poor_to_better / n:.2f}%)")
    print(f"  good -> medium/poor           : {good_to_worse:,} "
          f"({100.0 * good_to_worse / n:.2f}%)")
    print(f"\n  accuracy vs true class, Tier 3 : {acc3:.4f}")
    print(f"  accuracy vs true class, Tier 2b: {acc2b:.4f}")
    print(f"  accuracy change                : {acc2b - acc3:+.4f}")

    counts_out = counts.reset_index().rename(columns={"index": "from_class"})
    C.save_table("task8_class_transition_counts", counts_out)
    C.save_table(
        "task8_class_transition_rownorm",
        normalised.reset_index().rename(columns={"index": "from_class"}),
    )

    C.save_results(
        "task8_class_shift",
        {
            "pool": args.pool,
            "device": device,
            "n_holdout": n,
            "class_thresholds": {
                "good_below": C.CLASS_THRESHOLD_GOOD,
                "poor_at_or_above": C.CLASS_THRESHOLD_POOR,
            },
            "transition_counts": counts.values.tolist(),
            "transition_row_normalised_pct": normalised.values.tolist(),
            "class_order": list(C.CLASS_NAMES),
            "pct_changed_class": round(100.0 * changed / n, 2),
            "pct_poor_to_better": round(100.0 * poor_to_better / n, 2),
            "pct_good_to_worse": round(100.0 * good_to_worse / n, 2),
            "accuracy_tier3": round(acc3, 4),
            "accuracy_tier2b": round(acc2b, 4),
            "tier3": m3.to_dict(),
            "tier2b": m2b.to_dict(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
