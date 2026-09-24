#!/usr/bin/env python3
"""
shot_histogram.py
=================

Task 4 — shot-count distribution and shot-noise robustness check
(PAPER_OBJECTIVES.md Section 3.2).

WHAT IT REPORTS
---------------
1. The distribution of the `shots` field across the full dataset, split by
   device: the top 3-5 buckets by count, with the number of circuits and the
   share (%) each accounts for.

2. The Hellinger shot-noise floor for the largest bucket, as the upper bound
       H_floor ~= 1 / sqrt(shots)
   This is the level of Hellinger distance attributable to finite sampling
   alone, so it bounds how much of the measured H any model could possibly
   explain away.

3. A robustness check: Tier 1 LightGBM retrained on circuits from the largest
   shot bucket only. Holding shots fixed removes shot count as a predictor, so
   if R^2 stays high the headline result is not an artefact of the model simply
   learning "few shots -> noisy -> high H".

   Section 3.2: if the restricted R^2 is >= 0.85, the paper's "91%" claim
   survives. The script states that verdict explicitly.

USAGE
-----
    python experiments/shot_histogram.py
    python experiments/shot_histogram.py --no-retrain   # distribution only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import common as C

R2_SURVIVAL_THRESHOLD = 0.85


def shot_distribution(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Per-device and overall shot-bucket counts and shares."""
    rows = []
    for device in sorted(df[C.DEVICE_COL].unique()):
        sub = df[df[C.DEVICE_COL] == device]
        counts = sub["shots"].value_counts().head(top_n)
        for shots, n in counts.items():
            rows.append(
                {
                    "scope": C.DEVICE_DISPLAY.get(device, device),
                    "shots": int(shots),
                    "n_circuits": int(n),
                    "share_pct": round(100.0 * n / len(sub), 2),
                    "h_floor": round(1.0 / np.sqrt(float(shots)), 4)
                    if shots > 0
                    else None,
                }
            )

    counts = df["shots"].value_counts().head(top_n)
    for shots, n in counts.items():
        rows.append(
            {
                "scope": "ALL",
                "shots": int(shots),
                "n_circuits": int(n),
                "share_pct": round(100.0 * n / len(df), 2),
                "h_floor": round(1.0 / np.sqrt(float(shots)), 4) if shots > 0 else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(parser)
    parser.add_argument("--top-n", type=int, default=5, help="Buckets to report")
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Report the distribution only, skip the restricted Tier 1 retrain",
    )
    args = parser.parse_args()

    C.set_all_seeds()
    C.warn_if_alignment_stale(C.dataset_columns(args.dataset))

    cols = list(C.CIRCUIT_FEATURES) + [C.TARGET_COL, C.DEVICE_COL, C.GROUP_COL]
    df = C.load_dataset(cols, dataset=args.dataset, limit=args.limit)
    df = C.prepare_frame(df, C.CIRCUIT_FEATURES)
    df = df.dropna(subset=["shots"]).reset_index(drop=True)
    df["shots"] = df["shots"].astype("int64")

    print("=" * 74)
    print(f"TASK 4 — shot-count distribution over {len(df):,} circuits")
    print("=" * 74)

    table = shot_distribution(df, args.top_n)
    print(table.to_string(index=False))
    C.save_table("task4_shot_distribution", table)

    largest = int(df["shots"].value_counts().idxmax())
    largest_n = int((df["shots"] == largest).sum())
    h_floor = 1.0 / np.sqrt(largest)
    print(
        f"\n  largest bucket: shots={largest:,} "
        f"({largest_n:,} circuits, {100.0 * largest_n / len(df):.2f}%)"
    )
    print(f"  Hellinger shot-noise floor (upper bound 1/sqrt(shots)) = {h_floor:.4f}")

    payload = {
        "n_circuits": len(df),
        "largest_bucket_shots": largest,
        "largest_bucket_n": largest_n,
        "largest_bucket_share_pct": round(100.0 * largest_n / len(df), 2),
        "h_floor_largest_bucket": round(float(h_floor), 4),
        "distribution": table.to_dict(orient="records"),
    }

    if not args.no_retrain:
        print("\n" + "-" * 74)
        print(f"  ROBUSTNESS CHECK — Tier 1 LightGBM restricted to shots={largest:,}")
        print("-" * 74)
        restricted = df[df["shots"] == largest].reset_index(drop=True)
        restricted, mapping = C.add_device_encoding(restricted)

        # `shots` is constant inside the bucket, so it carries no signal here.
        # Keeping it in preserves the exact Tier 1 feature list; LightGBM
        # simply never finds a split on it.
        features = C.tier1_feature_columns()
        features = [f for f in features if f in restricted.columns]

        if len(restricted) < 500:
            print(f"  only {len(restricted):,} rows in the bucket; skipping retrain")
        else:
            train_idx, test_idx = C.group_holdout_split(restricted)
            train_df = restricted.iloc[train_idx].reset_index(drop=True)
            test_df = restricted.iloc[test_idx].reset_index(drop=True)
            print(f"  train={len(train_df):,}  holdout={len(test_df):,}  "
                  f"devices={mapping}")

            metrics = C.fit_and_score_lgbm(
                train_df, test_df, features, run_cv=not args.no_cv
            )
            print("   ", metrics.summary())

            survives = metrics.r2_holdout >= R2_SURVIVAL_THRESHOLD
            verdict = (
                f"R^2 = {metrics.r2_holdout:.4f} >= {R2_SURVIVAL_THRESHOLD}: "
                "the paper's headline 91% claim SURVIVES the shot-noise "
                "robustness check."
                if survives
                else f"R^2 = {metrics.r2_holdout:.4f} < {R2_SURVIVAL_THRESHOLD}: "
                "the headline claim does NOT survive restriction to a single "
                "shot bucket and must be qualified in the paper."
            )
            print(f"\n  {verdict}")

            payload["restricted_tier1"] = metrics.to_dict()
            payload["restricted_r2_threshold"] = R2_SURVIVAL_THRESHOLD
            payload["headline_claim_survives"] = bool(survives)

    C.save_results("task4_shot_histogram", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
