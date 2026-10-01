#!/usr/bin/env python3
"""
tier2a_sensor_only.py
==========================

Task 5 — Q-Exa sensor-only configuration (PAPER_OBJECTIVES.md Section 3.3).
Reviewer 3 asked for this explicitly; it is the experiment that separates the
sensor contribution from the calibration contribution.

WHAT IT TRAINS
--------------
Three LightGBM models on the *identical* Q-Exa training pool and holdout:

  Tier 3   21 circuit features                      (no sensors, no calibration)
  Tier 2a  21 circuit + sensor features             (calibration withheld)
  Tier 2b  21 circuit + sensor + calibration        (everything)

WHAT IT REPORTS
---------------
  R^2, MAE, RMSE for each, plus the two category contributions:

    DeltaR^2 (Tier 2a - Tier 3)  = sensor contribution
    DeltaR^2 (Tier 2b - Tier 2a) = calibration contribution over and above
                                   sensors

Section 6 requires these contributions to come from DeltaR^2, not from a share
of SHAP importance. Nothing in this script aggregates SHAP by category.

COMPARABILITY
-------------
The three models share one GroupShuffleSplit (seed 42, grouped by parent_id)
and one coverage-filtered feature pool computed once on the training rows, so
the only thing that differs between them is which feature categories are
visible. Same hyperparameters as Tier 2b throughout.

USAGE
-----
    python experiments/tier2a_sensor_only.py
    python experiments/tier2a_sensor_only.py --limit 50000 --no-cv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import common as C


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(parser)
    parser.add_argument(
        "--device",
        default=C.QEXA,
        help="Device to decompose (default Q-Exa; Marmot has no calibration)",
    )
    parser.add_argument(
        "--shap-top",
        type=int,
        default=20,
        help="Also write the top-N per-feature SHAP ranking for Tier 2b",
    )
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
    if not calib:
        print(
            f"  {label} has no calibration features resolved. The Tier 2b - Tier 2a "
            f"decomposition needs them; check the calibration manifest."
        )
        return 1

    all_features = list(C.CIRCUIT_FEATURES) + sensors + calib
    C.assert_no_cross_device_leakage(device, all_features, sensor_cols, calib_cols)

    print("=" * 74)
    print(f"TASK 5 — {label} feature-category decomposition")
    print("=" * 74)
    print(f"  candidate features: circuit={len(C.CIRCUIT_FEATURES)}, "
          f"sensor={len(sensors)}, calib={len(calib)}")

    df = C.load_dataset(
        all_features + [C.TARGET_COL, C.DEVICE_COL, C.GROUP_COL, C.TIME_COL],
        dataset=args.dataset,
        limit=args.limit,
    )
    df = df[df[C.DEVICE_COL] == device].reset_index(drop=True)
    df = C.apply_pool(df, device, args.pool)
    df = C.prepare_frame(df, all_features)
    if len(df) < 500:
        print(f"  only {len(df):,} rows for {device}; nothing to decompose")
        return 1

    # One split shared by all three models.
    train_idx, test_idx = C.group_holdout_split(df)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # One coverage filter, computed once on the training rows, so Tier 2a and
    # Tier 2b see exactly the same sensor columns.
    kept, dropped = C.coverage_filter(train_df, all_features)
    kept_sensors = [f for f in kept if f in set(sensors)]
    kept_calib = [f for f in kept if f in set(calib)]

    f_tier3 = C.tier3_feature_columns()
    f_tier2a = f_tier3 + kept_sensors
    f_tier2b = f_tier3 + kept_sensors + kept_calib

    print(f"  after >={C.MIN_FEATURE_COVERAGE:.0%} coverage filter: "
          f"{len(kept)} kept, {len(dropped)} dropped")
    print(f"  Tier 3  : {len(f_tier3)} features")
    print(f"  Tier 2a : {len(f_tier2a)} features "
          f"({len(C.CIRCUIT_FEATURES)} circuit + {len(kept_sensors)} sensor)")
    print(f"  Tier 2b : {len(f_tier2b)} features "
          f"(+ {len(kept_calib)} calibration)")
    print(f"  train={len(train_df):,}  holdout={len(test_df):,}")

    run_cv = not args.no_cv
    holder: dict = {}

    print("\n  [Tier 3] circuit only")
    m3 = C.fit_and_score_lgbm(train_df, test_df, f_tier3, run_cv=run_cv)
    print("   ", m3.summary())

    print("\n  [Tier 2a] circuit + sensor (calibration withheld)")
    m2a = C.fit_and_score_lgbm(train_df, test_df, f_tier2a, run_cv=run_cv)
    print("   ", m2a.summary())

    print("\n  [Tier 2b] circuit + sensor + calibration")
    m2b = C.fit_and_score_lgbm(
        train_df, test_df, f_tier2b, run_cv=run_cv, model_out=holder
    )
    print("   ", m2b.summary())

    sensor_contrib = C.delta_r2(m2a.r2_holdout, m3.r2_holdout)
    calib_contrib = C.delta_r2(m2b.r2_holdout, m2a.r2_holdout)
    total_contrib = C.delta_r2(m2b.r2_holdout, m3.r2_holdout)

    print("\n" + "-" * 74)
    print("  FEATURE-CATEGORY CONTRIBUTIONS (holdout DeltaR^2)")
    print("-" * 74)
    print(f"    sensor      (Tier 2a - Tier 3)  = {sensor_contrib:+.4f}")
    print(f"    calibration (Tier 2b - Tier 2a) = {calib_contrib:+.4f}")
    print(f"    combined    (Tier 2b - Tier 3)  = {total_contrib:+.4f}")

    table = pd.DataFrame(
        [
            {
                "tier": name,
                "n_features": m.n_features,
                "R2_holdout": round(m.r2_holdout, 4),
                "R2_cv": round(m.r2_cv_mean, 4) if np.isfinite(m.r2_cv_mean) else None,
                "MAE": round(m.mae_holdout, 4),
                "RMSE": round(m.rmse_holdout, 4),
            }
            for name, m in (("Tier 3", m3), ("Tier 2a", m2a), ("Tier 2b", m2b))
        ]
    )
    print("\n" + table.to_string(index=False))
    C.save_table("task5_decomposition", table)

    # Per-feature SHAP ranking only — never aggregated into category shares.
    shap_ranking = None
    if args.shap_top > 0:
        try:
            explainer = C.shap_tree_explainer(holder["model"])
            sample = test_df[f_tier2b]
            if len(sample) > 10_000:
                sample = sample.sample(n=10_000, random_state=C.SEED)
            values = explainer.shap_values(sample)
            ranking = (
                pd.DataFrame(
                    {
                        "feature": f_tier2b,
                        "mean_abs_shap": np.abs(values).mean(axis=0),
                    }
                )
                .sort_values("mean_abs_shap", ascending=False)
                .head(args.shap_top)
                .reset_index(drop=True)
            )
            ranking["category"] = [
                "calibration" if f in set(kept_calib)
                else "sensor" if f in set(kept_sensors)
                else "circuit"
                for f in ranking["feature"]
            ]
            # The name the feature carries in the released dataset, for the paper.
            ranking["release_name"] = [C.release_name(f) for f in ranking["feature"]]
            print(f"\n  Top-{args.shap_top} features by mean |SHAP| "
                  f"(per-feature ranking only):")
            print(ranking.to_string(index=False))
            C.save_table("task5_shap_ranking", ranking)
            shap_ranking = ranking.to_dict(orient="records")
        except Exception as exc:  # shap is optional at run time
            print(f"  SHAP ranking skipped: {exc}")

    C.save_results(
        "task5_sensor_only",
        {
            "pool": args.pool,
            "device": device,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_sensor_features": len(kept_sensors),
            "n_calib_features": len(kept_calib),
            "tier3": m3.to_dict(),
            "tier2a": m2a.to_dict(),
            "tier2b": m2b.to_dict(),
            "delta_r2_sensor_contribution": sensor_contrib,
            "delta_r2_calibration_contribution": calib_contrib,
            "delta_r2_combined": total_contrib,
            "shap_top_features": shap_ranking,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
