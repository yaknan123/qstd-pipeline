#!/usr/bin/env python3
"""
train_tiers.py
==============

Task 3 — pipeline re-run baselines (PAPER_OBJECTIVES.md Section 3.1).
Numbers produced here fill Table V in Section VI.

TIERS
-----
  Tier 1  global, device-agnostic, 22 features (21 circuit + device ID)
          LightGBM, Random Forest, NN+DevEmb
  Tier 2b Q-Exa, per-device: circuit + sensor + calibration   LightGBM
  Tier 2a Marmot, per-device: circuit + sensor                LightGBM
          (Marmot has no calibration data)
  Tier 3  per-device ablation: 21 circuit features only,      LightGBM
          trained on the same per-device pool as Tier 2

Every model reports R^2 on the 20% holdout, R^2 from 5-fold GroupKFold CV on
the training set, MAE and RMSE. Tier 1 models additionally report three-class
accuracy at H=0.3/0.6 and macro-F1.

The Q-Exa sensor-only configuration (Tier 2a Q-Exa, Task 5) lives in
tier2a_sensor_only.py, which also computes the DeltaR^2 decomposition.

USAGE
-----
    python experiments/train_tiers.py                  # all tiers
    python experiments/train_tiers.py --tier 1         # one tier
    python experiments/train_tiers.py --limit 50000 --no-cv   # smoke test

This trains several large models and is expected to be launched by the user,
not run inline as part of an edit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import common as C

# Below this many rows a per-device fit is not meaningful; report and move on
# rather than emitting a NaN R^2 that could be mistaken for a result.
MIN_DEVICE_ROWS = 500


def _log_header(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def build_tier1_frame(dataset, limit):
    """Load and prepare the pooled Tier 1 table (circuit features + device ID)."""
    cols = list(C.CIRCUIT_FEATURES) + [C.TARGET_COL, C.DEVICE_COL, C.GROUP_COL]
    df = C.load_dataset(cols, dataset=dataset, limit=limit)
    df = C.prepare_frame(df, C.CIRCUIT_FEATURES)
    df, mapping = C.add_device_encoding(df)
    print(f"  rows: {len(df):,}   device encoding: {mapping}")
    return df, mapping


def run_tier1(dataset, limit, run_cv, skip_nn):
    _log_header("TIER 1 — global, device-agnostic (21 circuit + device ID)")
    df, mapping = build_tier1_frame(dataset, limit)
    features = C.tier1_feature_columns()
    present = [f for f in features if f in df.columns]
    if len(present) != len(features):
        print(f"  WARNING: only {len(present)}/{len(features)} Tier 1 features present")
    features = present

    train_idx, test_idx = C.group_holdout_split(df)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    print(f"  train={len(train_df):,}  holdout={len(test_df):,}  features={len(features)}")

    y_test = test_df[C.TARGET_COL].astype("float32").to_numpy()
    yc_test = C.hellinger_class(y_test)
    results: dict[str, dict] = {}

    # --- LightGBM -----------------------------------------------------------
    print("\n  [Tier 1] LightGBM")
    holder: dict = {}
    m = C.fit_and_score_lgbm(
        train_df, test_df, features, run_cv=run_cv, model_out=holder
    )
    pred = holder["model"].predict(test_df[features])
    cls = C.classification_scores(yc_test, C.hellinger_class(pred))
    m.extra["classification"] = cls
    print("   ", m.summary())
    print(f"    accuracy={cls['accuracy']:.4f}  macro_F1={cls['macro_f1']:.4f}")
    results["lightgbm"] = m.to_dict()

    # --- Random Forest ------------------------------------------------------
    print("\n  [Tier 1] Random Forest")
    rf = C.make_rf_regressor()
    rf.fit(train_df[features], train_df[C.TARGET_COL].astype("float32"))
    rf_pred = rf.predict(test_df[features])
    r2, mae, rmse = C.regression_scores(y_test, rf_pred)
    rf_cv = (
        C.cv_regression(
            C.make_rf_regressor(),
            train_df[features],
            train_df[C.TARGET_COL].astype("float32").to_numpy(),
            train_df[C.GROUP_COL].to_numpy(),
        )
        if run_cv
        else {}
    )
    rf_cls = C.classification_scores(yc_test, C.hellinger_class(rf_pred))
    rf_m = C.RegressionMetrics(
        n_train=len(train_df),
        n_test=len(test_df),
        n_features=len(features),
        r2_holdout=r2,
        mae_holdout=mae,
        rmse_holdout=rmse,
        extra={"classification": rf_cls},
        **rf_cv,
    )
    print("   ", rf_m.summary())
    print(f"    accuracy={rf_cls['accuracy']:.4f}  macro_F1={rf_cls['macro_f1']:.4f}")
    results["random_forest"] = rf_m.to_dict()

    # --- NN + device embedding ---------------------------------------------
    if not skip_nn:
        print("\n  [Tier 1] NN + device embedding")
        try:
            import nn_devemb

            nn_m, nn_pred = nn_devemb.fit_and_score(
                train_df, test_df, list(C.CIRCUIT_FEATURES)
            )
            nn_cls = C.classification_scores(yc_test, C.hellinger_class(nn_pred))
            nn_m.extra["classification"] = nn_cls
            print("   ", nn_m.summary())
            print(
                f"    accuracy={nn_cls['accuracy']:.4f}  "
                f"macro_F1={nn_cls['macro_f1']:.4f}"
            )
            results["nn_devemb"] = nn_m.to_dict()
        except ImportError as exc:
            print(f"    skipped (torch unavailable): {exc}")

    return {"device_encoding": mapping, "features": features, "models": results}


def _device_feature_sets(device, columns, *, include_calibration=True):
    """Resolve one device's Tier 2 candidate features, leakage-checked."""
    sensor_cols, calib_cols = C.resolve_device_columns(columns)
    own_sensors = sensor_cols.get(device, [])
    own_calib = calib_cols.get(device, []) if include_calibration else []
    features = list(C.CIRCUIT_FEATURES) + own_sensors + own_calib
    C.assert_no_cross_device_leakage(device, features, sensor_cols, calib_cols)
    return features, own_sensors, own_calib


def run_tier2_and_3(device, dataset, limit, run_cv, pool=None):
    """Train Tier 2 (circuit + sensor [+ calib]) and Tier 3 (circuit only).

    Both use the identical per-device training pool and holdout so the
    DeltaR^2 between them is a clean feature-category contribution.
    """
    label = C.DEVICE_DISPLAY.get(device, device)
    tier2_name = "Tier 2b" if device == C.QEXA else "Tier 2a"
    _log_header(f"{tier2_name} + TIER 3 — {label} ({device}), per-device")

    columns = C.dataset_columns(dataset)
    features, sensors, calib = _device_feature_sets(device, columns)
    print(f"  candidate features: {len(features)} "
          f"(circuit={len(C.CIRCUIT_FEATURES)}, sensor={len(sensors)}, "
          f"calib={len(calib)})")

    load_cols = features + [C.TARGET_COL, C.DEVICE_COL, C.GROUP_COL, C.TIME_COL]
    df = C.load_dataset(load_cols, dataset=dataset, limit=limit)
    df = df[df[C.DEVICE_COL] == device].reset_index(drop=True)
    df = C.apply_pool(df, device, pool)
    df = C.prepare_frame(df, features)
    if len(df) < MIN_DEVICE_ROWS:
        print(f"  only {len(df):,} rows for {device} (< {MIN_DEVICE_ROWS:,}); skipping. "
              f"With --limit this just means the slice did not reach this device.")
        return None

    train_idx, test_idx = C.group_holdout_split(df)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    kept, dropped = C.coverage_filter(train_df, features)
    print(f"  after >={C.MIN_FEATURE_COVERAGE:.0%} coverage filter: "
          f"{len(kept)} kept, {len(dropped)} dropped")
    kept_sensors = [f for f in kept if f in set(sensors)]
    kept_calib = [f for f in kept if f in set(calib)]
    print(f"    circuit={len(C.CIRCUIT_FEATURES)}, sensor={len(kept_sensors)}, "
          f"calib={len(kept_calib)}")
    print(f"  train={len(train_df):,}  holdout={len(test_df):,}")

    print(f"\n  [{tier2_name}] LightGBM — circuit + sensor"
          + (" + calibration" if kept_calib else ""))
    m2 = C.fit_and_score_lgbm(train_df, test_df, kept, run_cv=run_cv)
    print("   ", m2.summary())

    # Tier 3: same pool, circuit features only, NO device identity column.
    print("\n  [Tier 3] LightGBM — 21 circuit features only")
    tier3_features = C.tier3_feature_columns()
    assert C.DEVICE_ID_FEATURE not in tier3_features, (
        "Section 6: Tier 3 must not contain a device identity feature"
    )
    m3 = C.fit_and_score_lgbm(train_df, test_df, tier3_features, run_cv=run_cv)
    print("   ", m3.summary())

    d_holdout = C.delta_r2(m2.r2_holdout, m3.r2_holdout)
    print(f"\n  DeltaR2 ({tier2_name} - Tier 3), holdout = {d_holdout:+.4f}")

    return {
        "device": device,
        "display": label,
        "tier2_name": tier2_name,
        "n_sensor_features": len(kept_sensors),
        "n_calib_features": len(kept_calib),
        "tier2": m2.to_dict(),
        "tier3": m3.to_dict(),
        "delta_r2_holdout": d_holdout,
        "delta_r2_cv": C.delta_r2(m2.r2_cv_mean, m3.r2_cv_mean)
        if np.isfinite(m2.r2_cv_mean) and np.isfinite(m3.r2_cv_mean)
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(parser)
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which tier to run ('2' runs Tier 2 and its matched Tier 3)",
    )
    parser.add_argument("--skip-nn", action="store_true", help="Skip the NN baseline")
    args = parser.parse_args()

    C.set_all_seeds()
    C.warn_if_alignment_stale(C.dataset_columns(args.dataset))
    if args.limit:
        print(f"  !! --limit {args.limit} set: smoke-test run, NOT paper numbers\n")

    payload: dict = {"dataset": args.limit and f"{args.dataset} (limit={args.limit})"
                     or args.dataset, "pool": args.pool}
    run_cv = not args.no_cv

    if args.tier in ("1", "all"):
        payload["tier1"] = run_tier1(args.dataset, args.limit, run_cv, args.skip_nn)

    if args.tier in ("2", "3", "all"):
        per_device = []
        for device in C.PAPER_DEVICES:
            result = run_tier2_and_3(device, args.dataset, args.limit, run_cv, args.pool)
            if result:
                per_device.append(result)
        payload["per_device"] = per_device

        if per_device:
            table = pd.DataFrame(
                [
                    {
                        "device": r["display"],
                        "tier": r["tier2_name"],
                        "n_features": r["tier2"]["n_features"],
                        "R2_holdout": round(r["tier2"]["r2_holdout"], 4),
                        "R2_cv": round(r["tier2"]["r2_cv_mean"], 4),
                        "MAE": round(r["tier2"]["mae_holdout"], 4),
                        "RMSE": round(r["tier2"]["rmse_holdout"], 4),
                        "delta_R2_vs_tier3": round(r["delta_r2_holdout"], 4),
                    }
                    for r in per_device
                ]
                + [
                    {
                        "device": r["display"],
                        "tier": "Tier 3",
                        "n_features": r["tier3"]["n_features"],
                        "R2_holdout": round(r["tier3"]["r2_holdout"], 4),
                        "R2_cv": round(r["tier3"]["r2_cv_mean"], 4),
                        "MAE": round(r["tier3"]["mae_holdout"], 4),
                        "RMSE": round(r["tier3"]["rmse_holdout"], 4),
                        "delta_R2_vs_tier3": 0.0,
                    }
                    for r in per_device
                ]
            )
            _log_header("PER-DEVICE SUMMARY")
            print(table.to_string(index=False))
            C.save_table("task3_per_device", table)

    C.save_results("task3_train_tiers", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
