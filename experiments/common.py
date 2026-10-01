#!/usr/bin/env python3
"""
common.py
=========

Shared configuration and helpers for every the platform-QFP paper experiment
(PAPER_OBJECTIVES.md Sections 3 and 4).

WHY THIS MODULE EXISTS
----------------------
Section 4 fixes a single reproducibility contract for all experiments (seed 42,
80/20 GroupShuffleSplit on parent_id, 5-fold GroupKFold, 10% coverage
threshold, one set of LightGBM hyperparameters). Section 6 forbids a specific
set of mistakes. Both are easy to violate accidentally when each experiment
re-implements its own data prep, so they are implemented exactly once here and
imported by every driver script.

The anti-requirements from Section 6 that this module enforces by construction:

  * Tier 2 is per-device. `tier2_feature_columns()` resolves features from one
    device's manifest entry only, and `assert_no_cross_device_leakage()` makes
    the guarantee checkable rather than merely intended.
  * Tier 3 is circuit-only. `CIRCUIT_FEATURES` has no device identity column,
    and `tier3_feature_columns()` returns exactly those 21 names. Callers that
    want device identity must ask for Tier 1 explicitly.
  * Seed 42 everywhere. `set_all_seeds()` covers python, numpy and torch;
    LGBM_PARAMS and RF_PARAMS carry `random_state=SEED`.
  * SHAP is for per-feature ranking only. `shap_tree_explainer()` hard-codes
    `feature_perturbation='tree_path_dependent'` and there is deliberately no
    helper here that aggregates SHAP into per-category shares — category
    contributions come from delta_r2() instead.

ALIGNMENT NOTE
--------------
Sensor features are computed over the lookback window ending at
`timestamp_completed`. The `count` and `fallback_used` columns emitted by the
extractor are provenance bookkeeping, not model features, and are excluded by
`SENSOR_FEATURE_SUFFIXES`.

A channel contributes EITHER four aggregates or one value, never both:

  * ordinary channels    -> `mean`, `min`, `max`, `sd`  (the Section 2.2 four)
  * policy channels      -> a single `value`

The second group is declared by `aggregation_policy` in
`sensor_calibration_query_pipeline/config/sensors_by_system.json`. It covers
channels the monitoring system already publishes as an aggregate (microphone_max,
magnetometer/stats/min/X, ...) and the 40 distributed temperature probes, all
reduced to the window mean. Re-aggregating an already-aggregated channel answers
no physical question — the min/max distinction is carried by WHICH channel it
is, not by how the window is reduced — and the probes are too numerous to
justify four features each.

`value` must stay in `SENSOR_FEATURE_SUFFIXES`: without it these columns are
extracted, stored, and then silently skipped by `resolve_device_columns()`,
which on the 2026-09-20 config would drop 54 of Marmot's 73 channels with no
error. Adding it cannot double-count, because the extractor emits only one form
per channel and the lookup is by exact column name.

All four Section 2.2 aggregates are in the contract. `sd` was briefly dropped
when it appeared unobtainable; the 2026-09-20 re-extraction produces it, so it
is back.

The original dataset must not be used for sensor work. Its sensor columns are
single carried-forward readings, not window aggregates: `count == 1` on every
row for 114 of 117 channels, with `min == max == mean`. `assert_windowed_sensors()`
detects that shape and raises, so it cannot be trained on by accident.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = Path(
    os.environ.get("MQSS_RESULTS_DIR", REPO_ROOT / "experiments" / "results")
)
BASE_RESULTS_DIR = RESULTS_DIR   # RESULTS_DIR moves to pool_<name>/ for non-default pools

# The dataset, under whichever name is present.
#
# Two names are in play and both are correct in their own context. Inside the
# build repository it is stage 10's output; a reader who downloaded the release
# has `qstd_v1.0.parquet`, because that is what the portal serves and what the
# README tells them to put in data/. Hard-coding either one strands the other,
# which is exactly what happened: the package shipped looking for the internal
# name while its own README asked for the published one.
#
# The published file is preferred when both exist, so a reader who also has a
# build tree gets the release they downloaded rather than a local artefact.
DATASET_CANDIDATES = (
    "qstd_v1.0.parquet",                    # the published release
    "qstd.parquet",                         # a renamed copy
    "stage10_cleaned_final_dataset.parquet",  # pipeline/10_clean_dataset.py output
)


def _resolve_dataset() -> Path:
    override = os.environ.get("QSTD_DATASET")
    if override:
        return Path(override).expanduser()
    for name in DATASET_CANDIDATES:
        if (DATA_DIR / name).exists():
            return DATA_DIR / name
    # Nothing present: name the published file, so the error a reader sees names
    # the file they were told to download.
    return DATA_DIR / DATASET_CANDIDATES[0]


DEFAULT_DATASET = _resolve_dataset()
SENSOR_MANIFEST = DATA_DIR / "sensors_by_system.json"

# The calibration manifest has shipped under two different names across
# pipeline revisions. Try both rather than silently training Tier 2b with zero
# calibration features, which would quietly collapse it into Tier 2a.
CALIB_MANIFEST_CANDIDATES = (
    DATA_DIR / "calibration_by_system.json",
    DATA_DIR / "calibration_data_by_system.json",
    REPO_ROOT
    / "sensor_calibration_query_pipeline"
    / "config"
    / "calibration_data_by_system.json",
    # The published code package keeps the same file under extraction/config/.
    REPO_ROOT / "extraction" / "config" / "calibration_data_by_system.json",
)


# ---------------------------------------------------------------------------
# Reproducibility contract (PAPER_OBJECTIVES.md Section 4)
# ---------------------------------------------------------------------------

SEED = 42
TEST_SIZE = 0.20
N_SPLITS_CV = 5
MIN_FEATURE_COVERAGE = 0.10

# ---------------------------------------------------------------------------
# Schema profile: internal build, or the published dataset
# ---------------------------------------------------------------------------
# The same analysis code runs on two schemas. Internally the columns carry the
# names the pipeline produced; the published dataset renames them to
# self-explanatory public ones and identifies a device by its technology rather
# than by the name it has inside the facility. Which schema is in front of us is
# decided by looking at the file, so neither copy of the code has to be edited
# and the published code is the code that produced the paper's numbers.
def _dataset_is_public(path: Path) -> bool:
    try:
        import pyarrow.parquet as pq
        return "device" in pq.ParquetFile(str(path)).schema_arrow.names
    except Exception:
        return False


PUBLIC_SCHEMA = _dataset_is_public(DEFAULT_DATASET)

TARGET_COL = "hellinger_native"
DEVICE_COL = "device" if PUBLIC_SCHEMA else "executed_resource"
GROUP_COL = "job_id" if PUBLIC_SCHEMA else "parent_id"

# Three-class fidelity thresholds. Section 6 forbids changing these without
# explicit discussion.
CLASS_THRESHOLD_GOOD = 0.3
CLASS_THRESHOLD_POOR = 0.6
CLASS_NAMES = ("good", "medium", "poor")

# Device identifiers, as they appear in DEVICE_COL. SC is the superconducting
# 20-qubit system, ION the trapped-ion one. The names are the published ones on
# the published dataset and the internal codes on an internal build.
if PUBLIC_SCHEMA:
    SC, ION = "superconducting_20q", "trapped_ion_20q"
    EXCLUDE_DEVICES = ("other_backend_1", "other_backend_2", "other_backend_3")
    DEVICE_DISPLAY = {SC: "superconducting", ION: "trapped ion"}
else:   # pragma: no cover - the published dataset is always the public schema
    raise SystemExit(
        "This dataset does not look like the published QSTD file: expected a "
        "'device' column. Point DEFAULT_DATASET at qstd_v1.0.parquet."
    )

# Historical aliases; the experiment scripts still use these names.
QEXA, MARMOT = SC, ION
PAPER_DEVICES = (SC, ION)

# ---------------------------------------------------------------------------
# Month pools for sensor/calibration tiers
# ---------------------------------------------------------------------------
# Sensor and calibration coverage depends on the month: Q-Exa sensors were not
# logged before 2025-03 and had an outage in 2025-09; Q-Exa calibration was
# barely published in 2025-04/05; Marmot sensors start in 2025-03. Training a
# sensor tier on every month makes most sensor cells empty (8.7% of Q-Exa
# records carry sensor data overall, so the 10% coverage filter removes every
# Q-Exa sensor feature) and lets missingness stand in for time. Tier 2 and its
# matched Tier 3 therefore train on a month pool. Circuit-only models (Tier 1,
# LODO, shot histogram) use every month.
#
#   A  Q-Exa 2025-04 onward without 2025-09     sensors ~85%, calibration ~70%
#   B  A without 2025-04 and 2025-05            sensors ~85%, calibration ~97%
#   Marmot uses 2025-03 onward in both pools.
TIME_COL = "completed_hour_utc" if PUBLIC_SCHEMA else "timestamp_completed_utc"
MONTH_POOLS: dict[str, dict[str, object]] = {
    "A": {QEXA: lambda m: (m >= "2025-04") & (m != "2025-09"),
          MARMOT: lambda m: m >= "2025-03"},
    "B": {QEXA: lambda m: (m >= "2025-06") & (m != "2025-09"),
          MARMOT: lambda m: m >= "2025-03"},
}
# Pool A chosen by the author on 2026-09-21 (34% more records than B; B's more
# complete calibration added nothing measurable).
DEFAULT_POOL: str | None = "A"


# Row pool AVAIL: "where the data exist". Instead of whole months it keeps each
# row that carries the device's own data, whenever it ran:
#   Q-Exa   >= 1 sensor reading AND >= 1 calibration value AND calibration no
#           older than CALIB_MAX_AGE_H
#   Marmot  >= 1 sensor reading (Marmot publishes no calibration)
# The age check exists because the calibration extractor queries one series per
# chunk of jobs and takes the latest sample at or before each job without
# enforcing the per-row lookback, so in sparse months a job can inherit a value
# weeks old (22,853 Q-Exa rows > 48 h; 10,668 of the 93,600 rows that have both
# sensors and calibration). The age comes from calibration_sample_ts_utc, which
# stage 10 drops, so it is read from stage 09 and joined on ROW_ID_COL. It is
# the newest sample across metrics, so it bounds staleness from below only.
# Results of non-default pools go to RESULTS_DIR/pool_<name>/ (see
# use_pool_results_dir) so they sit beside, not over, the pool A numbers.
AVAIL_POOL = "AVAIL"
POOL_CHOICES: list[str] = sorted(MONTH_POOLS) + [AVAIL_POOL]
CALIB_MAX_AGE_H = 48.0      # the calibration extractor's LOOKBACK_DAYS=2
ROW_ID_COL = "circuit_id" if PUBLIC_SCHEMA else "sub_id"
CALIB_AGE_SOURCE = DATA_DIR / "stage09_final_ml_dataset.parquet"
CALIB_SAMPLE_COL = "calibration_sample_ts_utc"


def calibration_age_hours(row_ids: pd.Series) -> pd.Series:
    """Hours between job completion and its calibration sample, from stage 09.

    Stage 09 is an internal file and is not part of the published dataset. When
    it is absent — the case when reproducing from QSTD — every calibration value
    present is known to be at most CALIB_MAX_AGE_H old, because the release
    builder blanks anything older, so the age is reported as 0 and the AVAIL
    filter reduces to "has calibration". On the internal build, where stale
    values could still exist, the real ages are read.
    """
    import pyarrow.parquet as pq

    if not CALIB_AGE_SOURCE.exists():
        print(f"  calibration age: {CALIB_AGE_SOURCE.name} absent; treating published "
              f"calibration as <= {CALIB_MAX_AGE_H:.0f} h old (the release guarantees it)")
        return pd.Series(0.0, index=range(len(row_ids)))

    src = pq.read_table(str(CALIB_AGE_SOURCE),
                        columns=[ROW_ID_COL, TIME_COL, CALIB_SAMPLE_COL]).to_pandas()
    age = (pd.to_datetime(src[TIME_COL], utc=True, errors="coerce")
           - pd.to_datetime(src[CALIB_SAMPLE_COL], utc=True, errors="coerce")
           ).dt.total_seconds() / 3600
    return pd.Series(age.to_numpy(), index=src[ROW_ID_COL]).reindex(row_ids.to_numpy())


# AVAIL rows are chosen once, on the reference (5-minute) dataset, and reused
# for every dataset, so window-ablation runs train and test on identical rows.
# Choosing per window would move rows in and out with the window length (a 2-min
# window catches fewer readings) and confound the window effect with the rows.
AVAIL_REFERENCE = DEFAULT_DATASET
_AVAIL_IDS: dict[str, set] = {}


def avail_row_ids(device: str) -> set:
    """Row ids of `device` that carry its own data in the reference dataset."""
    if device in _AVAIL_IDS:
        return _AVAIL_IDS[device]
    import pyarrow.parquet as pq

    sensor_cols, calib_cols = resolve_device_columns(dataset_columns(AVAIL_REFERENCE))
    sens = sensor_cols.get(device, [])
    cal = calib_cols.get(device, [])
    if not sens:
        raise ValueError(f"pool {AVAIL_POOL}: no {device} sensor columns in {AVAIL_REFERENCE}")
    ref = pq.read_table(str(AVAIL_REFERENCE),
                        columns=[ROW_ID_COL, DEVICE_COL, TIME_COL] + sens + cal).to_pandas()
    if not ref[ROW_ID_COL].is_unique:
        raise ValueError(
            f"{ROW_ID_COL} is not unique ({ref[ROW_ID_COL].nunique():,} distinct values "
            f"for {len(ref):,} rows). Pool {AVAIL_POOL} selects rows by this id, so a "
            f"per-job counter would pull in unrelated rows.")
    ref = ref[ref[DEVICE_COL].astype(str) == device].reset_index(drop=True)

    def any_value(cols):
        return ref[cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1).to_numpy()

    keep = any_value(sens)
    msg = f"sensor {keep.sum():,}"
    if cal:
        has_calib = any_value(cal)
        age = calibration_age_hours(ref[ROW_ID_COL]).to_numpy()
        fresh = has_calib & (age <= CALIB_MAX_AGE_H)
        msg += (f", calibration {has_calib.sum():,}, of which <= {CALIB_MAX_AGE_H:.0f} h "
                f"{fresh.sum():,}, sensor AND fresh calibration {(keep & fresh).sum():,}")
        keep &= fresh
    print(f"  pool {AVAIL_POOL} rows chosen on {AVAIL_REFERENCE.name} ({device}: {msg})")
    _AVAIL_IDS[device] = set(ref.loc[keep, ROW_ID_COL])
    return _AVAIL_IDS[device]


def apply_avail_pool(df: pd.DataFrame, device: str) -> pd.DataFrame:
    """Keep the rows of `device` that carry its own sensor (and calibration) data."""
    if ROW_ID_COL not in df.columns:
        raise ValueError(f"pool {AVAIL_POOL} needs {ROW_ID_COL}; load it with the data")
    out = df[df[ROW_ID_COL].isin(avail_row_ids(device))].reset_index(drop=True)
    months = pd.to_datetime(out[TIME_COL], utc=True).dt.strftime("%Y-%m")
    print(f"  pool {AVAIL_POOL}: {len(out):,} of {len(df):,} {device} rows; "
          f"{months.min()} to {months.max()}, {months.nunique()} months")
    return out


_RELEASE_NAMES = None


def release_name(col: str) -> str:
    """QSTD release name of a feature (release/column_names.py), for tables and plots.

    On the published dataset the columns already carry their release names, so
    there is nothing to look up and no mapping table is needed.

    Loaded by path: release/ has its own common.py, so it is never put on sys.path.
    """
    if PUBLIC_SCHEMA:
        return col
    global _RELEASE_NAMES
    if _RELEASE_NAMES is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "release_column_names", REPO_ROOT / "release" / "column_names.py")
        _RELEASE_NAMES = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_RELEASE_NAMES)
    return _RELEASE_NAMES.release_name(col)


def use_pool_results_dir(pool: str | None) -> Path:
    """Send results of a non-default pool to their own subdirectory."""
    global RESULTS_DIR
    RESULTS_DIR = BASE_RESULTS_DIR if pool == DEFAULT_POOL else BASE_RESULTS_DIR / f"pool_{pool}"
    return RESULTS_DIR


class PoolAction(argparse.Action):
    """--pool that also redirects the results directory when it is not the default."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        use_pool_results_dir(values)


def apply_pool(df: pd.DataFrame, device: str, pool: str | None) -> pd.DataFrame:
    """Keep one device's rows from the pool's months (by completion month, UTC),
    or, for pool AVAIL, the rows that carry the device's data."""
    if pool == AVAIL_POOL:
        return apply_avail_pool(df, device)
    if pool is None:
        print(f"\n  !! no month pool: training on every month. Sensor/calibration "
              f"coverage is low outside the pools, so sensor results are not "
              f"publication-ready. Pass --pool A or --pool B.\n")
        return df
    months = pd.to_datetime(df[TIME_COL], utc=True).dt.strftime("%Y-%m")
    keep = MONTH_POOLS[pool][device](months)
    out = df[keep.to_numpy()].reset_index(drop=True)
    print(f"  month pool {pool}: {len(out):,} of {len(df):,} {device} rows "
          f"({months[keep].min()} to {months[keep].max()})")
    return out


LGBM_PARAMS = dict(
    n_estimators=800,
    max_depth=10,
    num_leaves=64,
    learning_rate=0.05,
    reg_alpha=0.5,
    reg_lambda=2.0,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_samples=20,
    random_state=SEED,
    n_jobs=int(os.environ.get("MQSS_N_JOBS", "32")),
    verbose=-1,
    force_col_wise=True,
    max_bin=255,
)

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=int(os.environ.get("MQSS_N_JOBS", "32")),
)

NN_PARAMS = dict(
    hidden=128,
    emb_dim=4,
    lr=1e-3,
    batch_size=8192,
    epochs=40,
    eval_batch_size=16384,
)


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

# The 21 device-agnostic circuit features: 10 logical_* and 10 native_*
# features chosen by the user on 2026-09-21, plus shots. This is the Tier 3
# feature set verbatim, and the Tier 1 feature set before `device_encoded` is
# appended. Stage 10 keeps exactly these circuit columns; its CIRCUIT_FEATURES
# must equal this list.
CIRCUIT_FEATURES: tuple[str, ...] = (
    "logical_n_qubits",
    "logical_n_clbits",
    "logical_depth",
    "logical_size",
    "logical_n_1q_gates",
    "logical_n_2q_gates",
    "logical_n_measure",
    "logical_twoq_ratio",
    "logical_connectedness",
    "logical_liveliness",
    "native_n_qubits",
    "native_n_clbits",
    "native_depth",
    "native_size",
    "native_n_1q_gates",
    "native_n_2q_gates",
    "native_n_measure",
    "native_twoq_ratio",
    "native_connectedness",
    "native_liveliness",
    "shots",
)

DEVICE_ID_FEATURE = "device_encoded"

# Columns stage 10 keeps for the released dataset that are never model
# features. hellinger_logical has r=0.999 with TARGET_COL, so letting it into a
# feature set would leak the target; prepare_frame() refuses all of them.
METADATA_COLUMNS: tuple[str, ...] = (
    ("status", "submitted_hour_utc", "scheduled_hour_utc", "batch_size", "hellinger_logical")
    if PUBLIC_SCHEMA else
    ("id", "status", "timestamp_submitted_utc", "timestamp_scheduled_utc",
     "batch_size", "hellinger_logical")
)

# Section 2.2 aggregates that are model features. `count` and `fallback_used`
# are extractor provenance and must not be fed to the models.
SENSOR_FEATURE_SUFFIXES: tuple[str, ...] = ("mean", "min", "max", "sd", "value")
SENSOR_PROVENANCE_SUFFIXES: tuple[str, ...] = ("count", "fallback_used")


def tier1_feature_columns() -> list[str]:
    """Tier 1: 21 circuit features + device identity = 22 device-agnostic features."""
    return list(CIRCUIT_FEATURES) + [DEVICE_ID_FEATURE]


def tier3_feature_columns() -> list[str]:
    """Tier 3: circuit features only.

    Section 6 forbids device identity here. Returning a fresh list (not the
    tuple) keeps callers from mutating the canonical definition.
    """
    return list(CIRCUIT_FEATURES)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int = SEED) -> None:
    """Seed python, numpy and torch. Section 6: every experiment uses seed 42."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Manifests and device column resolution
# ---------------------------------------------------------------------------

def normalize_sensor_path(path: str) -> str:
    """Manifest sensor path -> dataframe column prefix.

    '<monitoring path>' -> '<internal column>'
    The ETL replaces '-' and '.' inside segments with '_' before joining
    segments with '__'.
    """
    return (
        path.lstrip("/")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "__")
        .lower()
    )


def normalize_calib_path(path: str) -> str:
    """Manifest calibration path -> exact dataframe column name.

    '<monitoring path>'          -> 'QB1_t1_time'
    '<monitoring path>'
                                                          -> 'TC_1_2_cz_gate_fidelity'
    Only the trailing two segments survive, joined by a single underscore.
    """
    segments = path.lstrip("/").split("/")
    if len(segments) >= 2:
        tag = segments[-2].replace("-", "_")
        return f"{tag}_{segments[-1]}"
    return path.lstrip("/").replace("-", "_").replace("/", "_")


def _resolve_calib_manifest() -> Path:
    for candidate in CALIB_MANIFEST_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No calibration manifest found. Looked for:\n  "
        + "\n  ".join(str(c) for c in CALIB_MANIFEST_CANDIDATES)
        + "\nTier 2b needs this file; without it the model would silently "
        "train with zero calibration features and collapse into Tier 2a."
    )


def load_manifests() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return (sensor_paths_by_device, calibration_paths_by_device).

    Raises rather than warning when a manifest is missing: a silently empty
    calibration set is the failure mode that produced a mislabelled Tier 2b.
    """
    if not SENSOR_MANIFEST.exists():
        raise FileNotFoundError(f"Sensor manifest not found: {SENSOR_MANIFEST}")

    with open(SENSOR_MANIFEST) as fh:
        raw_sensors = json.load(fh)
    # Two manifest schemas exist: a flat "sensors" list, and "sensor_categories"
    # grouping the same paths by category (the 2026-09 config). Reading only the
    # flat key against a categorised manifest resolves ZERO device columns, which
    # would silently turn every sensor tier into its circuit-only baseline.
    sensors = {}
    for dev, info in raw_sensors.items():
        if not isinstance(info, dict):
            continue
        paths = list(info.get("sensors") or [])
        for group in (info.get("sensor_categories") or {}).values():
            paths += [p for p in group if p not in paths]
        sensors[dev] = paths
    if not any(sensors.values()):
        raise ValueError(f"no sensor paths in {SENSOR_MANIFEST}; the manifest schema "
                         f"is not one this loader understands")

    with open(_resolve_calib_manifest()) as fh:
        raw_calib = json.load(fh)
    calib: dict[str, list[str]] = {}
    for dev, info in raw_calib.items():
        if not info.get("enabled", False):
            calib[dev] = []
            continue
        paths: list[str] = []
        for params in info.get("qubits", {}).values():
            paths.extend(params.values())
        for params in info.get("couplers", {}).values():
            paths.extend(params.values())
        calib[dev] = paths

    return sensors, calib


def resolve_device_columns(
    available_columns: Sequence[str],
    *,
    feature_suffixes: Sequence[str] = SENSOR_FEATURE_SUFFIXES,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Map manifest paths onto actual dataframe columns.

    Sensors match by prefix and keep only the aggregation suffixes in
    `feature_suffixes` (Section 2.2 features, not provenance). Calibration
    matches exactly, case-sensitively, with no suffix.

    On the published dataset there are no manifests to match against, and none
    are needed: a column's name says which device it belongs to (`sc_` for the
    superconducting system, `ion_` for the trapped-ion one) and whether it is
    calibration (`sc_cal_`). That naming is a guarantee of the release format,
    checked by the release validator, so reading it here is not a heuristic.
    """
    if PUBLIC_SCHEMA:
        columns = list(available_columns)
        calib_cols = {SC: sorted(c for c in columns if c.startswith("sc_cal_")),
                      ION: []}
        # Every published sensor column is a feature: the extractor's provenance
        # columns (count, fallback_used) are dropped at stage 10 and never
        # published, and a channel reduced to a single value carries no
        # statistic suffix at all, so filtering on suffixes would lose it.
        def _sensors(prefix):
            return sorted(c for c in columns
                          if c.startswith(prefix) and not c.startswith("sc_cal_"))
        sensor_cols = {SC: _sensors("sc_"), ION: _sensors("ion_")}
        return sensor_cols, calib_cols

    sensor_paths, calib_paths = load_manifests()
    columns = list(available_columns)
    lowered = {c.lower(): c for c in columns}

    sensor_cols: dict[str, list[str]] = {}
    for dev, paths in sensor_paths.items():
        found: set[str] = set()
        for path in paths:
            base = normalize_sensor_path(path)
            for suffix in feature_suffixes:
                col = lowered.get(f"{base}__{suffix}")
                if col is not None:
                    found.add(col)
        sensor_cols[dev] = sorted(found)

    calib_cols: dict[str, list[str]] = {}
    column_set = set(columns)
    for dev, paths in calib_paths.items():
        found = {
            name
            for name in (normalize_calib_path(p) for p in paths)
            if name in column_set
        }
        calib_cols[dev] = sorted(found)

    return sensor_cols, calib_cols


def assert_no_cross_device_leakage(
    device: str,
    features: Iterable[str],
    sensor_cols: dict[str, list[str]],
    calib_cols: dict[str, list[str]],
) -> None:
    """Fail loudly if a Tier 2 feature list contains another device's columns.

    Section 6 anti-requirement #1. Columns shared between devices (a sensor
    both systems publish) are allowed; only columns exclusive to some *other*
    device are leakage.
    """
    own: set[str] = set(sensor_cols.get(device, [])) | set(calib_cols.get(device, []))
    foreign: set[str] = set()
    for dev in set(sensor_cols) | set(calib_cols):
        if dev == device:
            continue
        foreign |= set(sensor_cols.get(dev, [])) | set(calib_cols.get(dev, []))

    exclusive_to_others = foreign - own
    offenders = sorted(set(features) & exclusive_to_others)
    if offenders:
        raise AssertionError(
            f"Cross-device leakage in Tier 2 features for {device}: "
            f"{len(offenders)} column(s) belong exclusively to another device, "
            f"e.g. {offenders[:5]}"
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def dataset_columns(dataset: Path | str = DEFAULT_DATASET) -> list[str]:
    """Read the parquet schema without loading any rows.

    PAPER_OBJECTIVES Section 9 forbids loading the full 1.4M-row frame just to
    inspect the schema.
    """
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(str(dataset)).schema_arrow.names)


def load_dataset(
    columns: Sequence[str],
    *,
    dataset: Path | str = DEFAULT_DATASET,
    limit: int | None = None,
) -> pd.DataFrame:
    """Column-selective read of the ML dataset.

    Only the requested columns are materialised, which keeps per-device runs to
    a few hundred MB instead of the full frame. `limit` reads just
    the first N rows and exists for smoke tests; it must not be used for any
    number that reaches the paper.

    Every load is checked by `assert_windowed_sensors()`, so the original
    carried-forward matrix cannot reach a model by accident.
    """
    import pyarrow.parquet as pq

    dataset = Path(dataset)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    available = set(dataset_columns(dataset))
    # The row id always rides along: pool AVAIL joins on it, and it is not a feature.
    wanted = [c for c in dict.fromkeys([*columns, ROW_ID_COL]) if c in available]
    missing = [c for c in dict.fromkeys(columns) if c not in available]
    if missing:
        print(f"  note: {len(missing)} requested column(s) absent from parquet, "
              f"e.g. {missing[:5]}")

    pf = pq.ParquetFile(str(dataset))
    if limit is None:
        df = pf.read(columns=wanted).to_pandas()
    else:
        batches = []
        seen = 0
        for batch in pf.iter_batches(batch_size=min(limit, 65536), columns=wanted):
            batches.append(batch)
            seen += batch.num_rows
            if seen >= limit:
                break
        import pyarrow as pa

        df = pa.Table.from_batches(batches).to_pandas().head(limit)

    if DEVICE_COL in df.columns:
        df[DEVICE_COL] = df[DEVICE_COL].astype(str)
        df = df[~df[DEVICE_COL].isin(EXCLUDE_DEVICES)].copy()

    df = df.reset_index(drop=True)
    assert_windowed_sensors(df)
    return df


def prepare_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Coerce features and target to numeric and drop rows without a target.

    Calibration columns arrive from the ETL as strings (empty '' when missing),
    so a plain astype would keep them unusable. `errors='coerce'` turns those
    into NaN, which LightGBM then handles natively.
    """
    leaked = [c for c in feature_cols if c in METADATA_COLUMNS]
    if leaked:
        raise ValueError(f"metadata columns are not model features: {leaked}")
    out = df.copy()
    for col in list(feature_cols) + [TARGET_COL]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[TARGET_COL, DEVICE_COL]).reset_index(drop=True)
    out["fidelity_class"] = hellinger_class(out[TARGET_COL].to_numpy())
    return out


def add_device_encoding(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add the Tier 1 `device_encoded` column with a stable, sorted mapping.

    Sorting by device name (rather than using LabelEncoder on whatever order
    the rows happen to arrive in) keeps the encoding identical across the
    pooled, LODO and shot-restricted runs, so their models stay comparable.
    """
    out = df.copy()
    mapping = {dev: i for i, dev in enumerate(sorted(out[DEVICE_COL].unique()))}
    out[DEVICE_ID_FEATURE] = out[DEVICE_COL].map(mapping).astype("int32")
    return out, mapping


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

# Five group splits. Every effect the paper reports is a small difference
# between two models, and a single split resolves such a difference only as well
# as the split itself is stable: which jobs land in the holdout moves a DeltaR^2
# by about as much as the DeltaR^2 is. Results are therefore reported as
# mean +/- sd over these five splits (experiments/split_repeats.py). The first
# seed is SEED, so a single-split run stays comparable with earlier work.
SPLIT_SEEDS: tuple[int, ...] = (SEED, 7, 123, 2024, 31337)


def group_holdout_split(df: pd.DataFrame, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """80/20 GroupShuffleSplit grouped by parent_id.

    Grouping by parent_id keeps every sub-job of one submitted circuit on the
    same side of the split, so a near-duplicate of a training circuit cannot
    appear in the holdout.

    `seed` selects the split. It defaults to SEED (42); pass one of SPLIT_SEEDS
    to repeat an evaluation over the five splits the paper reports.
    """
    from sklearn.model_selection import GroupShuffleSplit

    if GROUP_COL not in df.columns:
        raise KeyError(
            f"{GROUP_COL!r} missing; Section 4 requires grouping by parent_id"
        )
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[GROUP_COL].to_numpy()))
    return train_idx, test_idx


def group_kfold(n_groups: int):
    """5-fold GroupKFold, clamped when a device has fewer than 5 parent groups."""
    from sklearn.model_selection import GroupKFold

    return GroupKFold(n_splits=max(2, min(N_SPLITS_CV, n_groups)))


# ---------------------------------------------------------------------------
# Coverage filter (Section 2.3)
# ---------------------------------------------------------------------------

def coverage_filter(
    train_df: pd.DataFrame,
    candidate_features: Sequence[str],
    *,
    threshold: float = MIN_FEATURE_COVERAGE,
    always_keep: Sequence[str] = CIRCUIT_FEATURES,
) -> tuple[list[str], list[str]]:
    """Keep features whose non-null coverage on the TRAINING rows is >= threshold.

    Measured on training rows only so the holdout never influences which
    columns exist. Circuit features are exempt: they define the tier and must
    not silently disappear if one is sparse on a given device.
    """
    keep_always = set(always_keep)
    present = [f for f in candidate_features if f in train_df.columns]

    # A zero-filled absence is indistinguishable from a reading to notna(), so
    # this gate silently passes everything on a zero-filled dataset. Warn rather
    # than quietly report a coverage number that means nothing.
    warn_if_zero_filled(train_df, present)

    coverage = train_df[present].notna().mean()

    kept = [f for f in present if f in keep_always or coverage[f] >= threshold]
    dropped = [f for f in present if f not in kept]
    return kept, dropped


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def hellinger_class(h) -> np.ndarray:
    """Three-class fidelity label: 0 good (<0.3), 1 medium (0.3-0.6), 2 poor (>=0.6)."""
    arr = np.asarray(h, dtype=float)
    return np.where(
        arr < CLASS_THRESHOLD_GOOD, 0, np.where(arr < CLASS_THRESHOLD_POOR, 1, 2)
    ).astype(int)


@dataclass
class RegressionMetrics:
    """The metric block Section 3.1 requires for every trained model."""

    n_train: int
    n_test: int
    n_features: int
    r2_holdout: float
    mae_holdout: float
    rmse_holdout: float
    r2_cv_mean: float = float("nan")
    r2_cv_std: float = float("nan")
    mae_cv_mean: float = float("nan")
    rmse_cv_mean: float = float("nan")
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        cv = (
            f"{self.r2_cv_mean:.4f} ± {self.r2_cv_std:.4f}"
            if np.isfinite(self.r2_cv_mean)
            else "n/a"
        )
        return (
            f"R2(holdout)={self.r2_holdout:.4f}  R2(CV)={cv}  "
            f"MAE={self.mae_holdout:.4f}  RMSE={self.rmse_holdout:.4f}  "
            f"[{self.n_features} feats, train={self.n_train:,}, test={self.n_test:,}]"
        )


def regression_scores(y_true, y_pred) -> tuple[float, float, float]:
    """Return (R2, MAE, RMSE)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return (
        float(r2_score(y_true, y_pred)),
        float(mean_absolute_error(y_true, y_pred)),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def classification_scores(y_true, y_pred) -> dict:
    """Three-class accuracy and macro-F1 at H=0.3 / H=0.6 (Section 3.1)."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class_f1": [
            float(v) for v in f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
        ],
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1, 2]
        ).tolist(),
    }


def delta_r2(higher: float, lower: float) -> float:
    """Feature-category contribution as a difference of R2 values.

    Section 6 requires category contributions to be reported this way rather
    than as a share of SHAP importance. There is intentionally no
    SHAP-share-by-category helper in this module.
    """
    return float(higher - lower)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_lgbm_regressor(**overrides):
    """LightGBM regressor with the Section 4 hyperparameters.

    No imputer is wrapped around it: Section 4 specifies native missing-value
    handling, and LightGBM learns a default direction per split from the NaNs
    themselves. Imputing first would erase exactly the sparsity pattern the
    10%-coverage sensor columns carry.
    """
    import lightgbm as lgb

    params = {**LGBM_PARAMS, **overrides}
    return lgb.LGBMRegressor(**params)


def make_rf_regressor(**overrides):
    """Random Forest with median imputation (it has no native NaN handling)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    params = {**RF_PARAMS, **overrides}
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(**params)),
        ]
    )


def cv_regression(estimator, X, y, groups) -> dict:
    """5-fold GroupKFold CV. Callers must pass the TRAINING set only.

    Section 3.1 asks for "R^2 from 5-fold GroupKFold CV on the training set";
    running it over train+test pooled would report a number partly fitted on
    the holdout.
    """
    from sklearn.model_selection import cross_validate

    cv = group_kfold(pd.Series(groups).nunique())
    scores = cross_validate(
        estimator,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
        n_jobs=1,
    )
    return {
        "r2_cv_mean": float(scores["test_r2"].mean()),
        "r2_cv_std": float(scores["test_r2"].std()),
        "mae_cv_mean": float(-scores["test_neg_mean_absolute_error"].mean()),
        "rmse_cv_mean": float(-scores["test_neg_root_mean_squared_error"].mean()),
    }


def fit_and_score_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Sequence[str],
    *,
    run_cv: bool = True,
    model_out: dict | None = None,
) -> RegressionMetrics:
    """Fit LightGBM on train, score on holdout, optionally CV on train."""
    features = list(features)
    X_train = train_df[features]
    y_train = train_df[TARGET_COL].astype("float32").to_numpy()
    X_test = test_df[features]
    y_test = test_df[TARGET_COL].astype("float32").to_numpy()

    model = make_lgbm_regressor()
    model.fit(X_train, y_train)
    r2, mae, rmse = regression_scores(y_test, model.predict(X_test))

    cv_block: dict = {}
    if run_cv and GROUP_COL in train_df.columns:
        cv_block = cv_regression(
            make_lgbm_regressor(), X_train, y_train, train_df[GROUP_COL].to_numpy()
        )

    if model_out is not None:
        model_out["model"] = model
        model_out["features"] = features

    return RegressionMetrics(
        n_train=len(train_df),
        n_test=len(test_df),
        n_features=len(features),
        r2_holdout=r2,
        mae_holdout=mae,
        rmse_holdout=rmse,
        **cv_block,
    )


def shap_tree_explainer(model):
    """TreeExplainer pinned to tree_path_dependent (Section 6 anti-requirement #2).

    Used for per-feature ranking only. Category-level contributions come from
    delta_r2(), never from a share of total |SHAP|.
    """
    import shap

    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")


# ---------------------------------------------------------------------------
# Result output
# ---------------------------------------------------------------------------

def results_path(name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name


def save_results(name: str, payload: dict) -> Path:
    """Write one experiment's numbers as JSON next to the other results."""
    path = results_path(f"{name}.json")
    payload = {"seed": SEED, **payload}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"  wrote {path}")
    return path


def save_table(name: str, df: pd.DataFrame) -> Path:
    path = results_path(f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  wrote {path}")
    return path


def warn_if_alignment_stale(columns: Sequence[str]) -> None:
    """Warn when the dataset predates the Section 2.1 alignment change.

    The observable fingerprint is the completion-time anchor column. The `__sd`
    aggregate is deliberately not checked: it was dropped from the feature
    contract (see SENSOR_FEATURE_SUFFIXES), so its absence is expected rather
    than stale.
    """
    cols = list(columns)
    problems = []
    # Two names for the same anchor: the build repository calls it
    # timestamp_completed_utc, the published release completed_hour_utc. Checking
    # only the internal spelling told every reader of the published dataset that
    # their data was "not publication-ready", which was both wrong and alarming.
    if not any(c.startswith("timestamp_completed") or c.startswith("completed_hour")
               for c in cols):
        problems.append(
            "no completion-time anchor (timestamp_completed* or completed_hour*) "
            "-> built with the removed timestamp_scheduled anchor (Section 2.1)"
        )
    if problems:
        print("\n  !! DATASET IS STALE RELATIVE TO PAPER_OBJECTIVES:")
        for p in problems:
            print(f"     - {p}")
        print("     Numbers produced from it are not publication-ready until the "
              "pipeline is re-run.\n")


def assert_windowed_sensors(df, *, sample_rows: int = 50_000) -> None:
    """Refuse a dataset whose sensor columns are carried-forward point values.

    The original the platform-QFP matrix stored a single nearest reading per channel
    rather than an aggregate over the alignment window. The fingerprints are
    unmistakable and cheap to test:

      * `__count` is constant 1 wherever the channel is populated — a real
        window yields a spread of densities (0, 4, 5, 6, 10, ...).
      * `min == max` on every populated row, so `sd` is identically 0 and
        min/max/sd carry no information at all.

    Training on that shape silently reduces four aggregates to one point value
    and, because a carried-forward reading only changes when the sensor is
    re-read, lets it proxy for job identity. Raise rather than let a number be
    published from it.
    """
    cols = list(df.columns)
    counts = [c for c in cols if c.endswith("__count")]
    head = df.head(sample_rows)
    populated, suspects = [], []
    for c in counts:
        v = pd.to_numeric(head[c], errors="coerce")
        nz = v[v > 0]
        if len(nz) < 100:
            # Not populated in this slice — typically the other device's
            # channels. Judging it either way would be noise.
            continue
        populated.append(c)
        if nz.nunique() == 1 and nz.iloc[0] == 1:
            suspects.append(c)

    # Second fingerprint, for column-selective reads that never requested
    # `__count`: a carried-forward point value has min == max on every
    # populated row, so min/max/sd are degenerate.
    bases = {c[: -len("__max")] for c in cols if c.endswith("__max")}
    deg_pop, deg_flat = [], []
    for b in sorted(bases):
        lo, hi = f"{b}__min", f"{b}__max"
        if lo not in head.columns or hi not in head.columns:
            continue
        a = pd.to_numeric(head[lo], errors="coerce")
        z = pd.to_numeric(head[hi], errors="coerce")
        # A row counts as a real reading only if it is present and not the
        # all-zero fill that older extractor versions wrote for "no samples".
        # Including zero-fill would make every sparse channel look degenerate.
        both = a.notna() & z.notna() & ((a != 0) | (z != 0))
        if both.sum() < 100:
            continue
        deg_pop.append(b)
        if (a[both] == z[both]).all():
            deg_flat.append(b)

    if len(deg_pop) >= 5 and len(deg_flat) >= 0.9 * len(deg_pop):
        raise ValueError(
            f"sensor columns look carried-forward, not windowed: "
            f"{len(deg_flat)}/{len(deg_pop)} populated channels have "
            f"min == max on every row (e.g. {deg_flat[:3]}), so min/max/sd "
            f"carry no information. This is the original pre-rebuild matrix; "
            f"sensor results from it are not valid."
        )

    if len(populated) >= 5 and len(suspects) >= 0.9 * len(populated):
        raise ValueError(
            f"sensor columns look carried-forward, not windowed: "
            f"{len(suspects)}/{len(populated)} populated channels have "
            f"count == 1 on every row (e.g. {suspects[:3]}). This is the "
            f"original pre-rebuild matrix; sensor results from it are not "
            f"valid. Use a dataset built by the current extractor."
        )


def warn_if_zero_filled(df, feature_columns: Sequence[str]) -> None:
    """Warn when absent sensor readings were written as 0 instead of null.

    Extractor versions before the null-fill fix wrote 0 for "no samples in the
    window". 0 is a physically valid reading, so those cells are
    indistinguishable from measurements and every null-based coverage check —
    including `coverage_filter()` — reads them as covered.
    """
    present = [f for f in feature_columns if f in df.columns]
    if not present:
        return
    head = df.head(50_000)
    zero_heavy = 0
    for f in present:
        v = pd.to_numeric(head[f], errors="coerce")
        if v.notna().mean() > 0.99 and (v == 0).mean() > 0.5:
            zero_heavy += 1
    if zero_heavy >= max(3, len(present) // 10):
        print(
            f"\n  !! {zero_heavy}/{len(present)} sensor features are >50% exact "
            f"zero with almost no nulls.\n     Absent readings were probably "
            f"zero-filled rather than left null, which makes\n     "
            f"coverage_filter() overstate coverage. Re-extract with the current "
            f"script.\n"
        )


def add_common_args(parser):
    """CLI flags shared by every experiment driver."""
    parser.add_argument(
        "--dataset", default=str(DEFAULT_DATASET), help="Path to the ML parquet"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Read only the first N rows (smoke tests only, not for paper numbers)",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip GroupKFold CV (holdout metrics only); useful for quick checks",
    )
    parser.add_argument(
        "--pool",
        choices=POOL_CHOICES,
        default=DEFAULT_POOL,
        action=PoolAction,
        help="Month pool for sensor/calibration tiers (see MONTH_POOLS), or AVAIL "
             "for the rows where the data exist; non-default pools write to "
             "results/pool_<name>/",
    )
    return parser
