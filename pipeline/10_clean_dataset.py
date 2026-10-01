#!/usr/bin/env python3
"""
10_clean_dataset.py
===================

PURPOSE
-------
Turn the stage 09 ML dataset into the cleaned, typed dataset the experiments
train on (`stage10_cleaned_final_dataset.parquet`).

Stage 09 keeps everything a later stage might want to look up: DAG storage
references, extractor bookkeeping, duplicate circuit metrics, and every column
as a string. Stage 10 removes what must never reach a model and writes the
remaining numeric columns as float64. The column decisions come from the
2026-09-21 audit of the 20260920T231153Z rebuild (803 columns, 1,390,523 rows),
rechecked on the ext_20260510 build (804 columns, 1,558,442 rows);
the per-column reasoning is in analysis/column_decisions_20260920T231153Z.csv.

WHAT THIS SCRIPT DOES
---------------------
1. ROW FILTER: drops every job (parent_id) whose status is CANCELLED. On the
   2026-09-20 rebuild that is 77 whole jobs / 3,321 circuits (3,310 on Marmot);
   no job mixes CANCELLED and COMPLETED circuits.
2. RULE-BASED COLUMN FILTER: every input column must match exactly one rule in
   COLUMN_RULES below, or the script stops. A schema change upstream therefore
   has to be classified here deliberately instead of flowing into the models.
3. DATA-DRIVEN COLUMN FILTER: among the columns the rules keep as features, any
   that is empty, holds a single distinct value, or has a reading on fewer than
   MIN_COVERAGE of the kept rows is dropped. Sensor channels are judged as a
   whole (all aggregates or none).
   The coverage rule was added on 2026-09-23 for the extension to 2026-05-10,
   which brought in the the superconducting device's room room channels: they published for one 2.5 h span on
   2025-12-17 and cover 606 of 1,549,234 released rows (0.04%). A channel that exists for
   one morning is a timestamp marker, not a measurement. The next-sparsest
   channel (the lab occupancy) covers 2.9%, so the 1% threshold is well clear.
   On the 2026-09-20 rebuild this removes the dead the superconducting device's room channels and the ten
   Q-Exa heater/pump/pulsetube channels, which are on/off states that never
   change. Identity columns and the target are exempt.
4. TYPING: feature columns and the target become float64 (Hellinger distances
   off by rounding, e.g. 1 + 2e-16, are clipped to [0, 1]), the completion time
   becomes a UTC timestamp, and identifiers stay strings. A feature value that
   is present but not numeric aborts the run rather than silently becoming NaN.
5. REPORT: writes <output>.columns.json with the fate and reason of every input
   column, the row counts, and the data-driven drops.

RULES THAT ENCODE A DECISION (2026-09-21)
-----------------------------------------
- Cancelled jobs are removed.
- The batch columns (batch_index, batch_size, is_batch_job) are left out.
- calibration_sample_ts_utc is dropped.
- Only the circuit features in CIRCUIT_FEATURES below are kept: the 11
  logical_* and 10 native_* features chosen by the user on 2026-09-21, plus
  shots. Every other circuit column (delta_*, ratio_*, *_swap_count,
  *_component_count, *_measure_ratio, *_n_gate_types, ...) is dropped. The kept
  ones are exempt from the empty/constant filter (stage 10 warns instead).
  This list must equal CIRCUIT_FEATURES in experiments/common.py.
  The *_dag_* columns are DAG storage metadata and are dropped.

WHAT IS NOT A FEATURE BUT IS KEPT
---------------------------------
Identifiers: parent_id (group key for the split), sub_id (unique circuit id),
executed_resource (device), timestamp_completed_utc (month pools, time splits).

Metadata, kept in the released dataset by decision 2026-09-21 but never a
model feature: id, status, timestamp_submitted_utc, timestamp_scheduled_utc,
batch_size, hellinger_logical.

TIMESTAMPS
----------
The scheduler records timestamp_submitted and timestamp_scheduled as naive
Europe/Berlin wall-clock time. They are written as timestamp_submitted_utc and
timestamp_scheduled_utc, converted with the rule the sensor extractor used to
make timestamp_completed_utc (parse_ts: attach Europe/Berlin, convert to UTC;
in the repeated autumn hour zoneinfo's fold=0 picks the first occurrence).
That rule reproduces timestamp_completed_utc from timestamp_completed on all
1,390,523 rows of the 20260920T231153Z build, and on all 1,558,442 rows of
ext_20260510. Sub-second precision is kept.
A scheduling time more than 1 s after completion (1 record in that build) is
impossible and is left empty; the count is in the column report.
hellinger_logical has r=0.999 with the target, so it must stay out of every
feature set. The models read only the columns named in experiments/common.py.

USAGE
-----
python3 10_clean_dataset.py \
  --input  data/stage09_final_ml_dataset.parquet \
  --output data/stage10_cleaned_final_dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# The circuit features kept (see RULES above).
CIRCUIT_FEATURES = (
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

# ---------------------------------------------------------------------------
# Column rules. First match wins; every input column must match one.
# action: target | id | meta | feature | drop
# ---------------------------------------------------------------------------

EXACT = {
    # target and identity
    "hellinger_native": ("target", "Model target (TARGET_COL)."),
    "parent_id": ("id", "Job id; group key for the train/test split."),
    "sub_id": ("id", "Unique circuit id."),
    "executed_resource": ("id", "Device."),
    "timestamp_completed_utc": ("id", "Completion time (UTC); month pools and time splits."),
    # leakage
    "hellinger_logical": ("meta", "Metadata only: r=0.999 with the target, never a feature."),
    # job metadata
    "id": ("meta", "Job id as recorded (identical to parent_id); metadata."),
    "status": ("meta", "Job status; COMPLETED on every kept row after the CANCELLED filter."),
    "sensor_merge_found": ("drop", "Constant True."),
    "batch_index": ("drop", "Batch position; submission metadata, left out by decision 2026-09-21."),
    "batch_size": ("meta", "Circuits in the batch; metadata, not a feature."),
    "is_batch_job": ("drop", "Submission metadata, left out by decision 2026-09-21."),
    # time and extraction bookkeeping
    "timestamp_completed": ("drop", "Local-time copy of timestamp_completed_utc."),
    "timestamp_submitted": ("meta", "Submission time; written as timestamp_submitted_utc."),
    "timestamp_scheduled": ("meta", "Scheduling time; written as timestamp_scheduled_utc."),
    "window_start_utc": ("drop", "Sensor extraction bookkeeping."),
    "window_end_utc": ("drop", "Sensor extraction bookkeeping; equals timestamp_completed_utc."),
    "calibration_window_start_utc": ("drop", "Calibration extraction bookkeeping."),
    "calibration_window_end_utc": ("drop", "Calibration extraction bookkeeping."),
    "calibration_sample_ts_utc": ("drop", "Calibration sample time; dropped by decision 2026-09-21."),
}

PATTERNS = [
    (r".+__(count|fallback_used)$", "drop",
     "Sensor extractor provenance; count==0 exactly when the aggregate is null."),
    (r".+__(mean|min|max|sd|value)$", "feature", "Sensor window aggregate."),
    (r"^(QB\d+_(anharmonicity|frequency|fidelity_1qb_cliffords_xy))$", "drop",
     "Placeholder calibration metric: not published for this period (empty)."),
    (r"^(QB\d+_(fidelity_1qb_gates_averaged|single_shot_readout_fidelity)"
     r"|TC_\d+_\d+_(cz_gate_fidelity|fidelity_2qb_cliffords_averaged))$", "drop",
     "Calibration fidelity arrives from the monitoring system as an integer, so it is always 0 (or 1)."),
    (r"^readout_error_QB\d+$", "drop", "Derived: mean of error_0_1 and error_1_0."),
    (r"^QB\d+_(t1_time|t2_time|t2_echo_time|error_0_1|error_1_0)$", "feature",
     "Calibration value (scaled integer)."),
    (r"^(logical|native)_dag_", "drop",
     "DAG storage metadata, or a duplicate of a circuit feature."),
    (r"^(logical|native|delta|ratio)_", "drop",
     "Circuit column not in the feature set chosen on 2026-09-21."),
]
PATTERNS = [(re.compile(p), a, r) for p, a, r in PATTERNS]


SENSOR_AGG = re.compile(r".+__(mean|min|max|sd|value)$")

# Minimum share of kept rows a feature must have a reading on. Below this a
# channel carries almost no signal but marks the few rows it covers.
MIN_COVERAGE = 0.01

# Hellinger distances lie in [0, 1]. Floating-point rounding in stage 04 puts a
# handful of values at 1 + 2e-16; those are clipped to 1. Anything further out
# is a real error and stops the run.
HELLINGER_COLUMNS = ("hellinger_native", "hellinger_logical")
HELLINGER_TOL = 1e-9


def clip_unit_interval(vals: np.ndarray, name: str) -> np.ndarray:
    if (vals < -HELLINGER_TOL).any() or (vals > 1 + HELLINGER_TOL).any():
        raise ValueError(f"{name} outside [0, 1] beyond rounding tolerance")
    return np.clip(vals, 0.0, 1.0)


# Metadata columns written as float64; the other metadata stay strings.
META_NUMERIC = {"hellinger_logical", "batch_size"}

# Naive Europe/Berlin columns converted to UTC and renamed on output.
LOCAL_TZ = ZoneInfo("Europe/Berlin")
LOCAL_TIME_COLUMNS = {
    "timestamp_submitted": "timestamp_submitted_utc",
    "timestamp_scheduled": "timestamp_scheduled_utc",
}


def local_to_utc(values: pd.Series) -> pd.Series:
    """Naive Europe/Berlin strings -> UTC timestamps, same rule as parse_ts()."""
    uniq = values.dropna().unique()
    conv = {}
    for v in uniq:
        dt = pd.Timestamp(str(v).strip()).to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=LOCAL_TZ)
        conv[v] = pd.Timestamp(dt.astimezone(timezone.utc))
    return pd.to_datetime(values.map(conv), utc=True)


def classify(col: str) -> tuple[str, str]:
    if col in CIRCUIT_FEATURES:
        return "feature", "Circuit feature, chosen 2026-09-21."
    if col in EXACT:
        return EXACT[col]
    for rx, action, reason in PATTERNS:
        if rx.match(col):
            return action, reason
    raise KeyError(col)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NULL_STRINGS = ["", "nan", "NaN", "None", "none", "null", "NULL", "<NA>"]


def to_float(arr: pa.Array | pa.ChunkedArray) -> tuple[np.ndarray, int]:
    """Parse a column to float64. Returns (values, count of unparseable values)."""
    s = arr.to_pandas()
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        s = s.where(~s.isin(NULL_STRINGS))
    if pd.api.types.is_bool_dtype(s):
        s = s.astype("float64")
    num = pd.to_numeric(s, errors="coerce")
    bad = int((num.isna() & s.notna()).sum())
    return num.to_numpy(dtype="float64"), bad


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage 10: clean and type the ML dataset.")
    ap.add_argument("--input", required=True, help="Stage 09 parquet")
    ap.add_argument("--output", required=True, help="Stage 10 parquet")
    ap.add_argument("--report", default=None,
                    help="Column report JSON (default: <output>.columns.json)")
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--progress-every", type=int, default=20,
                    help="Report progress every N row groups")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    src = Path(args.input)
    out = Path(args.output)
    report_path = Path(args.report) if args.report else out.with_suffix(".columns.json")

    pf = pq.ParquetFile(str(src))
    names = pf.schema_arrow.names
    n_in = pf.metadata.num_rows
    print(f"[INFO] 10_clean_dataset.py started")
    print(f"[INFO] Input        : {src}  ({n_in:,} rows, {len(names)} columns)")
    print(f"[INFO] Output       : {out}")

    # ---- 1. classify every column --------------------------------------------
    decisions: dict[str, dict] = {}
    unknown = []
    for c in names:
        try:
            action, reason = classify(c)
        except KeyError:
            unknown.append(c)
            continue
        decisions[c] = {"action": action, "reason": reason}
    if unknown:
        print(f"[ERROR] {len(unknown)} column(s) match no rule in COLUMN_RULES; classify them "
              f"in pipeline/10_clean_dataset.py first:", file=sys.stderr)
        for c in unknown:
            print(f"  - {c}", file=sys.stderr)
        return 2
    for required in ("hellinger_native", "parent_id", "executed_resource", "status"):
        if required not in names:
            print(f"[ERROR] required column missing: {required}", file=sys.stderr)
            return 2

    # ---- 2. row filter: whole cancelled jobs ---------------------------------
    meta = pf.read(columns=["parent_id", "status"]).to_pandas()
    cancelled_jobs = set(meta.loc[meta["status"] == "CANCELLED", "parent_id"])
    keep_mask = ~meta["parent_id"].isin(cancelled_jobs).to_numpy()
    n_out = int(keep_mask.sum())
    print(f"[INFO] Row filter   : dropping {len(cancelled_jobs):,} CANCELLED jobs = "
          f"{n_in - n_out:,} rows; keeping {n_out:,}")
    del meta

    # ---- 3. data-driven filter on rule-kept features -------------------------
    features = [c for c, d in decisions.items() if d["action"] == "feature"]
    print(f"[INFO] Rule-kept features: {len(features)}; checking for empty/constant columns")
    bad_parse: dict[str, int] = {}
    degenerate: dict[str, str] = {}   # column -> "empty" | "constant (<v>)"
    for i, c in enumerate(features, 1):
        vals, bad = to_float(pf.read(columns=[c]).column(0))
        if bad:
            bad_parse[c] = bad
        v = vals[keep_mask]
        v = v[~np.isnan(v)]
        if v.size == 0:
            degenerate[c] = "empty"
        elif np.unique(v).size == 1:
            degenerate[c] = f"constant ({v[0]:g})"
        elif v.size < MIN_COVERAGE * n_out:
            degenerate[c] = (f"sparse ({v.size:,} of {n_out:,} rows, "
                             f"{100 * v.size / n_out:.3f}%)")
        if i % 50 == 0:
            print(f"[INFO]   checked {i}/{len(features)}")

    # A sensor channel is judged as a whole: its aggregates are dropped only if
    # every one of them is empty or constant. A constant min (a counter that
    # always bottoms out at 0) is still a valid reading, and dropping it alone
    # would leave the channel with an incomplete set of aggregates.
    channels: dict[str, list[str]] = {}
    for c in features:
        if SENSOR_AGG.match(c):
            channels.setdefault(c.rsplit("__", 1)[0], []).append(c)
    for c in features:
        if c not in degenerate:
            continue
        if c in CIRCUIT_FEATURES:
            # Chosen by hand; flag, never drop.
            decisions[c]["reason"] += f" NOTE: {degenerate[c]} on the kept rows."
            print(f"[WARN] circuit feature {c} is {degenerate[c]}; kept")
            continue
        m = SENSOR_AGG.match(c)
        if m:
            siblings = channels[c.rsplit("__", 1)[0]]
            if not all(x in degenerate for x in siblings):
                decisions[c]["reason"] += f" NOTE: {degenerate[c]}; kept with its channel."
                continue
        reason = (f"{degenerate[c].capitalize()}."
                  if degenerate[c].startswith("sparse")
                  else f"{degenerate[c].capitalize()} on every kept row.")
        decisions[c] = {"action": "drop", "reason": reason}
    if bad_parse:
        print(f"[ERROR] non-numeric values in feature columns (would silently become NaN):",
              file=sys.stderr)
        for c, n in list(bad_parse.items())[:20]:
            print(f"  - {c}: {n:,} values", file=sys.stderr)
        return 3

    keep_cols = [c for c in names if decisions[c]["action"] != "drop"]
    n_feat = sum(decisions[c]["action"] == "feature" for c in keep_cols)
    n_meta = sum(decisions[c]["action"] == "meta" for c in keep_cols)
    float_cols = {c for c in keep_cols
                  if decisions[c]["action"] in ("feature", "target") or c in META_NUMERIC}
    n_auto = sum(1 for c in features if decisions[c]["action"] == "drop")
    print(f"[INFO] Data-driven drops: {n_auto}")
    print(f"[INFO] Keeping {len(keep_cols)} of {len(names)} columns "
          f"({n_feat} features + target + {n_meta} metadata + "
          f"{len(keep_cols) - n_feat - n_meta - 1} identifiers)")

    # ---- 4. write --------------------------------------------------------------
    fields = []
    for c in keep_cols:
        if c in float_cols:
            fields.append(pa.field(c, pa.float64()))
        elif c == "timestamp_completed_utc":
            fields.append(pa.field(c, pa.timestamp("us", tz="UTC")))
        elif c in LOCAL_TIME_COLUMNS:
            fields.append(pa.field(LOCAL_TIME_COLUMNS[c], pa.timestamp("us", tz="UTC")))
        else:
            fields.append(pa.field(c, pa.string()))
    schema = pa.schema(fields)

    tmp = out.with_name(out.name + ".tmp")
    offset = 0
    written = 0
    n_sched_dropped = 0
    with pq.ParquetWriter(str(tmp), schema, compression=args.compression) as writer:
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=keep_cols)
            m = keep_mask[offset: offset + tbl.num_rows]
            offset += tbl.num_rows
            if not m.any():
                continue
            tbl = tbl.filter(pa.array(m))
            arrays = []
            completed = pd.to_datetime(tbl.column("timestamp_completed_utc").to_pandas(),
                                       utc=True, errors="coerce")
            for c, f in zip(keep_cols, fields):
                col = tbl.column(c)
                if c in LOCAL_TIME_COLUMNS:
                    ts = local_to_utc(col.to_pandas())
                    if c == "timestamp_scheduled":
                        # The scheduler log occasionally records a scheduling time
                        # after completion (job 215418: two days later). Beyond the
                        # 1 s rounding of timestamp_completed_utc that is impossible,
                        # so the value is dropped rather than published.
                        late = (ts - completed) > pd.Timedelta(seconds=1)
                        late = late.fillna(False).to_numpy()
                        if late.any():
                            n_sched_dropped += int(late.sum())
                            ts = ts.mask(late)
                    arrays.append(pa.array(ts, type=f.type))
                elif f.name in float_cols:
                    vals, _ = to_float(col)
                    if f.name in HELLINGER_COLUMNS:
                        vals = clip_unit_interval(vals, f.name)
                    arrays.append(pa.array(vals, type=pa.float64(), from_pandas=True))
                elif f.name == "timestamp_completed_utc":
                    ts = pd.to_datetime(col.to_pandas(), utc=True, errors="coerce")
                    arrays.append(pa.array(ts, type=f.type))
                else:
                    arrays.append(col.cast(pa.string()))
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
            written += int(m.sum())
            if (rg + 1) % args.progress_every == 0:
                print(f"[INFO] row_groups={rg + 1}/{pf.num_row_groups} rows_written={written:,} "
                      f"elapsed={time.time() - t0:.0f}s")
    if written != n_out:
        print(f"[ERROR] wrote {written:,} rows, expected {n_out:,}", file=sys.stderr)
        return 4
    tmp.replace(out)

    # ---- 5. report -------------------------------------------------------------
    report = {
        "input": str(src),
        "output": str(out),
        "rows_in": n_in,
        "rows_out": n_out,
        "cancelled_jobs_dropped": len(cancelled_jobs),
        "columns_in": len(names),
        "columns_out": len(keep_cols),
        "scheduled_after_completed_set_null": n_sched_dropped,
        "features_out": n_feat,
        "metadata_out": n_meta,
        "data_driven_drops": sorted(
            c for c in features if decisions[c]["action"] == "drop"),
        "columns": decisions,
    }
    report_path.write_text(json.dumps(report, indent=1))

    print(f"[DONE] Rows in / out          : {n_in:,} / {n_out:,}")
    print(f"[DONE] Columns in / out       : {len(names)} / {len(keep_cols)}")
    print(f"[DONE] Features (float64)     : {n_feat}")
    print(f"[DONE] Metadata (not features): {n_meta}")
    print(f"[DONE] Scheduled > completed : {n_sched_dropped} (timestamp_scheduled_utc left empty)")
    print(f"[DONE] Column report          : {report_path}")
    print(f"[DONE] Output size bytes      : {out.stat().st_size:,}")
    print(f"[DONE] Total execution time (s): {time.time() - t0:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
