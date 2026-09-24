#!/usr/bin/env python3
"""
08_merge_circuit_data_and_sensor_data.py
========================================

PURPOSE
-------
Merge the prepared circuit dataset with the sensor dataset in a way that works
for both:

1. non-expanded circuit datasets, where the circuit table still uses `id`
2. expanded batch-job datasets, where the circuit table contains:
      parent_id, sub_id, batch_index, batch_size, is_batch_job

THIS VERSION SUPPORTS
---------------------
- CSV input/output
- Parquet input/output

The file type is detected automatically from the file extension:
- .csv
- .parquet

WHY THIS SCRIPT IS NEEDED
-------------------------
After stage 01, one original job may expand into multiple circuit rows:

    parent_id = 100001
    sub_id    = 100001_0
    sub_id    = 100001_1
    sub_id    = 100001_2

The sensor dataset was extracted using the original job id only:

    id = 100001

Therefore, the correct merge rule is:

- circuit side join key:
      parent_id   if available
      otherwise id

- sensor side join key:
      id   (or the column specified by --sensor-id-col)

This means the same sensor row is repeated for every expanded sub-job belonging
to the same parent job. That is the correct behavior because all those sub-jobs
share the same execution and environmental context: the upstream extractor
aggregates the monitoring system sensor values over the window
[timestamp_completed - WINDOW_MIN, timestamp_completed] (default WINDOW_MIN = 5
minutes) of the parent job, and all sub-jobs belonging to that parent run within
the same hardware execution.

DESIGN
------
This script:
1. Loads the circuit dataset fully.
2. Chooses the correct circuit-side join key:
      - parent_id if present
      - otherwise id
3. Reads the sensor dataset:
      - fully if small enough
      - in chunks/batches if very large
4. Filters sensor rows to only the needed ids
5. Optionally deduplicates or aggregates duplicate sensor rows per id
6. Removes sensor columns that would duplicate circuit columns
7. LEFT JOINs sensor data onto the circuit table so all circuit rows are kept

KEY ROBUSTNESS FEATURES
-----------------------
✔ Works for expanded and non-expanded circuit datasets
✔ Works for CSV and Parquet
✔ Standardizes join keys to string on both sides
✔ Handles very large sensor files by chunking/batching
✔ Supports duplicate sensor rows per id:
    - first / last / none
    - or aggregate numeric columns by mean and non-numeric columns by first
✔ Preserves all circuit rows by using a LEFT JOIN
✔ Adds merge sanity flag:
    - sensor_merge_found
✔ Removes duplicate/overlapping columns from the sensor side before merge

DUPLICATE COLUMN POLICY
-----------------------
If the sensor dataframe contains columns that already exist in the circuit
dataframe, those overlapping sensor columns are dropped BEFORE merge, except:

- the sensor join column itself, which is renamed to `sensor_source_id`

This prevents creation of columns like:
- timestamp_scheduled_x
- timestamp_scheduled_y
- status_x
- status_y

The circuit-side column is kept as authoritative.

PARALLEL SUPPORT
----------------
When the sensor file is very large and read in chunks/batches, you may use:

    --jobs N

to filter multiple sensor chunks in parallel before they are written to the
temporary filtered file.

INPUTS
------
--circuit : circuit dataset CSV or Parquet
--sensor  : sensor dataset CSV or Parquet containing original job id
--output  : merged dataset CSV or Parquet

OPTIONAL
--------
--sensor-id-col          : sensor id column name (default: id)
--sensor-chunksize       : rows/batch size for streaming large sensor file
--small-threshold-gb     : below this size, load sensor fully
--dedup-policy           : first / last / none
--aggregate-duplicates   : aggregate duplicate sensor rows per id
--strict-expanded-check  : fail if expanded dataset markers are inconsistent
--progress-every-chunks  : print progress every N chunks in large-file mode
--jobs                   : worker processes for large-file chunk filtering
--parquet-compression    : Parquet compression codec for output (default: snappy)

EXAMPLES
--------
CSV -> Parquet:
python3 08_merge_circuit_data_and_sensor_data.py \
  --circuit stage07_with_dags.csv \
  --sensor  circuit_job_and_sensor_and_calibration_data.csv \
  --output  stage08_merged.parquet \
  --aggregate-duplicates

Parquet -> Parquet:
python3 08_merge_circuit_data_and_sensor_data.py \
  --circuit /path/to/stage07_with_dags.parquet \
  --sensor  /path/to/circuit_job_and_sensor_and_calibration_data.parquet \
  --output  /path/to/stage08_merged.parquet \
  --strict-expanded-check \
  --dedup-policy none \
  --jobs 32 \
  --parquet-compression snappy
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, Iterator, List, Set, Tuple

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


# ---------------------------------------------------------------------
# File format helpers
# ---------------------------------------------------------------------

def _detect_file_type(path: str) -> str:
    """
    Detect file type from extension.
    """
    lower = path.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".parquet"):
        return "parquet"
    raise ValueError(
        f"Unsupported file extension for: {path}\n"
        "Supported extensions are: .csv and .parquet"
    )


def _require_pyarrow_for_parquet() -> None:
    """
    Ensure pyarrow is available when Parquet is used.
    """
    if pq is None or pa is None:
        raise ImportError(
            "Parquet support requires pyarrow. "
            "Install it with: pip install pyarrow"
        )


def _read_table_full(path: str) -> pd.DataFrame:
    """
    Read a CSV or Parquet file fully into memory.
    """
    file_type = _detect_file_type(path)

    if file_type == "csv":
        return pd.read_csv(path, low_memory=False)

    _require_pyarrow_for_parquet()
    return pd.read_parquet(path)


def _iter_table_chunks(path: str, chunksize: int) -> Iterator[pd.DataFrame]:
    """
    Yield chunks/batches from a CSV or Parquet file.

    For CSV:
    - uses pandas chunked CSV reading

    For Parquet:
    - uses pyarrow ParquetFile.iter_batches
    """
    file_type = _detect_file_type(path)

    if file_type == "csv":
        yield from pd.read_csv(path, chunksize=chunksize, low_memory=False)
        return

    _require_pyarrow_for_parquet()
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=chunksize):
        yield batch.to_pandas()


def _write_table(
    df: pd.DataFrame,
    path: str,
    parquet_compression: str = "snappy",
) -> None:
    """
    Write dataframe to CSV or Parquet based on output extension.
    """
    file_type = _detect_file_type(path)

    if file_type == "csv":
        df.to_csv(path, index=False)
        return

    _require_pyarrow_for_parquet()
    df.to_parquet(path, index=False, compression=parquet_compression)


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def _standardize_key(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Force a join column to a consistent string dtype.

    This avoids silent join failures caused by dtype mismatches such as:
    - int on one side
    - string/object on the other
    """
    df[col] = df[col].astype(str).str.strip()
    return df


def _choose_circuit_join_key(df: pd.DataFrame) -> str:
    """
    Choose the correct circuit-side join key.

    Rules:
    - if parent_id exists, use parent_id
    - otherwise fall back to id
    """
    if "parent_id" in df.columns:
        return "parent_id"

    if "id" in df.columns:
        return "id"

    raise ValueError("Circuit dataset must contain either 'parent_id' or 'id'.")


def _validate_circuit_df_for_current_flow(df: pd.DataFrame, strict_expanded_check: bool) -> None:
    """
    Validate that the circuit dataframe is consistent with the current workflow.
    """
    if "parent_id" not in df.columns and "id" not in df.columns:
        raise ValueError("Circuit dataset must contain either 'parent_id' or 'id'.")

    if strict_expanded_check and "parent_id" in df.columns:
        expected = ["sub_id", "batch_index", "batch_size", "is_batch_job"]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(
                "Expanded dataset appears incomplete. Missing column(s): "
                f"{missing}"
            )

    if strict_expanded_check and "batch_len_mismatch_circuit_vs_executed_vs_result" in df.columns:
        mismatch_count = int(pd.to_numeric(
            df["batch_len_mismatch_circuit_vs_executed_vs_result"],
            errors="coerce"
        ).fillna(0).astype(int).sum())

        if mismatch_count > 0:
            raise ValueError(
                "Circuit dataset contains rows with "
                "batch_len_mismatch_circuit_vs_executed_vs_result=True. "
                "Resolve those mismatches before sensor merge."
            )


# ---------------------------------------------------------------------
# Duplicate handling
# ---------------------------------------------------------------------

def _aggregate_sensor_duplicates(sensor_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Aggregate duplicate sensor rows per id.

    Strategy:
    - numeric columns: mean
    - non-numeric columns: first
    """
    if sensor_df.empty:
        return sensor_df

    num_cols = sensor_df.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = [c for c in sensor_df.columns if c not in num_cols and c != id_col]

    agg: Dict[str, str] = {}
    for c in num_cols:
        agg[c] = "mean"
    for c in other_cols:
        agg[c] = "first"

    return sensor_df.groupby(id_col, as_index=False).agg(agg)


def _prepare_sensor_df(
    sensor_df: pd.DataFrame,
    sensor_id_col: str,
    id_set: Set[str],
    aggregate_duplicates: bool,
    dedup_policy: str,
) -> pd.DataFrame:
    """
    Standardize, filter, and deduplicate/aggregate a sensor dataframe.
    """
    if sensor_id_col not in sensor_df.columns:
        raise ValueError(f"Sensor dataset must contain column '{sensor_id_col}'.")

    sensor_df = _standardize_key(sensor_df, sensor_id_col)
    sensor_df = sensor_df[sensor_df[sensor_id_col].isin(id_set)]

    if sensor_df.empty:
        return sensor_df

    if aggregate_duplicates:
        sensor_df = _aggregate_sensor_duplicates(sensor_df, sensor_id_col)
    else:
        if dedup_policy in ("first", "last"):
            sensor_df = sensor_df.drop_duplicates(subset=[sensor_id_col], keep=dedup_policy)

    return sensor_df


def _drop_sensor_columns_that_duplicate_circuit_columns(
    circuit_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    sensor_id_col: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop sensor-side columns that already exist in the circuit dataframe.

    Returns:
        cleaned_sensor_df, dropped_overlap_columns
    """
    circuit_cols = set(circuit_df.columns)
    sensor_cols = set(sensor_df.columns)

    overlap = sorted(
        c for c in (sensor_cols & circuit_cols)
        if c != sensor_id_col
    )

    if overlap:
        sensor_df = sensor_df.drop(columns=overlap, errors="ignore")

    return sensor_df, overlap


def _drop_exact_duplicate_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate column names while keeping the first occurrence.
    """
    return df.loc[:, ~df.columns.duplicated()].copy()


# ---------------------------------------------------------------------
# Parallel chunk helpers
# ---------------------------------------------------------------------

def _prepare_sensor_chunk_worker(
    args: Tuple[pd.DataFrame, str, Set[str]]
) -> pd.DataFrame:
    """
    Worker for large-file chunk filtering.

    Duplicate handling is intentionally disabled here and applied only after
    all filtered rows are collected, to keep semantics identical to the
    sequential path.
    """
    chunk, sensor_id_col, id_set = args
    return _prepare_sensor_df(
        sensor_df=chunk,
        sensor_id_col=sensor_id_col,
        id_set=id_set,
        aggregate_duplicates=False,
        dedup_policy="none",
    )


def _pool_chunksize(n_items: int, jobs: int) -> int:
    if n_items <= 0:
        return 1
    approx = n_items // max(1, jobs * 4)
    return max(1, min(32, approx if approx > 0 else 1))


# ---------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------

def _add_merge_sanity_columns(
    merged: pd.DataFrame,
    sensor_source_id_col: str = "sensor_source_id",
) -> pd.DataFrame:
    """
    Add simple merge sanity columns.
    """
    merged["sensor_merge_found"] = ~merged[sensor_source_id_col].isna()
    return merged


def _fmt_gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.3f} GB"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge circuit data with sensor data using parent_id->sensor id when available."
    )
    parser.add_argument("--circuit", required=True, help="Circuit CSV or Parquet (usually the smaller table).")
    parser.add_argument("--sensor", required=True, help="Sensor CSV or Parquet (can be very large).")
    parser.add_argument("--output", required=True, help="Output merged CSV or Parquet.")

    parser.add_argument(
        "--sensor-id-col",
        default="id",
        help="Sensor dataset id column name (default: id).",
    )
    parser.add_argument(
        "--sensor-chunksize",
        type=int,
        default=500_000,
        help="Rows per chunk/batch when reading a large sensor file.",
    )
    parser.add_argument(
        "--small-threshold-gb",
        type=float,
        default=2.0,
        help="If sensor file <= this size, read it fully into memory.",
    )
    parser.add_argument(
        "--dedup-policy",
        choices=["first", "last", "none"],
        default="first",
        help="How to handle duplicate sensor rows per id before merge.",
    )
    parser.add_argument(
        "--aggregate-duplicates",
        action="store_true",
        help="Aggregate duplicate sensor rows per id (numeric=mean, non-numeric=first).",
    )
    parser.add_argument(
        "--strict-expanded-check",
        action="store_true",
        help="Fail if the circuit dataset looks like an expanded dataset but key expanded columns are missing or mismatched.",
    )
    parser.add_argument(
        "--progress-every-chunks",
        type=int,
        default=1,
        help="Print progress every N sensor chunks in large-file mode.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker processes for large-file chunk filtering. Use 1 for sequential mode.",
    )
    parser.add_argument(
        "--parquet-compression",
        default="snappy",
        help="Parquet compression codec for output (default: snappy).",
    )

    args = parser.parse_args()

    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    detected_cpus = cpu_count()
    if args.jobs > detected_cpus:
        print(f"[INFO] Requested jobs ({args.jobs}) exceed detected CPU count ({detected_cpus}).")

    circuit_type = _detect_file_type(args.circuit)
    sensor_type = _detect_file_type(args.sensor)
    output_type = _detect_file_type(args.output)

    overall_start = time.time()

    print("[INFO] Sensor merge started")
    print(f"[INFO] Circuit file:           {args.circuit}")
    print(f"[INFO] Sensor file:            {args.sensor}")
    print(f"[INFO] Output file:            {args.output}")
    print(f"[INFO] Circuit type:           {circuit_type}")
    print(f"[INFO] Sensor type:            {sensor_type}")
    print(f"[INFO] Output type:            {output_type}")
    print(f"[INFO] Sensor id column:       {args.sensor_id_col}")
    print(f"[INFO] Sensor chunksize:       {args.sensor_chunksize}")
    print(f"[INFO] Small threshold (GB):   {args.small_threshold_gb}")
    print(f"[INFO] Dedup policy:           {args.dedup_policy}")
    print(f"[INFO] Aggregate duplicates:   {args.aggregate_duplicates}")
    print(f"[INFO] Strict expanded check:  {args.strict_expanded_check}")
    print(f"[INFO] Progress every chunks:  {args.progress_every_chunks}")
    print(f"[INFO] Worker processes:       {args.jobs}")
    print(f"[INFO] Parquet compression:    {args.parquet_compression}")

    # ---------------------------------------------------------
    # Load circuit data
    # ---------------------------------------------------------
    t0 = time.time()
    print("[INFO] Loading circuit dataset...")
    circuit_df = _read_table_full(args.circuit)
    print(f"[INFO] Circuit rows loaded:     {len(circuit_df)}")

    _validate_circuit_df_for_current_flow(
        circuit_df,
        strict_expanded_check=args.strict_expanded_check,
    )

    circuit_join_col = _choose_circuit_join_key(circuit_df)
    circuit_df = _standardize_key(circuit_df, circuit_join_col)
    id_set = set(circuit_df[circuit_join_col].tolist())

    print(f"[INFO] Circuit join key used:   {circuit_join_col}")
    print(f"[INFO] Unique join ids needed:  {len(id_set)}")
    print(f"[INFO] Circuit load time (s):   {time.time() - t0:.2f}")

    # ---------------------------------------------------------
    # Decide how to read sensor data
    # ---------------------------------------------------------
    sensor_size_bytes = os.path.getsize(args.sensor)
    small_threshold_bytes = int(args.small_threshold_gb * 1024 * 1024 * 1024)

    print(f"[INFO] Sensor file size:        {_fmt_gb(sensor_size_bytes)}")
    print(f"[INFO] Small-file threshold:    {_fmt_gb(small_threshold_bytes)}")

    # ---------------------------------------------------------
    # Case 1: small sensor file -> load fully
    # ---------------------------------------------------------
    if sensor_size_bytes <= small_threshold_bytes:
        t1 = time.time()
        print("[INFO] Reading sensor file fully into memory...")

        sensor_df = _read_table_full(args.sensor)
        sensor_rows_original = len(sensor_df)
        print(f"[INFO] Sensor rows loaded:      {sensor_rows_original}")

        sensor_df = _prepare_sensor_df(
            sensor_df=sensor_df,
            sensor_id_col=args.sensor_id_col,
            id_set=id_set,
            aggregate_duplicates=args.aggregate_duplicates,
            dedup_policy=args.dedup_policy,
        )

        print(f"[INFO] Sensor rows kept:        {len(sensor_df)}")
        print(f"[INFO] Sensor prep time (s):    {time.time() - t1:.2f}")

        sensor_df, dropped_overlap_cols = _drop_sensor_columns_that_duplicate_circuit_columns(
            circuit_df=circuit_df,
            sensor_df=sensor_df,
            sensor_id_col=args.sensor_id_col,
        )

        if dropped_overlap_cols:
            print(f"[INFO] Dropped overlapping sensor columns before merge: {', '.join(dropped_overlap_cols)}")
        else:
            print("[INFO] No overlapping sensor columns needed to be dropped before merge.")

        sensor_df = sensor_df.rename(columns={args.sensor_id_col: "sensor_source_id"})

        t2 = time.time()
        print("[INFO] Performing merge...")
        merged = circuit_df.merge(
            sensor_df,
            left_on=circuit_join_col,
            right_on="sensor_source_id",
            how="left",
            suffixes=("", "__sensordup"),
        )
        merged = _drop_exact_duplicate_column_names(merged)
        merged = _add_merge_sanity_columns(merged)
        merge_time = time.time() - t2

        matched_rows = int(merged["sensor_merge_found"].sum())
        unmatched_rows = len(merged) - matched_rows

        print(f"[INFO] Merge time (s):          {merge_time:.2f}")
        print(f"[INFO] Rows with sensor match:  {matched_rows}/{len(merged)}")
        print(f"[INFO] Rows without match:      {unmatched_rows}/{len(merged)}")

        t3 = time.time()
        print("[INFO] Writing merged output...")
        _write_table(
            merged,
            args.output,
            parquet_compression=args.parquet_compression,
        )
        write_time = time.time() - t3

        total_time = time.time() - overall_start
        print("\n========== FINAL REPORT ==========")
        print(f"Saved merged dataset to:      {args.output}")
        print(f"Circuit join key used:        {circuit_join_col}")
        print(f"Sensor join key used:         {args.sensor_id_col}")
        print(f"Circuit rows:                 {len(circuit_df)}")
        print(f"Sensor rows original:         {sensor_rows_original}")
        print(f"Sensor rows after filter:     {len(sensor_df)}")
        print(f"Output rows:                  {len(merged)}")
        print(f"Rows with sensor match:       {matched_rows}/{len(merged)}")
        print(f"Rows without sensor match:    {unmatched_rows}/{len(merged)}")
        print(f"Sensor overlap cols dropped:  {len(dropped_overlap_cols)}")
        print(f"Output write time (s):        {write_time:.2f}")
        print(f"Total execution time (s):     {total_time:.2f}")
        print("==================================")
        return

    # ---------------------------------------------------------
    # Case 2: large sensor file -> chunked filter, temp CSV, then merge
    # ---------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        filtered_path = os.path.join(td, "sensor_filtered.csv")
        first_write = True

        total_sensor_rows_seen = 0
        total_sensor_rows_kept = 0
        chunk_count = 0
        chunk_start = time.time()

        print("[INFO] Large sensor file detected; switching to chunked mode...")
        print(f"[INFO] Temporary filtered file: {filtered_path}")

        if args.jobs <= 1:
            for chunk in _iter_table_chunks(args.sensor, args.sensor_chunksize):
                chunk_count += 1
                rows_in_chunk = len(chunk)
                total_sensor_rows_seen += rows_in_chunk

                filtered_chunk = _prepare_sensor_df(
                    sensor_df=chunk,
                    sensor_id_col=args.sensor_id_col,
                    id_set=id_set,
                    aggregate_duplicates=False,
                    dedup_policy="none",
                )

                kept_in_chunk = len(filtered_chunk)
                total_sensor_rows_kept += kept_in_chunk

                if not filtered_chunk.empty:
                    filtered_chunk.to_csv(
                        filtered_path,
                        mode="w" if first_write else "a",
                        header=first_write,
                        index=False,
                    )
                    first_write = False

                if args.progress_every_chunks > 0 and (
                    chunk_count % args.progress_every_chunks == 0
                ):
                    elapsed = time.time() - chunk_start
                    print(
                        f"[INFO] Sensor chunk progress | "
                        f"chunks={chunk_count} | "
                        f"rows seen={total_sensor_rows_seen} | "
                        f"rows kept={total_sensor_rows_kept} | "
                        f"elapsed={elapsed:.2f}s"
                    )
        else:
            buffered_chunks: List[pd.DataFrame] = []

            def flush_buffer(buffer: List[pd.DataFrame]) -> Tuple[int, int, bool]:
                nonlocal first_write
                if not buffer:
                    return 0, 0, first_write

                worker_args = [(c, args.sensor_id_col, id_set) for c in buffer]
                seen = sum(len(c) for c in buffer)
                kept = 0

                with Pool(processes=args.jobs) as pool:
                    results = list(
                        pool.map(
                            _prepare_sensor_chunk_worker,
                            worker_args,
                            chunksize=_pool_chunksize(len(worker_args), args.jobs),
                        )
                    )

                for filtered_chunk in results:
                    kept += len(filtered_chunk)
                    if not filtered_chunk.empty:
                        filtered_chunk.to_csv(
                            filtered_path,
                            mode="w" if first_write else "a",
                            header=first_write,
                            index=False,
                        )
                        first_write = False

                return seen, kept, first_write

            for chunk in _iter_table_chunks(args.sensor, args.sensor_chunksize):
                chunk_count += 1
                buffered_chunks.append(chunk)

                if len(buffered_chunks) >= max(2, args.jobs * 2):
                    seen, kept, _ = flush_buffer(buffered_chunks)
                    total_sensor_rows_seen += seen
                    total_sensor_rows_kept += kept
                    buffered_chunks = []

                    if args.progress_every_chunks > 0 and (
                        chunk_count % args.progress_every_chunks == 0
                    ):
                        elapsed = time.time() - chunk_start
                        print(
                            f"[INFO] Sensor chunk progress | "
                            f"chunks={chunk_count} | "
                            f"rows seen={total_sensor_rows_seen} | "
                            f"rows kept={total_sensor_rows_kept} | "
                            f"elapsed={elapsed:.2f}s"
                        )

            if buffered_chunks:
                seen, kept, _ = flush_buffer(buffered_chunks)
                total_sensor_rows_seen += seen
                total_sensor_rows_kept += kept
                buffered_chunks = []

        filter_time = time.time() - chunk_start
        print(
            f"[INFO] Chunk filtering complete | "
            f"chunks={chunk_count} | rows seen={total_sensor_rows_seen} | "
            f"rows kept={total_sensor_rows_kept} | time={filter_time:.2f}s"
        )

        if first_write:
            circuit_df["sensor_merge_found"] = False
            print("[INFO] No matching sensor rows found after chunk filtering.")
            print("[INFO] Writing circuit-only output...")
            t_out = time.time()
            _write_table(
                circuit_df,
                args.output,
                parquet_compression=args.parquet_compression,
            )
            write_time = time.time() - t_out

            total_time = time.time() - overall_start
            print("\n========== FINAL REPORT ==========")
            print(f"No matching sensor rows found. Wrote circuit-only output to: {args.output}")
            print(f"Circuit join key used:        {circuit_join_col}")
            print(f"Sensor join key used:         {args.sensor_id_col}")
            print(f"Circuit rows:                 {len(circuit_df)}")
            print(f"Sensor rows seen:             {total_sensor_rows_seen}")
            print(f"Sensor rows kept:             0")
            print(f"Output rows:                  {len(circuit_df)}")
            print(f"Rows with sensor match:       0/{len(circuit_df)}")
            print(f"Output write time (s):        {write_time:.2f}")
            print(f"Total execution time (s):     {total_time:.2f}")
            print("==================================")
            return

        t_load_filtered = time.time()
        print("[INFO] Loading filtered sensor subset...")
        sensor_filtered = pd.read_csv(filtered_path, low_memory=False)
        sensor_filtered = _standardize_key(sensor_filtered, args.sensor_id_col)
        print(f"[INFO] Filtered sensor rows loaded: {len(sensor_filtered)}")
        print(f"[INFO] Filtered subset load time (s): {time.time() - t_load_filtered:.2f}")

        t_dedup = time.time()
        print("[INFO] Applying duplicate handling on filtered subset...")
        if args.aggregate_duplicates:
            sensor_filtered = _aggregate_sensor_duplicates(sensor_filtered, args.sensor_id_col)
        else:
            if args.dedup_policy in ("first", "last"):
                sensor_filtered = sensor_filtered.drop_duplicates(
                    subset=[args.sensor_id_col],
                    keep=args.dedup_policy,
                )
        print(f"[INFO] Filtered sensor rows after dedup/agg: {len(sensor_filtered)}")
        print(f"[INFO] Dedup/aggregation time (s): {time.time() - t_dedup:.2f}")

        sensor_filtered, dropped_overlap_cols = _drop_sensor_columns_that_duplicate_circuit_columns(
            circuit_df=circuit_df,
            sensor_df=sensor_filtered,
            sensor_id_col=args.sensor_id_col,
        )

        if dropped_overlap_cols:
            print(f"[INFO] Dropped overlapping sensor columns before merge: {', '.join(dropped_overlap_cols)}")
        else:
            print("[INFO] No overlapping sensor columns needed to be dropped before merge.")

        sensor_filtered = sensor_filtered.rename(columns={args.sensor_id_col: "sensor_source_id"})

        t_merge = time.time()
        print("[INFO] Performing merge...")
        merged = circuit_df.merge(
            sensor_filtered,
            left_on=circuit_join_col,
            right_on="sensor_source_id",
            how="left",
            suffixes=("", "__sensordup"),
        )
        merged = _drop_exact_duplicate_column_names(merged)
        merged = _add_merge_sanity_columns(merged)
        merge_time = time.time() - t_merge

        matched_rows = int(merged["sensor_merge_found"].sum())
        unmatched_rows = len(merged) - matched_rows

        print(f"[INFO] Merge time (s):          {merge_time:.2f}")
        print(f"[INFO] Rows with sensor match:  {matched_rows}/{len(merged)}")
        print(f"[INFO] Rows without match:      {unmatched_rows}/{len(merged)}")

        t_write = time.time()
        print("[INFO] Writing merged output...")
        _write_table(
            merged,
            args.output,
            parquet_compression=args.parquet_compression,
        )
        write_time = time.time() - t_write

        total_time = time.time() - overall_start
        print("\n========== FINAL REPORT ==========")
        print(f"Saved merged dataset to:      {args.output}")
        print(f"Circuit join key used:        {circuit_join_col}")
        print(f"Sensor join key used:         {args.sensor_id_col}")
        print(f"Circuit rows:                 {len(circuit_df)}")
        print(f"Sensor rows seen:             {total_sensor_rows_seen}")
        print(f"Sensor rows kept in filter:   {total_sensor_rows_kept}")
        print(f"Filtered rows after dedup:    {len(sensor_filtered)}")
        print(f"Output rows:                  {len(merged)}")
        print(f"Rows with sensor match:       {matched_rows}/{len(merged)}")
        print(f"Rows without sensor match:    {unmatched_rows}/{len(merged)}")
        print(f"Sensor overlap cols dropped:  {len(dropped_overlap_cols)}")
        print(f"Output write time (s):        {write_time:.2f}")
        print(f"Total execution time (s):     {total_time:.2f}")
        print("==================================")


if __name__ == "__main__":
    main()