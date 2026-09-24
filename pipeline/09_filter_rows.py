#!/usr/bin/env python3
"""
09_filter_valid_rows.py
=======================

PURPOSE
-------
Final cleanup stage after circuit/sensor/calibration merge.

This version is intended to run AFTER:

    07_add_dags.py
    08_merge_circuit_data_and_sensor_data.py

WHAT THIS SCRIPT DOES
---------------------
1. Optionally drops large/intermediate columns from the output Parquet
2. Keeps DAG retrieval metadata needed for later lookup in dag.sqlite
3. Filters OUT rows that do not have BOTH:
   - logical data available
   - native data available

DEFAULT ROW FILTER POLICY
-------------------------
A row is kept only if at least one logical column and at least one native column
from the configured required sets are usable.

DEFAULT logical availability columns:
- logical_dag_ok
- logical_dag_ref
- logical_depth
- logical_n_qubits
- logical_sim_status
- hellinger_logical

DEFAULT native availability columns:
- native_dag_ok
- native_dag_ref
- native_depth
- native_n_qubits
- native_sim_status
- hellinger_native

RULE
----
Keep row if:
    logical side has at least one usable value
AND
    native side has at least one usable value

IMPORTANT KEEP POLICY
---------------------
This script intentionally keeps the following columns because you requested them
for later DAG retrieval and decomposition analysis:

- logical_depth_before_decompose
- logical_depth_after_decompose
- native_depth_before_decompose
- native_depth_after_decompose
- logical_dag_ref
- logical_dag_hash
- logical_dag_blob_bytes
- native_dag_ref
- native_dag_hash
- native_dag_blob_bytes

UPDATED WORKFLOW NOTES
----------------------
This version is aligned with the updated workflow where:
- stage 01 expands batch jobs BEFORE normalization
- downstream stages operate row-wise
- stage 07 may already have dropped some raw circuit/result columns
- stage 08 adds sensor_merge_found and sensor/calibration columns
- stage-01 raw batch diagnostic columns may still exist in some datasets

Therefore:
- missing optional columns are tolerated
- keep columns always override drop columns
- row filtering is based on availability, not on one exact schema only
- raw batch diagnostic columns are treated as intermediate/debug columns by default

DESIGN IMPROVEMENT
------------------
Older Parquet versions of this stage wrote many temporary .part_*.parquet files
and then combined them at the end. That final combine step can become memory-
heavy on large datasets.

This updated version avoids that problem by:
- reading the input Parquet in chunks
- filtering each chunk in memory
- appending directly to ONE output Parquet file via pyarrow.ParquetWriter

This is much safer for large-scale workflows.

INPUT
-----
--input   : input Parquet dataset
--output  : final cleaned Parquet dataset

OPTIONAL
--------
--chunksize             : rows per chunk
--progress-every        : print progress every N rows
--drop-cols             : extra columns to drop
--drop-cols-file        : text file containing extra columns to drop
--keep-cols             : extra columns to force-keep
--keep-cols-file        : text file containing extra columns to force-keep
--drop-dag-ref-cols     : compatibility flag; keep columns still win
--no-default-drops      : disable built-in default drop list
--logical-required-cols : override logical availability columns
--native-required-cols  : override native availability columns
--disable-row-filter    : keep all rows, only perform column dropping
--threads               : DuckDB thread count
--memory-limit          : DuckDB memory limit, e.g. 64GB
--compression           : output Parquet compression codec

USAGE
-----
Basic run:
python3 09_filter_valid_rows.py \
  --input /path/to/stage08_merged.parquet \
  --output /path/to/stage09_final_ml_dataset.parquet

Chunked run with progress:
python3 09_filter_valid_rows.py \
  --input /path/to/stage08_merged.parquet \
  --output /path/to/stage09_final_ml_dataset.parquet \
  --chunksize 10000 \
  --progress-every 100000

Keep all rows but still drop heavy columns:
python3 09_filter_valid_rows.py \
  --input /path/to/stage08_merged.parquet \
  --output /path/to/stage09_final_ml_dataset.parquet \
  --disable-row-filter

Add extra drop columns:
python3 09_filter_valid_rows.py \
  --input /path/to/stage08_merged.parquet \
  --output /path/to/stage09_final_ml_dataset.parquet \
  --drop-cols owner budget calibration_status

Override required availability columns:
python3 09_filter_valid_rows.py \
  --input /path/to/stage08_merged.parquet \
  --output /path/to/stage09_final_ml_dataset.parquet \
  --logical-required-cols logical_dag_ok logical_depth hellinger_logical \
  --native-required-cols native_dag_ok native_depth hellinger_native

Example for your workflow:
python3 09_filter_valid_rows.py \
  --input /path/to/stage08_merged.parquet \
  --output /path/to/stage09_final_ml_dataset.parquet \
  --chunksize 10000 \
  --progress-every 100000 \
  --compression zstd
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, List, Set, Tuple

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# =====================================================================
# DEFAULT DROP / KEEP CONFIGURATION
# =====================================================================

DEFAULT_DROP_COLS = [
    # -----------------------------------------------------------------
    # Heavy raw payload columns
    # These are usually large string payloads and are often no longer
    # needed once the dataset has reached the final ML-preparation stage.
    # -----------------------------------------------------------------
    "circuit",
    "executed_circuit",
    "circuit_qasm2_norm",
    "executed_circuit_qasm2_norm",
    "executed_circuit_qasm2_norm_stripped",
    "circuit_best_for_sim",
    "executed_circuit_best_for_sim",
    "circuit_norm_text",
    "executed_circuit_norm_text",
    "result",
    "result_expanded",
    "ideal_result_logical",
    "ideal_result_native",
    "result_aligned_logical",
    "result_aligned_native",

    # -----------------------------------------------------------------
    # Parse / normalization / simulation debug columns
    # These are mainly useful for debugging upstream issues, not for the
    # final clean ML dataset.
    # -----------------------------------------------------------------
    "circuit_parse_error",
    "executed_circuit_parse_error",
    "result_parse_error",
    "ideal_result_logical_error",
    "ideal_result_native_error",
    "circuit_format",
    "executed_circuit_format",
    "circuit_parse_ok",
    "executed_circuit_parse_ok",
    "circuit_kind",
    "executed_circuit_kind",
    "circuit_n",
    "executed_circuit_n",
    "circuit_source_syntax",
    "executed_circuit_source_syntax",
    "circuit_parser_used",
    "executed_circuit_parser_used",
    "circuit_norm_format",
    "executed_circuit_norm_format",
    "circuit_best_for_sim_format",
    "executed_circuit_best_for_sim_format",
    "result_kind",
    "result_n",
    "batch_len_mismatch_circuit_vs_executed_vs_result",

    # -----------------------------------------------------------------
    # Stage-01 raw batch diagnostics
    # These describe how stage 01 interpreted raw batch payloads. They are
    # treated as intermediate debugging fields by default.
    # -----------------------------------------------------------------
    "circuit_batch_raw_kind",
    "circuit_batch_raw_n",
    "circuit_batch_raw_error",
    "executed_circuit_batch_raw_kind",
    "executed_circuit_batch_raw_n",
    "executed_circuit_batch_raw_error",
    "result_batch_raw_kind",
    "result_batch_raw_n",
    "result_batch_raw_error",

    # -----------------------------------------------------------------
    # Redundant qubit-count columns
    # We usually keep logical_n_qubits and native_n_qubits from the feature
    # stage, so these earlier simulation-side versions can be dropped.
    # -----------------------------------------------------------------
    "logical_num_qubits",
    "native_num_qubits",

    # -----------------------------------------------------------------
    # Simulation status / debug columns
    # These can be dropped by default because later stages already contain
    # the usable outputs. Keep override still applies if needed.
    # -----------------------------------------------------------------
    "logical_sim_status",
    "native_sim_status",

    # -----------------------------------------------------------------
    # Verification / endianness debug columns
    # Useful for debugging but often not needed in the final ML dataset.
    # -----------------------------------------------------------------
    "logical_verify_status",
    "logical_verify_error",
    "logical_best_endianness_raw",
    "logical_best_endianness",
    "logical_endianness_confident",
    "logical_endianness_used_fallback",
    "native_verify_status",
    "native_verify_error",
    "native_best_endianness_raw",
    "native_best_endianness",
    "native_endianness_confident",
    "native_endianness_used_fallback",

    # -----------------------------------------------------------------
    # Hellinger bookkeeping / error columns
    # The final scalar Hellinger values are kept, but these mode/error
    # helper fields are usually not needed.
    # -----------------------------------------------------------------
    "hellinger_logical_mode_used",
    "hellinger_native_mode_used",
    "hellinger_logical_result_source",
    "hellinger_native_result_source",
    "hellinger_logical_error",
    "hellinger_native_error",

    # -----------------------------------------------------------------
    # Feature extraction debug
    # -----------------------------------------------------------------
    "feature_error",

    # -----------------------------------------------------------------
    # DAG diagnostics usually not needed in final ML output
    # Core DAG retrieval columns are preserved separately by keep rules.
    # -----------------------------------------------------------------
    "logical_dag_source_col",
    "logical_dag_attempts",
    "logical_dag_used_fallback",
    "logical_dag_error_category",
    "logical_dag_error",
    "native_dag_source_col",
    "native_dag_attempts",
    "native_dag_used_fallback",
    "native_dag_error_category",
    "native_dag_error",
    "row_drop_reason",

    # -----------------------------------------------------------------
    # Merge helper column
    # This was only needed to attach sensor rows during stage 08.
    # -----------------------------------------------------------------
    "sensor_source_id",

    # -----------------------------------------------------------------
    # Submission / audit metadata often not needed for ML
    # -----------------------------------------------------------------
    # Note: timestamp_completed is intentionally NOT dropped here. It is the
    # anchor for the sensor/calibration extraction window
    # ([timestamp_completed - WINDOW_MIN, timestamp_completed]) and is kept as
    # provenance for the merged sensor aggregates. timestamp_submitted is
    # also kept (decision 2026-09-21): stage 10 carries it as metadata.
    "owner",
    "budget",

    # -----------------------------------------------------------------
    # Calibration query audit/debug columns
    # -----------------------------------------------------------------
    "calibration_target_time_utc",
    "calibration_query_start_utc",
    "calibration_query_end_utc",
    "calibration_status",
    "calibration_resource_used",
]

DEFAULT_DAG_KEEP_COLS = [
    # Keep enough DAG metadata for later lookup and downstream DAG analysis.
    "logical_dag_ok",
    "logical_dag_ref",
    "logical_dag_hash",
    "logical_dag_blob_bytes",
    "logical_dag_depth",
    "logical_dag_num_qubits",
    "logical_dag_num_clbits",
    "logical_dag_op_nodes",
    "logical_dag_twoq_ops",
    "logical_dag_measure_ops",
    "logical_dag_layers",
    "logical_dag_gate_histogram",
    "native_dag_ok",
    "native_dag_ref",
    "native_dag_hash",
    "native_dag_blob_bytes",
    "native_dag_depth",
    "native_dag_num_qubits",
    "native_dag_num_clbits",
    "native_dag_op_nodes",
    "native_dag_twoq_ops",
    "native_dag_measure_ops",
    "native_dag_layers",
    "native_dag_gate_histogram",
]

REQUESTED_EXTRA_KEEP_COLS = [
    # These were explicitly requested for decomposition and DAG retrieval.
    "logical_depth_before_decompose",
    "logical_depth_after_decompose",
    "native_depth_before_decompose",
    "native_depth_after_decompose",
    "logical_dag_ref",
    "logical_dag_hash",
    "logical_dag_blob_bytes",
    "native_dag_ref",
    "native_dag_hash",
    "native_dag_blob_bytes",
]

DEFAULT_DAG_REF_DROP_COLS = [
    # Compatibility option.
    # Even if this is requested, explicit keep columns still win.
    "logical_dag_ref",
    "logical_dag_hash",
    "logical_dag_blob_bytes",
    "native_dag_ref",
    "native_dag_hash",
    "native_dag_blob_bytes",
]

PROTECTED_KEEP_COLS = [
    # Identity and important workflow columns that should remain by default.
    #
    # Sensor/calibration window anchor:
    # The upstream extractor uses timestamp_completed as the anchor and
    # aggregates the monitoring system sensor values over [timestamp_completed - WINDOW_MIN,
    # timestamp_completed] (default WINDOW_MIN = 5 minutes). The original
    # timestamp_scheduled anchor is kept as legacy provenance only.
    "id",
    "parent_id",
    "sub_id",
    "batch_index",
    "batch_size",
    "is_batch_job",
    "executed_resource",
    "timestamp_scheduled",
    "timestamp_scheduled_utc",
    "timestamp_completed",
    "timestamp_completed_utc",
    "window_start_utc",
    "window_end_utc",
    "status",
    "shots",
    "sensor_merge_found",
    "calibration_time_delta_seconds",
]


# =====================================================================
# DEFAULT ROW-AVAILABILITY COLUMNS
# =====================================================================

DEFAULT_LOGICAL_REQUIRED_COLS = [
    "logical_dag_ok",
    "logical_dag_ref",
    "logical_depth",
    "logical_n_qubits",
    "logical_sim_status",
    "hellinger_logical",
]

DEFAULT_NATIVE_REQUIRED_COLS = [
    "native_dag_ok",
    "native_dag_ref",
    "native_depth",
    "native_n_qubits",
    "native_sim_status",
    "hellinger_native",
]


# =====================================================================
# GENERAL HELPERS
# =====================================================================

def parse_cols_file(path: str) -> List[str]:
    """
    Read one column name per line from a text file.

    File format
    -----------
    - one column name per line
    - blank lines are ignored
    - lines starting with '#' are treated as comments
    """
    cols: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            cols.append(s)
    return cols


def unique_preserve_order(items: List[str]) -> List[str]:
    """
    Deduplicate a list while preserving original order.

    This is useful because:
    - some columns may appear in multiple drop/keep sources
    - stable ordering makes logs and output easier to reason about
    """
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def is_empty(value: Any) -> bool:
    """
    Treat common empty-like forms as missing.

    This stage often sees columns serialized from Parquet through DuckDB and
    pandas, so several string-like null forms are treated as empty.
    """
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "<na>"}


def is_truthy_string(value: Any) -> bool:
    """
    Interpret common string forms of True.

    Examples treated as true
    ------------------------
    - True
    - "true"
    - "1"
    - "yes"
    - "y"
    """
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"true", "1", "yes", "y"}


def is_nonzero_numberish(value: Any) -> bool:
    """
    Return True if value can be interpreted as a nonzero finite number.

    This helper is kept available for future rule extensions, even though the
    current row-availability logic mainly uses direct numeric validity checks.
    """
    if value is None:
        return False
    try:
        x = float(value)
        if pd.isna(x):
            return False
        return x != 0.0
    except Exception:
        return False


def has_side_available(row: dict, cols: List[str]) -> bool:
    """
    Return True if the row has at least one usable value among the given columns.

    Column-type-specific rules
    --------------------------
    - *_dag_ok columns:
        must be truthy
    - *_sim_status columns:
        accept SUCCESS / OK / TRUE-like values
    - hellinger_* columns:
        accept any finite numeric value
    - all other columns:
        only need to be non-empty

    This design allows the script to work across slightly different schemas
    without requiring every row to have the exact same set of fields.
    """
    for col in cols:
        if col not in row:
            continue

        val = row.get(col, "")

        if col.endswith("_dag_ok"):
            if is_truthy_string(val):
                return True

        elif col.endswith("_sim_status"):
            s = str(val).strip().lower()
            if s in {"success", "ok", "true", "1", "yes"}:
                return True

        elif col.startswith("hellinger_"):
            try:
                x = float(val)
                if not pd.isna(x):
                    return True
            except Exception:
                pass

        else:
            if not is_empty(val):
                return True

    return False


def compute_drop_and_keep(
    input_fields: List[str],
    base_drop_cols: List[str],
    extra_drop_cols: List[str],
    keep_cols: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Compute the final column-drop result.

    Returns
    -------
    present_drop_cols
        requested drop columns that exist in the input and are not protected
        by keep rules

    missing_drop_cols
        requested drop columns that do not exist in the input

    output_fields
        final ordered output schema

    Rule
    ----
    keep columns always override drop columns
    """
    input_field_set = set(input_fields)
    keep_set = set(keep_cols)

    requested_drop_cols = unique_preserve_order(base_drop_cols + extra_drop_cols)

    present_drop_cols = [
        c for c in requested_drop_cols
        if c in input_field_set and c not in keep_set
    ]
    missing_drop_cols = [c for c in requested_drop_cols if c not in input_field_set]

    output_fields = [c for c in input_fields if c not in set(present_drop_cols)]
    return present_drop_cols, missing_drop_cols, output_fields


# =====================================================================
# DUCKDB / PARQUET HELPERS
# =====================================================================

def sql_escape_string(s: str) -> str:
    """
    Escape single quotes for SQL string embedding.
    """
    return s.replace("'", "''")


def quote_ident(name: str) -> str:
    """
    Quote an SQL identifier safely.

    This is used when selecting user-provided or schema-derived column names
    through DuckDB SQL.
    """
    return '"' + name.replace('"', '""') + '"'


def get_parquet_columns(con: duckdb.DuckDBPyConnection, input_path: str) -> List[str]:
    """
    Read input Parquet schema and return column names.

    This allows the script to:
    - inspect available columns before reading all row data
    - compute keep/drop rules safely
    """
    escaped = sql_escape_string(input_path)
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{escaped}')").fetchall()
    return [r[0] for r in rows]


def get_total_rows(con: duckdb.DuckDBPyConnection, input_path: str) -> int:
    """
    Count total rows in the input Parquet file.
    """
    escaped = sql_escape_string(input_path)
    return int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{escaped}')").fetchone()[0])


def read_parquet_chunk(
    con: duckdb.DuckDBPyConnection,
    input_path: str,
    offset: int,
    limit: int,
    columns: List[str],
) -> pd.DataFrame:
    """
    Read one chunk from the input Parquet using DuckDB.

    Why LIMIT/OFFSET here
    ---------------------
    This stage is designed for large datasets and must avoid loading the entire
    Parquet file into memory. Each chunk is read, filtered, and written before
    the next chunk is loaded.
    """
    escaped = sql_escape_string(input_path)
    select_cols = ", ".join(quote_ident(c) for c in columns)

    sql = f"""
        SELECT {select_cols}
        FROM read_parquet('{escaped}')
        LIMIT {int(limit)} OFFSET {int(offset)}
    """
    return con.execute(sql).fetchdf()


def append_parquet_chunk(
    df: pd.DataFrame,
    output_path: str,
    writer: pq.ParquetWriter | None,
    compression: str = "zstd",
) -> pq.ParquetWriter:
    """
    Append one pandas DataFrame chunk to the output Parquet file.

    Behavior
    --------
    - first non-empty chunk creates the ParquetWriter
    - later chunks are appended with the same writer

    Why this design
    ---------------
    This avoids:
    - producing many temporary .part_* files
    - doing a final full-file merge
    """
    table = pa.Table.from_pandas(df, preserve_index=False)

    if writer is None:
        # A column that is empty in the first chunk comes out as Arrow type
        # null, which no later chunk with values can be written into. Store it
        # as string (stage 10 types every column).
        schema = pa.schema(
            [f.with_type(pa.string()) if pa.types.is_null(f.type) else f
             for f in table.schema],
            metadata=table.schema.metadata,
        )
        writer = pq.ParquetWriter(
            output_path,
            schema,
            compression=compression,
        )

    # Later chunks: a column that is empty in THIS chunk (e.g. no job in it has
    # calibration) comes out as null; cast to the file's schema. A no-op when
    # the types already match.
    writer.write_table(table.cast(writer.schema))
    return writer


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    """
    CLI entry point.

    High-level flow
    ---------------
    1. inspect schema
    2. compute final drop/keep rules
    3. read input Parquet in chunks
    4. optionally filter rows by logical/native availability
    5. write surviving rows directly into one output Parquet
    """
    ap = argparse.ArgumentParser(
        description=(
            "Final cleanup stage: optionally drop heavy/intermediate columns "
            "and remove rows without both logical and native availability."
        )
    )
    ap.add_argument("--input", required=True, help="Input Parquet")
    ap.add_argument("--output", required=True, help="Output Parquet")
    ap.add_argument("--chunksize", type=int, default=10000, help="Rows per chunk")
    ap.add_argument("--progress-every", type=int, default=100000, help="Report progress every N rows")
    ap.add_argument("--jobs", type=int, default=1, help="Accepted for workflow compatibility; not used")

    ap.add_argument("--drop-cols", nargs="*", default=[], help="Additional column names to drop")
    ap.add_argument("--drop-cols-file", default=None, help="Optional text file with one extra column name per line to drop")
    ap.add_argument("--keep-cols", nargs="*", default=[], help="Column names to force-keep even if they appear in a drop list")
    ap.add_argument("--keep-cols-file", default=None, help="Optional text file with one column name per line to force-keep")
    ap.add_argument("--drop-dag-ref-cols", action="store_true", help="Accepted for compatibility; requested keep columns still win")
    ap.add_argument("--no-default-drops", action="store_true", help="Do not apply the built-in default drop list")
    ap.add_argument("--keep-missing-ok", action="store_true", help="Accepted for compatibility; missing columns are already tolerated")

    ap.add_argument(
        "--logical-required-cols",
        nargs="*",
        default=DEFAULT_LOGICAL_REQUIRED_COLS,
        help="Logical-side columns used to decide whether logical data is available"
    )
    ap.add_argument(
        "--native-required-cols",
        nargs="*",
        default=DEFAULT_NATIVE_REQUIRED_COLS,
        help="Native-side columns used to decide whether native data is available"
    )
    ap.add_argument(
        "--disable-row-filter",
        action="store_true",
        help="Disable dropping rows without both logical and native availability"
    )

    ap.add_argument("--threads", type=int, default=0, help="DuckDB worker threads")
    ap.add_argument("--memory-limit", default=None, help="Optional DuckDB memory limit, e.g. 64GB")
    ap.add_argument(
        "--compression",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"],
        default="zstd",
        help="Output Parquet compression codec",
    )

    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input Parquet not found: {args.input}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    # -----------------------------------------------------------------
    # STEP 1: BUILD FINAL DROP / KEEP LISTS
    # -----------------------------------------------------------------
    # Extra drop/keep columns may come from the CLI directly or from text files.
    extra_drop_cols: List[str] = list(args.drop_cols or [])
    if args.drop_cols_file:
        extra_drop_cols.extend(parse_cols_file(args.drop_cols_file))

    keep_cols: List[str] = list(args.keep_cols or [])
    if args.keep_cols_file:
        keep_cols.extend(parse_cols_file(args.keep_cols_file))

    # Keep precedence is important.
    # Protected and requested DAG columns are injected first, then any user
    # keep columns are added.
    keep_cols = unique_preserve_order(
        PROTECTED_KEEP_COLS
        + DEFAULT_DAG_KEEP_COLS
        + REQUESTED_EXTRA_KEEP_COLS
        + keep_cols
    )

    base_drop_cols: List[str] = []
    if not args.no_default_drops:
        base_drop_cols.extend(DEFAULT_DROP_COLS)

    # This is mainly a compatibility flag. The keep list still wins.
    if args.drop_dag_ref_cols:
        base_drop_cols.extend(DEFAULT_DAG_REF_DROP_COLS)

    # -----------------------------------------------------------------
    # STEP 2: INITIALIZE RUNTIME STATE
    # -----------------------------------------------------------------
    total_rows_read = 0
    total_rows_written = 0
    total_rows_dropped_missing_side = 0
    start_time = time.time()

    compression = None if args.compression == "none" else args.compression
    writer: pq.ParquetWriter | None = None
    duck = duckdb.connect()

    try:
        # -------------------------------------------------------------
        # STEP 3: OPTIONAL DUCKDB RUNTIME TUNING
        # -------------------------------------------------------------
        if args.threads and args.threads > 0:
            duck.execute(f"PRAGMA threads={int(args.threads)}")
        if args.memory_limit:
            duck.execute(f"PRAGMA memory_limit='{sql_escape_string(args.memory_limit)}'")

        # -------------------------------------------------------------
        # STEP 4: INSPECT INPUT SCHEMA
        # -------------------------------------------------------------
        input_fields = get_parquet_columns(duck, args.input)
        total_input_rows = get_total_rows(duck, args.input)

        present_drop_cols, missing_drop_cols, output_fields = compute_drop_and_keep(
            input_fields=input_fields,
            base_drop_cols=base_drop_cols,
            extra_drop_cols=extra_drop_cols,
            keep_cols=keep_cols,
        )

        print("[INFO] 09_filter_valid_rows.py started")
        print(f"[INFO] Input Parquet : {args.input}")
        print(f"[INFO] Output Parquet: {args.output}")
        print(f"[INFO] Chunksize     : {args.chunksize}")
        print(f"[INFO] Jobs          : {args.jobs} (accepted for compatibility; stage runs sequentially)")
        print(f"[INFO] Input rows    : {total_input_rows}")
        print(f"[INFO] Compression   : {args.compression}")
        print(f"[INFO] Force-kept columns available in output: {sum(1 for c in keep_cols if c in set(input_fields))}")
        print(f"[INFO] Dropping {len(present_drop_cols)} columns present in input")
        print(f"[INFO] Row filter enabled: {not args.disable_row_filter}")
        print(f"[INFO] Logical availability columns: {', '.join(args.logical_required_cols)}")
        print(f"[INFO] Native availability columns : {', '.join(args.native_required_cols)}")

        if present_drop_cols:
            print("[INFO] Present drop columns:")
            for c in present_drop_cols:
                print(f"  - {c}")

        if missing_drop_cols:
            print(f"[INFO] {len(missing_drop_cols)} requested drop columns were not present in input")
            for c in missing_drop_cols:
                print(f"  - {c}")

        # -------------------------------------------------------------
        # STEP 5: PROCESS INPUT IN CHUNKS
        # -------------------------------------------------------------
        # Each chunk is:
        # - read from input Parquet
        # - converted to row dicts
        # - filtered
        # - slimmed to final output columns
        # - appended to the output Parquet
        n_chunks = (total_input_rows + args.chunksize - 1) // args.chunksize if total_input_rows > 0 else 0

        for chunk_idx in range(n_chunks):
            offset = chunk_idx * args.chunksize
            limit = min(args.chunksize, total_input_rows - offset)

            batch_df = read_parquet_chunk(
                duck,
                args.input,
                offset,
                limit,
                input_fields,
            )

            batch_rows = batch_df.to_dict(orient="records")
            kept_rows = []

            for row in batch_rows:
                total_rows_read += 1

                # -----------------------------------------------------
                # ROW FILTER RULE
                # -----------------------------------------------------
                # Keep row only if:
                #   logical side is available
                #   AND
                #   native side is available
                #
                # This can be disabled with --disable-row-filter if the user
                # wants a pure column-pruning run.
                if not args.disable_row_filter:
                    logical_ok = has_side_available(row, args.logical_required_cols)
                    native_ok = has_side_available(row, args.native_required_cols)

                    if not (logical_ok and native_ok):
                        total_rows_dropped_missing_side += 1
                        continue

                slim_row = {k: row.get(k, None) for k in output_fields}
                kept_rows.append(slim_row)
                total_rows_written += 1

                if args.progress_every > 0 and total_rows_read % args.progress_every == 0:
                    elapsed = time.time() - start_time
                    rate = total_rows_read / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[INFO] rows_read={total_rows_read} "
                        f"rows_written={total_rows_written} "
                        f"rows_dropped_missing_side={total_rows_dropped_missing_side} "
                        f"elapsed={elapsed:.2f}s "
                        f"rate={rate:.2f} rows/s",
                        flush=True,
                    )

            # Write chunk only if at least one row survived.
            if kept_rows:
                out_df = pd.DataFrame(kept_rows, columns=output_fields)
                writer = append_parquet_chunk(
                    df=out_df,
                    output_path=args.output,
                    writer=writer,
                    compression=compression or "zstd",
                )

        # -------------------------------------------------------------
        # STEP 6: HANDLE EMPTY OUTPUT CASE
        # -------------------------------------------------------------
        # If no chunk produced surviving rows, we still write an empty Parquet
        # with the correct schema.
        if writer is None:
            empty_df = pd.DataFrame(columns=output_fields)
            table = pa.Table.from_pandas(empty_df, preserve_index=False)
            pq.write_table(
                table,
                args.output,
                compression=compression or "zstd",
            )

    finally:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass
        duck.close()

    total_time = time.time() - start_time
    out_size = os.path.getsize(args.output) if os.path.exists(args.output) else 0

    print("[DONE] 09_filter_valid_rows.py finished")
    print(f"[DONE] Input Parquet : {args.input}")
    print(f"[DONE] Output Parquet: {args.output}")
    print(f"[DONE] Rows read                  : {total_rows_read}")
    print(f"[DONE] Rows written               : {total_rows_written}")
    print(f"[DONE] Rows dropped missing side  : {total_rows_dropped_missing_side}")
    print(f"[DONE] Columns dropped            : {len(present_drop_cols)}")
    print(f"[DONE] Columns kept               : {len(output_fields)}")
    print(f"[DONE] Output size bytes          : {out_size}")
    print(f"[DONE] Total execution time (s)   : {total_time:.2f}")


if __name__ == "__main__":
    main()