#!/usr/bin/env python3
"""
04_compute_two_hellinger_distances.py
=====================================

PURPOSE
-------
Compute two Hellinger distances for each row in the dataset:

1. hellinger_logical
   Compare:
       result_aligned_logical  vs  ideal_result_logical

2. hellinger_native
   Compare:
       result_aligned_native  vs  ideal_result_native

This script is designed to flow directly from:

    03_verify_result_endianness.py

which normally produces:
- result_aligned_logical
- result_aligned_native
- logical_best_endianness
- native_best_endianness

UPDATED PIPELINE ALIGNMENT
--------------------------
Stage 01 now expands batch jobs BEFORE normalization.
Stage 03 now verifies endianness row-wise on single counts payloads.

Therefore stage 04 now assumes:

- one row = one measured result dict
- one row = one ideal logical counts dict
- one row = one ideal native counts dict

This means:

- list payloads are treated as upstream errors
- singleton list wrappers are still tolerated and collapsed
- Hellinger is computed strictly row-wise: dict vs dict

WHY THIS SCRIPT EXISTS
----------------------
After stage 02, you have:
- ideal_result_logical
- ideal_result_native

After stage 03, you have aligned hardware/device counts in a consistent
bit-order convention for both logical and native comparisons.

This stage computes the final scalar Hellinger targets needed for:
- analysis
- correlation studies
- feature engineering
- ML dataset preparation

IMPORTANT DESIGN CHOICE
-----------------------
This script does NOT decide endianness again.

Preferred behavior:
- for logical comparison, use result_aligned_logical
- for native comparison, use result_aligned_native

Fallback behavior:
- if aligned result columns are missing or empty, fall back to:
    result + logical_best_endianness
    result + native_best_endianness

UPDATED STORAGE / MEMORY DESIGN
-------------------------------
This version avoids the older pattern of:
- writing many parquet .part_* files
- combining them at the end

Instead, it:
1. reads Parquet input in streaming batches
2. processes each chunk independently
3. writes each processed chunk directly into ONE final Parquet file

IMPORTANT NORMALIZATION
-----------------------
This script normalizes bitstring keys by removing spaces between grouped
classical registers, for example:

    "00010001 00000000" -> "0001000100000000"

This keeps stage 04 aligned with stage 03 behavior.

INPUT PARQUET (required columns)
--------------------------------
Required:
- ideal_result_logical
- ideal_result_native

Preferred from stage 03:
- result_aligned_logical
- result_aligned_native

Fallback-supported:
- result
- logical_best_endianness
- native_best_endianness

OUTPUT PARQUET
--------------
The output is the same dataset plus these additional columns:

- hellinger_logical
- hellinger_native
- hellinger_logical_mode_used
- hellinger_native_mode_used
- hellinger_logical_result_source
- hellinger_native_result_source
- hellinger_logical_error
- hellinger_native_error

USAGE
-----
Basic run:
python3 04_compute_two_hellinger_distances.py \
  --input /path/to/stage03_verified.parquet \
  --output /path/to/stage04_with_hellinger.parquet

Chunked run:
python3 04_compute_two_hellinger_distances.py \
  --input /path/to/stage03_verified.parquet \
  --output /path/to/stage04_with_hellinger.parquet \
  --chunksize 20000

Chunked run with progress:
python3 04_compute_two_hellinger_distances.py \
  --input /path/to/stage03_verified.parquet \
  --output /path/to/stage04_with_hellinger.parquet \
  --chunksize 20000 \
  --progress-every 1000

Parallel run:
python3 04_compute_two_hellinger_distances.py \
  --input /path/to/stage03_verified.parquet \
  --output /path/to/stage04_with_hellinger.parquet \
  --jobs 4

Parallel chunked run:
python3 04_compute_two_hellinger_distances.py \
  --input /path/to/stage03_verified.parquet \
  --output /path/to/stage04_with_hellinger.parquet \
  --chunksize 20000 \
  --progress-every 1000 \
  --jobs 4

Small test run:
python3 04_compute_two_hellinger_distances.py \
  --input /path/to/stage03_verified.parquet \
  --output /path/to/stage04_test.parquet \
  --max-rows 100 \
  --progress-every 20

Example for your workflow:
python3 04_compute_two_hellinger_distances.py \
  --input /path/to/stage03_verified.parquet \
  --output /path/to/stage04_with_hellinger.parquet \
  --chunksize 20000 \
  --progress-every 1000 \
  --jobs 48 \
  --compression zstd

Restricted-column run:
python3 04_compute_two_hellinger_distances.py \
  --input /path/to/stage03_verified.parquet \
  --output /path/to/stage04_with_hellinger.parquet \
  --usecols ideal_result_logical,ideal_result_native,result_aligned_logical,result_aligned_native,logical_best_endianness,native_best_endianness,result
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import time
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


Counts = Dict[str, int]


# =====================================================================
# BASIC HELPERS
# =====================================================================

def _is_empty(cell: Any) -> bool:
    """
    Return True if a cell should be treated as empty.

    This normalizes several common empty forms.
    """
    if cell is None:
        return True
    if isinstance(cell, float) and np.isnan(cell):
        return True

    s = str(cell).strip()
    return s == "" or s.lower() == "nan"


def _safe_json_loads(s: str) -> Tuple[bool, Any]:
    """
    Safe JSON parser that never raises.
    """
    try:
        return True, json.loads(s)
    except Exception:
        return False, None


def _safe_literal_eval(s: str) -> Tuple[bool, Any]:
    """
    Safe Python literal parser that never raises.

    This is useful when cells contain Python dict/list strings instead
    of strict JSON.
    """
    try:
        return True, ast.literal_eval(s)
    except Exception:
        return False, None


def _loads_json_or_literal(s: str) -> Tuple[bool, Any]:
    """
    Try JSON first, then Python literal parsing.
    """
    ok, obj = _safe_json_loads(s)
    if ok:
        return True, obj
    ok, obj = _safe_literal_eval(s)
    if ok:
        return True, obj
    return False, None


def _coerce_singleton_list_payload(obj: Any) -> Any:
    """
    Collapse a singleton list into its only item.

    This allows harmless wrappers such as:
        [{"00": 10, "11": 5}]
    to behave like a single counts dictionary.
    """
    if isinstance(obj, list) and len(obj) == 1:
        return obj[0]
    return obj


def _unwrap_json_repeatedly(s: str, max_depth: int = 6) -> Any:
    """
    Repeatedly unwrap string-encoded JSON or Python-literal payloads.

    Handles repeated encodings such as:
        "\"{\\\"00\\\": 10}\""
    """
    obj: Any = s

    for _ in range(max_depth):
        if not isinstance(obj, str):
            return obj

        t = obj.strip()
        if not t:
            return ""

        if t[0] in ['"', "[", "{"]:
            ok, parsed = _loads_json_or_literal(t)
            if ok:
                obj = parsed
                continue

        return t

    return obj


def parse_csv_list_arg(raw: Optional[str]) -> Optional[List[str]]:
    """
    Parse comma-separated CLI column lists into Python lists.
    """
    if raw is None or str(raw).strip() == "":
        return None
    return [x.strip() for x in str(raw).split(",") if x.strip()]


# =====================================================================
# COUNTS PARSING
# =====================================================================

def _normalize_counts_keys(counts: Counts) -> Counts:
    """
    Normalize bitstring keys by removing spaces between grouped registers.

    Example:
        "00010001 00000000" -> "0001000100000000"
    """
    out: Counts = {}
    for k, v in counts.items():
        nk = str(k).replace(" ", "")
        out[nk] = out.get(nk, 0) + int(v)
    return out


def _parse_counts_dict(obj: Dict[Any, Any]) -> Counts:
    """
    Normalize a generic dict-like counts object into str->int and normalize keys.
    """
    raw = {str(k): int(v) for k, v in obj.items()}
    return _normalize_counts_keys(raw)


def _parse_single_counts_cell(cell: Any, label: str) -> Counts:
    """
    Parse one counts cell into a SINGLE counts dictionary.

    This stage assumes stage 01 already expanded batch jobs.
    Therefore list[dict] payloads are treated as upstream errors,
    except singleton wrappers which are collapsed.
    """
    if _is_empty(cell):
        raise ValueError(f"Empty/NaN {label} cell")

    raw = str(cell).strip()
    obj = _unwrap_json_repeatedly(raw)
    obj = _coerce_singleton_list_payload(obj)

    if isinstance(obj, str):
        ok, parsed = _loads_json_or_literal(obj)
        if ok:
            obj = parsed
        else:
            raise ValueError(f"{label} string could not be decoded")

    if isinstance(obj, dict):
        return _parse_counts_dict(obj)

    if isinstance(obj, list):
        raise ValueError(f"{label} still contains list payload after stage-01 expansion")

    raise ValueError(f"{label} must decode to dict")


# =====================================================================
# HELLINGER UTILITIES
# =====================================================================

def _reverse_bitstrings(counts: Counts) -> Counts:
    """
    Reverse bitstrings in one counts dictionary.

    Example:
        "0011" -> "1100"
    """
    out: Counts = {}
    for k, v in counts.items():
        rk = k[::-1]
        out[rk] = out.get(rk, 0) + int(v)
    return out


def _normalize(counts: Counts) -> Dict[str, float]:
    """
    Convert a counts dictionary into a probability dictionary.
    """
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in counts.items()}


def _hellinger(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Compute Hellinger distance between two probability dictionaries.
    """
    keys = set(p) | set(q)
    if not keys:
        return float("nan")

    p_vec = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    q_vec = np.array([q.get(k, 0.0) for k in keys], dtype=float)

    return float(np.sqrt(0.5 * np.sum((np.sqrt(p_vec) - np.sqrt(q_vec)) ** 2)))


def _distance_pair(result_counts: Counts, ideal_counts: Counts) -> float:
    """
    Compute Hellinger distance for one dict-vs-dict pair.
    """
    return _hellinger(_normalize(result_counts), _normalize(ideal_counts))


# =====================================================================
# MODE / RESULT-SOURCE HELPERS
# =====================================================================

def _sanitize_mode(mode_value: Any) -> str:
    """
    Normalize endianness mode to either:
    - normal
    - reversed

    Empty values default to normal.
    """
    if _is_empty(mode_value):
        return "normal"

    mode = str(mode_value).strip().lower()

    if mode in {"normal", "reversed"}:
        return mode

    raise ValueError(f"Invalid endianness mode: {mode_value}")


def _apply_mode_to_counts_cell(cell: Counts, mode: str) -> Counts:
    """
    Apply a chosen endianness mode to a parsed counts payload.
    """
    if mode == "normal":
        return cell
    if mode == "reversed":
        return _reverse_bitstrings(cell)
    raise ValueError(f"Unsupported mode: {mode}")


def _choose_result_source(
    row: Dict[str, Any],
    aligned_col: str,
    fallback_mode_col: str,
) -> Tuple[Any, str, str]:
    """
    Choose the result payload to use for one target.

    Preference
    ----------
    1. aligned result column from stage 03
    2. raw result + endianness mode column

    Returns
    -------
    payload, result_source_description, mode_used_description
    """
    if aligned_col in row and not _is_empty(row[aligned_col]):
        return row[aligned_col], f"aligned:{aligned_col}", "aligned"

    if "result" not in row:
        raise ValueError(
            f"Missing aligned result column '{aligned_col}' and fallback column 'result' is not present"
        )

    if fallback_mode_col not in row:
        raise ValueError(
            f"Missing aligned result column '{aligned_col}' and fallback mode column '{fallback_mode_col}' is not present"
        )

    raw_result = _parse_single_counts_cell(row["result"], "result")
    mode = _sanitize_mode(row[fallback_mode_col])
    aligned_result = _apply_mode_to_counts_cell(raw_result, mode)

    return (
        json.dumps(aligned_result, ensure_ascii=False, sort_keys=True),
        f"fallback:result+{fallback_mode_col}",
        f"fallback:{mode}",
    )


# =====================================================================
# ROW PROCESSING
# =====================================================================

def _compute_row_from_dict(
    row: Dict[str, Any],
    ideal_logical_col: str,
    ideal_native_col: str,
    logical_aligned_result_col: str,
    native_aligned_result_col: str,
    logical_mode_col: str,
    native_mode_col: str,
) -> Dict[str, Any]:
    """
    Compute both logical and native Hellinger values for one row.

    Each side is handled independently, so one side may succeed even if the
    other side fails.
    """
    outputs: Dict[str, Any] = {
        "hellinger_logical": np.nan,
        "hellinger_native": np.nan,
        "hellinger_logical_mode_used": "",
        "hellinger_native_mode_used": "",
        "hellinger_logical_result_source": "",
        "hellinger_native_result_source": "",
        "hellinger_logical_error": "",
        "hellinger_native_error": "",
    }

    # ------------------------------------------------------------
    # Logical comparison
    # ------------------------------------------------------------
    try:
        logical_result_payload, logical_result_source, logical_mode_used = _choose_result_source(
            row=row,
            aligned_col=logical_aligned_result_col,
            fallback_mode_col=logical_mode_col,
        )
        result_logical = _parse_single_counts_cell(logical_result_payload, logical_aligned_result_col)
        ideal_logical = _parse_single_counts_cell(row[ideal_logical_col], ideal_logical_col)

        outputs["hellinger_logical"] = _distance_pair(
            result_counts=result_logical,
            ideal_counts=ideal_logical,
        )
        outputs["hellinger_logical_mode_used"] = logical_mode_used
        outputs["hellinger_logical_result_source"] = logical_result_source
    except Exception as exc:
        outputs["hellinger_logical_error"] = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------
    # Native comparison
    # ------------------------------------------------------------
    try:
        native_result_payload, native_result_source, native_mode_used = _choose_result_source(
            row=row,
            aligned_col=native_aligned_result_col,
            fallback_mode_col=native_mode_col,
        )
        result_native = _parse_single_counts_cell(native_result_payload, native_aligned_result_col)
        ideal_native = _parse_single_counts_cell(row[ideal_native_col], ideal_native_col)

        outputs["hellinger_native"] = _distance_pair(
            result_counts=result_native,
            ideal_counts=ideal_native,
        )
        outputs["hellinger_native_mode_used"] = native_mode_used
        outputs["hellinger_native_result_source"] = native_result_source
    except Exception as exc:
        outputs["hellinger_native_error"] = f"{type(exc).__name__}: {exc}"

    return outputs


def _compute_row_worker(args: Tuple[Dict[str, Any], str, str, str, str, str, str]) -> Dict[str, Any]:
    """
    Multiprocessing worker wrapper.
    """
    row, ideal_logical_col, ideal_native_col, logical_aligned_result_col, native_aligned_result_col, logical_mode_col, native_mode_col = args
    return _compute_row_from_dict(
        row=row,
        ideal_logical_col=ideal_logical_col,
        ideal_native_col=ideal_native_col,
        logical_aligned_result_col=logical_aligned_result_col,
        native_aligned_result_col=native_aligned_result_col,
        logical_mode_col=logical_mode_col,
        native_mode_col=native_mode_col,
    )


def _pool_chunksize(n_rows: int, jobs: int) -> int:
    """
    Choose a reasonable multiprocessing chunksize.
    """
    if n_rows <= 0:
        return 1
    approx = n_rows // max(1, jobs * 4)
    return max(1, min(128, approx if approx > 0 else 1))


# =====================================================================
# DATAFRAME PROCESSING
# =====================================================================

def _process_dataframe(
    df: pd.DataFrame,
    ideal_logical_col: str,
    ideal_native_col: str,
    logical_aligned_result_col: str,
    native_aligned_result_col: str,
    logical_mode_col: str,
    native_mode_col: str,
    progress_every: int,
    label: str,
    jobs: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Process one dataframe batch from start to finish.

    Steps
    -----
    1. validate required columns
    2. compute row-wise logical/native Hellinger values
    3. append new columns to the dataframe
    4. return the dataframe and summary stats
    """
    required_cols = [ideal_logical_col, ideal_native_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    total = len(df)
    start = time.time()
    print(f"[INFO] Starting Hellinger computation for {label}: {total} row(s)", flush=True)

    row_dicts = df.to_dict(orient="records")
    worker_args = [
        (
            row,
            ideal_logical_col,
            ideal_native_col,
            logical_aligned_result_col,
            native_aligned_result_col,
            logical_mode_col,
            native_mode_col,
        )
        for row in row_dicts
    ]

    # ------------------------------------------------------------
    # Sequential mode
    # ------------------------------------------------------------
    if jobs <= 1:
        results = []
        for pos, args_row in enumerate(worker_args, start=1):
            results.append(_compute_row_worker(args_row))
            if progress_every > 0 and (pos % progress_every == 0 or pos == total):
                elapsed = time.time() - start
                rate = pos / elapsed if elapsed > 0 else 0.0
                logical_ok = sum(1 for r in results if r["hellinger_logical_error"] == "")
                native_ok = sum(1 for r in results if r["hellinger_native_error"] == "")
                print(
                    f"[INFO] {label}: processed {pos}/{total} rows | "
                    f"logical ok={logical_ok} | native ok={native_ok} | "
                    f"elapsed={elapsed:.2f}s | rate={rate:.2f} rows/s",
                    flush=True,
                )

    # ------------------------------------------------------------
    # Parallel mode
    # ------------------------------------------------------------
    else:
        print(f"[INFO] Using {jobs} worker process(es)", flush=True)
        with Pool(processes=jobs) as pool:
            results = []
            for pos, out in enumerate(
                pool.imap(_compute_row_worker, worker_args, chunksize=_pool_chunksize(total, jobs)),
                start=1,
            ):
                results.append(out)
                if progress_every > 0 and (pos % progress_every == 0 or pos == total):
                    elapsed = time.time() - start
                    rate = pos / elapsed if elapsed > 0 else 0.0
                    logical_ok = sum(1 for r in results if r["hellinger_logical_error"] == "")
                    native_ok = sum(1 for r in results if r["hellinger_native_error"] == "")
                    print(
                        f"[INFO] {label}: processed {pos}/{total} rows | "
                        f"logical ok={logical_ok} | native ok={native_ok} | "
                        f"elapsed={elapsed:.2f}s | rate={rate:.2f} rows/s",
                        flush=True,
                    )

    results_df = pd.DataFrame(results)
    out_df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

    logical_ok = int((out_df["hellinger_logical_error"] == "").sum())
    native_ok = int((out_df["hellinger_native_error"] == "").sum())
    logical_fail = total - logical_ok
    native_fail = total - native_ok

    duration = time.time() - start

    stats = {
        "rows": total,
        "logical_ok": logical_ok,
        "native_ok": native_ok,
        "logical_fail": logical_fail,
        "native_fail": native_fail,
        "duration_sec": duration,
    }

    print(
        f"[INFO] Completed {label} | "
        f"rows={total} | logical ok={logical_ok}/{total} | "
        f"native ok={native_ok}/{total} | duration={duration:.2f}s",
        flush=True,
    )

    return out_df, stats


# =====================================================================
# PARQUET STREAMING WRITER
# =====================================================================

class SafeParquetChunkWriter:
    """
    Append chunked pandas DataFrames into ONE final Parquet file.

    This avoids temporary part files and avoids a final Parquet merge.
    """

    def __init__(self, output_path: str, compression: str = "zstd") -> None:
        self.output_path = output_path
        self.compression = compression
        self.writer: Optional[pq.ParquetWriter] = None
        self.schema: Optional[pa.Schema] = None
        self.columns: Optional[List[str]] = None

    def _align_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align later chunks to the schema established by the first chunk.
        """
        assert self.columns is not None
        for col in self.columns:
            if col not in df.columns:
                df[col] = None
        return df[self.columns].copy()

    def write(self, df: pd.DataFrame) -> None:
        """
        Write one processed dataframe chunk into the final Parquet file.
        """
        if df is None or df.empty:
            return

        if self.writer is None:
            table = pa.Table.from_pandas(df, preserve_index=False)
            self.schema = table.schema
            self.columns = list(df.columns)
            self.writer = pq.ParquetWriter(
                self.output_path,
                self.schema,
                compression=self.compression,
            )
            self.writer.write_table(table)
            return

        df2 = self._align_df(df)
        table = pa.Table.from_pandas(df2, schema=self.schema, preserve_index=False)
        self.writer.write_table(table)

    def close(self) -> None:
        """
        Close the underlying Parquet writer safely.
        """
        if self.writer is not None:
            self.writer.close()
            self.writer = None


# =====================================================================
# PARQUET INPUT HELPERS
# =====================================================================

def get_input_columns(input_path: str) -> List[str]:
    """
    Read Parquet schema column names without loading full data.
    """
    pf = pq.ParquetFile(input_path)
    return pf.schema_arrow.names


def get_total_rows(input_path: str) -> int:
    """
    Read total row count from Parquet metadata.
    """
    pf = pq.ParquetFile(input_path)
    return pf.metadata.num_rows


def iter_parquet_batches(
    input_path: str,
    batch_size: int,
    columns: Optional[List[str]] = None,
) -> Iterable[pd.DataFrame]:
    """
    Stream Parquet input in batches.

    This keeps memory usage stable for large datasets.
    """
    pf = pq.ParquetFile(input_path)

    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    """
    CLI entry point.

    This function:
    1. validates arguments
    2. checks the input schema
    3. chooses chunked or all-at-once processing
    4. computes row-wise logical/native Hellinger values
    5. writes the final Parquet output
    6. prints a final summary report
    """
    parser = argparse.ArgumentParser(
        description="Compute logical and native Hellinger distances using aligned row-wise result counts."
    )
    parser.add_argument("--input", required=True, help="Input Parquet file")
    parser.add_argument("--output", required=True, help="Output Parquet file")

    parser.add_argument(
        "--ideal-logical-col",
        default="ideal_result_logical",
        help="Column containing logical ideal counts.",
    )
    parser.add_argument(
        "--ideal-native-col",
        default="ideal_result_native",
        help="Column containing native ideal counts.",
    )
    parser.add_argument(
        "--logical-aligned-result-col",
        default="result_aligned_logical",
        help="Preferred aligned result column for logical comparison.",
    )
    parser.add_argument(
        "--native-aligned-result-col",
        default="result_aligned_native",
        help="Preferred aligned result column for native comparison.",
    )
    parser.add_argument(
        "--logical-mode-col",
        default="logical_best_endianness",
        help="Fallback mode column for logical comparison if aligned result column is absent or empty.",
    )
    parser.add_argument(
        "--native-mode-col",
        default="native_best_endianness",
        help="Fallback mode column for native comparison if aligned result column is absent or empty.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for testing on the first N rows.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=0,
        help="0 = process all rows at once",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N rows. Use 0 to disable periodic progress output.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes. Use 1 for sequential mode.",
    )
    parser.add_argument(
        "--usecols",
        default=None,
        help="Optional comma-separated input columns to load. Must include required columns.",
    )
    parser.add_argument(
        "--compression",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"],
        default="zstd",
        help="Parquet compression codec.",
    )
    args = parser.parse_args()

    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    detected_cpus = cpu_count()
    if args.jobs > detected_cpus:
        print(f"[INFO] Requested jobs ({args.jobs}) exceed detected CPU count ({detected_cpus}).", flush=True)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    requested_usecols = parse_csv_list_arg(args.usecols)

    required_cols = {
        args.ideal_logical_col,
        args.ideal_native_col,
    }

    if requested_usecols is not None:
        missing_required = required_cols - set(requested_usecols)
        if missing_required:
            raise SystemExit(
                f"--usecols is missing required column(s): {sorted(missing_required)}"
            )

    input_cols = get_input_columns(args.input)
    missing = [c for c in required_cols if c not in input_cols]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    total_input_rows = get_total_rows(args.input)
    effective_total_rows = min(total_input_rows, args.max_rows) if args.max_rows is not None else total_input_rows

    overall_start = time.time()
    total_rows_processed = 0
    total_logical_ok = 0
    total_native_ok = 0
    chunk_count = 0

    compression = args.compression
    writer: Optional[SafeParquetChunkWriter] = None

    print("[INFO] Hellinger computation started", flush=True)
    print(f"[INFO] Input file:                  {args.input}", flush=True)
    print(f"[INFO] Output file:                 {args.output}", flush=True)
    print(f"[INFO] Logical ideal col:           {args.ideal_logical_col}", flush=True)
    print(f"[INFO] Native ideal col:            {args.ideal_native_col}", flush=True)
    print(f"[INFO] Logical aligned result col:  {args.logical_aligned_result_col}", flush=True)
    print(f"[INFO] Native aligned result col:   {args.native_aligned_result_col}", flush=True)
    print(f"[INFO] Logical fallback mode col:   {args.logical_mode_col}", flush=True)
    print(f"[INFO] Native fallback mode col:    {args.native_mode_col}", flush=True)
    print(f"[INFO] Chunksize:                   {args.chunksize if args.chunksize > 0 else 'all-at-once'}", flush=True)
    print(f"[INFO] Max rows:                    {args.max_rows if args.max_rows is not None else 'all'}", flush=True)
    print(f"[INFO] Progress every:              {args.progress_every}", flush=True)
    print(f"[INFO] Worker processes (--jobs):   {args.jobs}", flush=True)
    print(f"[INFO] Compression:                 {args.compression}", flush=True)
    print(f"[INFO] Input rows:                  {total_input_rows}", flush=True)
    print(f"[INFO] Rows to process:             {effective_total_rows}", flush=True)

    try:
        # ---------------------------------------------------------
        # Chunked / streaming mode
        # ---------------------------------------------------------
        if args.chunksize and args.chunksize > 0:
            writer = SafeParquetChunkWriter(args.output, compression=compression)
            n_chunks = int(math.ceil(effective_total_rows / args.chunksize)) if effective_total_rows > 0 else 0

            rows_seen = 0
            for chunk_idx, chunk in enumerate(
                iter_parquet_batches(args.input, batch_size=args.chunksize, columns=requested_usecols),
                start=1,
            ):
                if args.max_rows is not None:
                    remaining = effective_total_rows - rows_seen
                    if remaining <= 0:
                        break
                    if len(chunk) > remaining:
                        chunk = chunk.head(remaining).copy()

                chunk_count += 1
                label = f"chunk {chunk_count}"

                out, stats = _process_dataframe(
                    df=chunk.copy(),
                    ideal_logical_col=args.ideal_logical_col,
                    ideal_native_col=args.ideal_native_col,
                    logical_aligned_result_col=args.logical_aligned_result_col,
                    native_aligned_result_col=args.native_aligned_result_col,
                    logical_mode_col=args.logical_mode_col,
                    native_mode_col=args.native_mode_col,
                    progress_every=args.progress_every,
                    label=label,
                    jobs=args.jobs,
                )

                writer.write(out)

                rows_seen += len(chunk)
                total_rows_processed += stats["rows"]
                total_logical_ok += stats["logical_ok"]
                total_native_ok += stats["native_ok"]

                print(
                    f"[INFO] Written {label} to output | "
                    f"input rows read so far={rows_seen}/{effective_total_rows} | "
                    f"cumulative rows={total_rows_processed} | "
                    f"cumulative logical ok={total_logical_ok} | "
                    f"cumulative native ok={total_native_ok}",
                    flush=True,
                )

        # ---------------------------------------------------------
        # All-at-once mode
        # ---------------------------------------------------------
        else:
            df = next(
                iter_parquet_batches(
                    args.input,
                    batch_size=max(effective_total_rows, 1),
                    columns=requested_usecols,
                )
            )

            out, stats = _process_dataframe(
                df=df,
                ideal_logical_col=args.ideal_logical_col,
                ideal_native_col=args.ideal_native_col,
                logical_aligned_result_col=args.logical_aligned_result_col,
                native_aligned_result_col=args.native_aligned_result_col,
                logical_mode_col=args.logical_mode_col,
                native_mode_col=args.native_mode_col,
                progress_every=args.progress_every,
                label="full dataset",
                jobs=args.jobs,
            )

            out.to_parquet(
                args.output,
                index=False,
                compression=None if compression == "none" else compression,
            )

            total_rows_processed = stats["rows"]
            total_logical_ok = stats["logical_ok"]
            total_native_ok = stats["native_ok"]
            chunk_count = 1

        total_duration = time.time() - overall_start
        logical_fail = total_rows_processed - total_logical_ok
        native_fail = total_rows_processed - total_native_ok
        out_size = os.path.getsize(args.output) if os.path.exists(args.output) else 0

        print("\n========== FINAL REPORT ==========", flush=True)
        print(f"Saved output to:                 {args.output}", flush=True)
        print(f"Input file:                      {args.input}", flush=True)
        print(f"Rows processed:                  {total_rows_processed}", flush=True)
        print(f"Output size bytes:               {out_size}", flush=True)
        print(f"Chunks processed:                {chunk_count if args.chunksize and args.chunksize > 0 else 1}", flush=True)
        print(f"Chunksize used:                  {args.chunksize if args.chunksize > 0 else 'all-at-once'}", flush=True)
        print(f"Progress interval:               {args.progress_every}", flush=True)
        print(f"Worker processes used:           {args.jobs}", flush=True)

        print("\n--- LOGICAL HELLINGER SUMMARY ---", flush=True)
        print(f"Logical Hellinger computed:      {total_logical_ok}/{total_rows_processed}", flush=True)
        print(f"Logical Hellinger failed:        {logical_fail}/{total_rows_processed}", flush=True)

        print("\n--- NATIVE HELLINGER SUMMARY ---", flush=True)
        print(f"Native Hellinger computed:       {total_native_ok}/{total_rows_processed}", flush=True)
        print(f"Native Hellinger failed:         {native_fail}/{total_rows_processed}", flush=True)

        if total_duration > 0:
            print("\n--- PERFORMANCE SUMMARY ---", flush=True)
            print(f"Total execution time (s):        {total_duration:.2f}", flush=True)
            print(f"Average rows/sec:                {total_rows_processed / total_duration:.2f}", flush=True)
            if args.chunksize and args.chunksize > 0 and chunk_count > 0:
                print(f"Average sec/chunk:               {total_duration / chunk_count:.2f}", flush=True)

        print("==================================", flush=True)

    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()