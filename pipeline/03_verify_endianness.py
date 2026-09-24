#!/usr/bin/env python3
"""
03_verify_result_endianness.py
==============================

PURPOSE
-------
Determine the best bit ordering for comparing the measured hardware/device
result against the ideal logical and ideal native counts.

This stage compares:

1. result vs ideal_result_logical
2. result vs ideal_result_native

under two possible interpretations of the measured bitstrings:

- normal bit order
- reversed bit order

The script chooses the better orientation using L1 distance on normalized
probability distributions.

UPDATED PIPELINE ALIGNMENT
--------------------------
Stage 01 now expands batch jobs BEFORE normalization.

Therefore this stage now assumes:

- one row = one measured result dict
- one row = one ideal logical counts dict
- one row = one ideal native counts dict

So:

- list payloads are treated as upstream errors
- singleton list wrappers are still tolerated and collapsed
- result alignment is strictly row-wise

WHY THIS STAGE EXISTS
---------------------
Qiskit-based ideal counts and hardware/device counts may use different
display/bit-order conventions. If this is not corrected, downstream distance
metrics such as Hellinger distance can become misleading.

This stage therefore:

1. checks which bit ordering matches better for each row
2. records both raw and final endianness decisions
3. creates aligned result columns for direct downstream use

UPDATED STORAGE / MEMORY DESIGN
-------------------------------
This version avoids:
- writing many parquet .part_* files
- combining them at the end

Instead, it:

1. reads Parquet input in streaming batches
2. processes each chunk independently
3. writes each processed chunk directly into ONE final Parquet file

RAW UNCERTAINTY
---------------
If the difference between the normal-order L1 distance and the reversed-order
L1 distance is too small:

    abs(l1_normal - l1_reversed) < epsilon

then the raw decision is marked as:

    uncertain

UNCERTAIN ROW HANDLING
----------------------
After raw decisions are computed:

1. the script determines the dominant confident mode across the processed batch
2. uncertain rows are resolved using fallback logic

Fallback priority per target:
- use the same target's raw confident mode if available
- otherwise use the other target's raw confident mode from the same row
- otherwise use the dataset-level dominant confident mode
- if no dominant mode exists, use --default-fallback-mode

IMPORTANT NOTE ABOUT CHUNKED MODE
---------------------------------
In chunked mode, dominant-mode fallback is computed per processed chunk,
not globally across the whole dataset.

OUTPUT COLUMNS ADDED
--------------------
Logical:
- logical_verify_status
- logical_verify_error
- logical_best_endianness_raw
- logical_best_endianness
- logical_endianness_confident
- logical_endianness_used_fallback
- result_aligned_logical

Native:
- native_verify_status
- native_verify_error
- native_best_endianness_raw
- native_best_endianness
- native_endianness_confident
- native_endianness_used_fallback
- result_aligned_native

REQUIRED INPUT COLUMNS
----------------------
At minimum, the input Parquet must contain:

- result                     (or another column passed via --result-col)
- ideal_result_logical
- ideal_result_native

COMMON WORKFLOW POSITION
------------------------
Typical pipeline order:

    01_parse_and_normalize_both_circuits.py
    02_simulate_ideal_result.py
    03_verify_result_endianness.py
    04_compute_two_hellinger_distances.py

USAGE
-----
Basic run:
python3 03_verify_result_endianness.py \
  --input /path/to/stage02_with_ideal.parquet \
  --output /path/to/stage03_verified.parquet

Run with progress reporting:
python3 03_verify_result_endianness.py \
  --input /path/to/stage02_with_ideal.parquet \
  --output /path/to/stage03_verified.parquet \
  --progress-every 1000

Run with a smaller uncertainty threshold:
python3 03_verify_result_endianness.py \
  --input /path/to/stage02_with_ideal.parquet \
  --output /path/to/stage03_verified.parquet \
  --uncertainty-epsilon 0.001

Chunked run for large files:
python3 03_verify_result_endianness.py \
  --input /path/to/stage02_with_ideal.parquet \
  --output /path/to/stage03_verified.parquet \
  --chunksize 20000 \
  --progress-every 1000

All-at-once run on a small dataset:
python3 03_verify_result_endianness.py \
  --input /path/to/stage02_with_ideal.parquet \
  --output /path/to/stage03_verified.parquet \
  --chunksize 0

Example for your workflow:
python3 03_verify_result_endianness.py \
  --input /path/to/stage02_with_ideal.parquet \
  --output /path/to/stage03_verified.parquet \
  --chunksize 10000 \
  --progress-every 1000 \
  --uncertainty-epsilon 0.001 \
  --default-fallback-mode normal \
  --compression zstd

Run with an alternate measured-result column:
python3 03_verify_result_endianness.py \
  --input /path/to/input.parquet \
  --output /path/to/output.parquet \
  --result-col result_expanded

Run while loading only needed columns:
python3 03_verify_result_endianness.py \
  --input /path/to/stage02_with_ideal.parquet \
  --output /path/to/stage03_verified.parquet \
  --usecols id,result,ideal_result_logical,ideal_result_native
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# A single row in this stage is expected to contain exactly one counts dict.
Counts = Dict[str, int]


# =====================================================================
# BASIC HELPERS
# =====================================================================

def _is_empty(cell: Any) -> bool:
    """
    Return True if a value should be treated as empty.

    This centralizes the empty-value rule used throughout the script.
    """
    if cell is None:
        return True
    if isinstance(cell, float) and np.isnan(cell):
        return True
    s = str(cell).strip()
    return s == "" or s.lower() == "nan"


def _safe_json_loads(s: str) -> Tuple[bool, Any]:
    """
    Safe JSON parser.

    Returns
    -------
    (ok, obj)
        ok=True  -> parsing succeeded
        ok=False -> parsing failed and obj is None
    """
    try:
        return True, json.loads(s)
    except Exception:
        return False, None


def _safe_literal_eval(s: str) -> Tuple[bool, Any]:
    """
    Safe Python-literal parser.

    This helps with payloads that are Python dict/list strings rather than
    strict JSON.
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
    Collapse a singleton list wrapper into its only element.

    This tolerates upstream payloads such as:
        [{"00": 5, "11": 7}]
    when they are clearly just wrappers around one row-level payload.
    """
    if isinstance(obj, list) and len(obj) == 1:
        return obj[0]
    return obj


def _unwrap_json_repeatedly(s: str, max_depth: int = 6) -> Any:
    """
    Repeatedly unwrap nested string-encoded JSON / literal payloads.

    This is useful for cases like:
        "\"{\\\"00\\\": 10}\""

    Parameters
    ----------
    s : str
        Input string to decode.
    max_depth : int
        Maximum number of repeated decoding attempts.

    Returns
    -------
    Any
        Decoded object, or the original stripped string if decoding stops.
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
    Parse a comma-separated CLI string into a Python list.

    Example
    -------
    "id,result,ideal_result_logical" ->
        ["id", "result", "ideal_result_logical"]
    """
    if raw is None or str(raw).strip() == "":
        return None
    return [x.strip() for x in str(raw).split(",") if x.strip()]


# =====================================================================
# COUNTS PARSING AND NORMALIZATION
# =====================================================================

def _normalize_counts_keys(counts: Counts) -> Counts:
    """
    Normalize bitstring keys by removing spaces between register groups.

    Example
    -------
    "00010001 00000000" -> "0001000100000000"

    This keeps counts compatible even if different stages or systems format
    grouped classical registers differently.
    """
    out: Counts = {}
    for k, v in counts.items():
        nk = str(k).replace(" ", "")
        out[nk] = out.get(nk, 0) + int(v)
    return out


def _parse_counts_dict(obj: Dict[Any, Any]) -> Counts:
    """
    Convert a generic dict-like counts object into normalized str->int counts.
    """
    raw = {str(k): int(v) for k, v in obj.items()}
    return _normalize_counts_keys(raw)


def _parse_single_counts_cell(cell: Any, label: str) -> Counts:
    """
    Parse one counts cell into a SINGLE counts dictionary.

    This stage assumes stage 01 already expanded batch jobs.
    Therefore list[dict] payloads are treated as upstream errors, except
    singleton wrappers which are collapsed.

    Parameters
    ----------
    cell : Any
        Raw cell value from the dataframe.
    label : str
        Human-readable label used in error messages.

    Returns
    -------
    Counts
        Parsed row-level counts dictionary.
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
        raise ValueError(
            f"{label} still contains list payload after stage-01 expansion"
        )

    raise ValueError(f"{label} must decode to dict")


def _counts_to_json(cell: Counts) -> str:
    """
    Serialize a parsed counts dict back to stable JSON.

    The stable ordering makes outputs easier to compare and debug.
    """
    return json.dumps(cell, ensure_ascii=False, sort_keys=True)


# =====================================================================
# PROBABILITY HELPERS
# =====================================================================

def _reverse_bitstrings(counts: Counts) -> Counts:
    """
    Reverse every bitstring key in a counts dictionary.

    Example
    -------
    "0110" -> "0110"[::-1] -> "0110"
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


def _l1_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Compute L1 distance between two probability dictionaries.

    L1 distance is used here only to decide which bit ordering is closer.
    """
    keys = set(p) | set(q)
    if not keys:
        return float("nan")
    return float(sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys))


# =====================================================================
# ENDIANNESS COMPARISON
# =====================================================================

def _compare_pair_with_confidence(
    result_counts: Counts,
    ideal_counts: Counts,
    epsilon: float,
) -> Tuple[str, bool]:
    """
    Compare one measured counts dict against one ideal counts dict.

    The measured counts are evaluated under two interpretations:
    - normal bit order
    - reversed bit order

    The decision is based on which orientation gives the smaller L1 distance
    to the ideal distribution.

    Parameters
    ----------
    result_counts : Counts
        Measured hardware/device counts.
    ideal_counts : Counts
        Ideal logical/native counts for the same row.
    epsilon : float
        Minimum distance gap required for a confident decision.

    Returns
    -------
    (best_mode, confident)
        best_mode is one of:
        - "normal"
        - "reversed"
        - "uncertain"
    """
    p_normal = _normalize(result_counts)
    p_reversed = _normalize(_reverse_bitstrings(result_counts))
    q_ideal = _normalize(ideal_counts)

    l1_normal = _l1_distance(p_normal, q_ideal)
    l1_reversed = _l1_distance(p_reversed, q_ideal)
    l1_gap = abs(l1_normal - l1_reversed)

    if np.isnan(l1_gap) or l1_gap < epsilon:
        return "uncertain", False

    best_mode = "normal" if l1_normal <= l1_reversed else "reversed"
    return best_mode, True


# =====================================================================
# ALIGNMENT HELPERS
# =====================================================================

def _aligned_result_json_from_mode(result_cell: Counts, mode: str) -> str:
    """
    Build the aligned measured-result JSON using the chosen final mode.
    """
    if mode == "normal":
        return _counts_to_json(result_cell)
    if mode == "reversed":
        return _counts_to_json(_reverse_bitstrings(result_cell))
    raise ValueError(f"Unsupported final mode for alignment: {mode}")


def _dominant_mode_from_series(series: pd.Series) -> str | None:
    """
    Return the dominant confident mode from a pandas Series.

    Only "normal" and "reversed" values are considered.
    """
    vals = [
        str(x).strip().lower()
        for x in series.tolist()
        if str(x).strip().lower() in {"normal", "reversed"}
    ]
    if not vals:
        return None

    counts = Counter(vals)
    return "normal" if counts["normal"] >= counts["reversed"] else "reversed"


def _resolve_mode(
    own_raw: str,
    other_raw: str,
    dominant_mode: str | None,
    default_fallback_mode: str,
    assume_mode: str | None = None,
) -> Tuple[str, bool]:
    """
    Resolve the final endianness mode after raw row-level decisions exist.

    With `assume_mode` set, that mode is used for every row and the detector's
    verdict is recorded but not acted on. This is the correct setting whenever
    the platform's bit order is known: the devices in this dataset were confirmed
    on hardware (2026-09-24) to return Qiskit's little-endian convention, bit n-1
    leftmost and bit 0 rightmost, so the mode is `normal` and detection can only
    do harm. It does: on low-fidelity circuits the measured distribution is close
    to noise, the two alignments land within a hair of each other, and an epsilon
    of 0.001 is tight enough that one wins "confidently" by chance. On a 10,000
    circuit audit that mislabelled 10% of rows, with a median error in the
    resulting Hellinger distance of 0.0064 and a tail reaching 0.65.

    Without `assume_mode`, the historical fallback order applies:
    1. use this target's own confident raw mode
    2. otherwise use the other target's confident raw mode in the same row
    3. otherwise use the dominant confident mode of the current batch
    4. otherwise use the configured default fallback mode

    Step 3 makes a row's value depend on which other rows share its chunk, which
    is a second reason to prefer `assume_mode`.

    Returns
    -------
    (final_mode, used_fallback)
    """
    own_raw = str(own_raw).strip().lower()
    other_raw = str(other_raw).strip().lower()

    if assume_mode in {"normal", "reversed"}:
        # `used_fallback` stays False: nothing was guessed, the convention is known.
        return assume_mode, False

    if own_raw in {"normal", "reversed"}:
        return own_raw, False

    if other_raw in {"normal", "reversed"}:
        return other_raw, True

    if dominant_mode in {"normal", "reversed"}:
        return dominant_mode, True

    return default_fallback_mode, True


# =====================================================================
# ROW PROCESSING
# =====================================================================

def _process_row(row: pd.Series, epsilon: float, result_col: str) -> Dict[str, Any]:
    """
    Compute raw row-level endianness decisions for both logical and native.

    This function only computes the raw decisions.
    Final fallback resolution is done later at dataframe/chunk level.
    """
    out = {
        "logical_verify_status": "FAILED",
        "logical_verify_error": "",
        "logical_best_endianness_raw": "",
        "logical_best_endianness": "",
        "logical_endianness_confident": False,
        "logical_endianness_used_fallback": False,
        "result_aligned_logical": "",

        "native_verify_status": "FAILED",
        "native_verify_error": "",
        "native_best_endianness_raw": "",
        "native_best_endianness": "",
        "native_endianness_confident": False,
        "native_endianness_used_fallback": False,
        "result_aligned_native": "",
    }

    # -------------------------------------------------------------
    # Parse the measured result once.
    # If this fails, both logical and native verification fail.
    # -------------------------------------------------------------
    try:
        result_cell = _parse_single_counts_cell(row[result_col], result_col)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        out["logical_verify_error"] = msg
        out["native_verify_error"] = msg
        return out

    # -------------------------------------------------------------
    # Compare measured result against ideal logical counts.
    # -------------------------------------------------------------
    try:
        ideal_logical = _parse_single_counts_cell(
            row["ideal_result_logical"],
            "ideal_result_logical",
        )
        best_mode, confident = _compare_pair_with_confidence(
            result_cell,
            ideal_logical,
            epsilon,
        )
        out["logical_best_endianness_raw"] = best_mode
        out["logical_endianness_confident"] = confident
        out["logical_verify_status"] = "SUCCESS"
    except Exception as exc:
        out["logical_verify_error"] = f"{type(exc).__name__}: {exc}"

    # -------------------------------------------------------------
    # Compare measured result against ideal native counts.
    # -------------------------------------------------------------
    try:
        ideal_native = _parse_single_counts_cell(
            row["ideal_result_native"],
            "ideal_result_native",
        )
        best_mode, confident = _compare_pair_with_confidence(
            result_cell,
            ideal_native,
            epsilon,
        )
        out["native_best_endianness_raw"] = best_mode
        out["native_endianness_confident"] = confident
        out["native_verify_status"] = "SUCCESS"
    except Exception as exc:
        out["native_verify_error"] = f"{type(exc).__name__}: {exc}"

    return out


def _resolve_dataframe_modes(
    df: pd.DataFrame,
    default_fallback_mode: str,
    assume_mode: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Resolve final endianness modes after raw per-row decisions are computed.

    This function applies the fallback policy to uncertain rows and records
    whether fallback was needed.
    """
    logical_dominant = _dominant_mode_from_series(df["logical_best_endianness_raw"])
    native_dominant = _dominant_mode_from_series(df["native_best_endianness_raw"])

    overall_vals = []
    overall_vals.extend(
        str(x).strip().lower()
        for x in df["logical_best_endianness_raw"].tolist()
        if str(x).strip().lower() in {"normal", "reversed"}
    )
    overall_vals.extend(
        str(x).strip().lower()
        for x in df["native_best_endianness_raw"].tolist()
        if str(x).strip().lower() in {"normal", "reversed"}
    )

    overall_dominant = None
    if overall_vals:
        counts = Counter(overall_vals)
        overall_dominant = "normal" if counts["normal"] >= counts["reversed"] else "reversed"

    logical_fallback_pool = logical_dominant if logical_dominant else overall_dominant
    native_fallback_pool = native_dominant if native_dominant else overall_dominant

    final_logical = []
    logical_used_fallback = []
    final_native = []
    native_used_fallback = []

    for row in df.itertuples(index=False):
        row_d = row._asdict()

        logical_mode, logical_fb = _resolve_mode(
            own_raw=row_d["logical_best_endianness_raw"],
            other_raw=row_d["native_best_endianness_raw"],
            dominant_mode=logical_fallback_pool,
            default_fallback_mode=default_fallback_mode,
            assume_mode=assume_mode,
        )
        native_mode, native_fb = _resolve_mode(
            own_raw=row_d["native_best_endianness_raw"],
            other_raw=row_d["logical_best_endianness_raw"],
            dominant_mode=native_fallback_pool,
            default_fallback_mode=default_fallback_mode,
            assume_mode=assume_mode,
        )

        final_logical.append(logical_mode)
        logical_used_fallback.append(logical_fb)
        final_native.append(native_mode)
        native_used_fallback.append(native_fb)

    df["logical_best_endianness"] = final_logical
    df["logical_endianness_used_fallback"] = logical_used_fallback
    df["native_best_endianness"] = final_native
    df["native_endianness_used_fallback"] = native_used_fallback

    disagreed_logical = int(sum(
        1 for raw, fin in zip(df["logical_best_endianness_raw"], final_logical)
        if str(raw).strip().lower() in {"normal", "reversed"}
        and str(raw).strip().lower() != str(fin).strip().lower()))
    disagreed_native = int(sum(
        1 for raw, fin in zip(df["native_best_endianness_raw"], final_native)
        if str(raw).strip().lower() in {"normal", "reversed"}
        and str(raw).strip().lower() != str(fin).strip().lower()))

    stats = {
        "assume_mode": assume_mode or "",
        "logical_detector_overridden": disagreed_logical,
        "native_detector_overridden": disagreed_native,
        "logical_dominant_mode": logical_dominant or "",
        "native_dominant_mode": native_dominant or "",
        "overall_dominant_mode": overall_dominant or "",
        "logical_fallback_count": int(pd.Series(logical_used_fallback).sum()),
        "native_fallback_count": int(pd.Series(native_used_fallback).sum()),
    }
    return df, stats


def _build_aligned_result_columns(df: pd.DataFrame, result_col: str) -> pd.DataFrame:
    """
    Build final aligned measured-result columns using the resolved final modes.
    """
    aligned_logical = []
    aligned_native = []

    for _, row in df.iterrows():
        try:
            result_cell = _parse_single_counts_cell(row[result_col], result_col)
            aligned_logical.append(
                _aligned_result_json_from_mode(
                    result_cell,
                    str(row["logical_best_endianness"]).strip().lower(),
                )
            )
        except Exception:
            aligned_logical.append("")

        try:
            result_cell = _parse_single_counts_cell(row[result_col], result_col)
            aligned_native.append(
                _aligned_result_json_from_mode(
                    result_cell,
                    str(row["native_best_endianness"]).strip().lower(),
                )
            )
        except Exception:
            aligned_native.append("")

    df["result_aligned_logical"] = aligned_logical
    df["result_aligned_native"] = aligned_native
    return df


# =====================================================================
# DATAFRAME / CHUNK PROCESSING
# =====================================================================

def _process_dataframe(
    df: pd.DataFrame,
    progress_every: int,
    label: str,
    epsilon: float,
    default_fallback_mode: str,
    result_col: str,
    assume_mode: str | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Process one dataframe batch from start to finish.

    Steps
    -----
    1. verify required columns exist
    2. compute raw row-wise decisions
    3. resolve final modes using chunk-level fallback logic
    4. build aligned result columns
    5. return processed dataframe and summary stats
    """
    required_cols = [result_col, "ideal_result_logical", "ideal_result_native"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    raw_rows = []
    total = len(df)
    logical_ok = 0
    native_ok = 0
    logical_failed = 0
    native_failed = 0
    logical_uncertain_raw = 0
    native_uncertain_raw = 0

    start = time.time()
    print(f"[INFO] Starting verification for {label}: {total} row(s)", flush=True)

    for pos, (_, row) in enumerate(df.iterrows(), start=1):
        out = _process_row(row, epsilon=epsilon, result_col=result_col)
        raw_rows.append(out)

        if out["logical_verify_status"] == "SUCCESS":
            logical_ok += 1
            if out["logical_best_endianness_raw"] == "uncertain":
                logical_uncertain_raw += 1
        else:
            logical_failed += 1

        if out["native_verify_status"] == "SUCCESS":
            native_ok += 1
            if out["native_best_endianness_raw"] == "uncertain":
                native_uncertain_raw += 1
        else:
            native_failed += 1

        if progress_every > 0 and (pos % progress_every == 0 or pos == total):
            elapsed = time.time() - start
            rate = pos / elapsed if elapsed > 0 else 0.0
            print(
                f"[INFO] {label}: processed {pos}/{total} rows | "
                f"logical ok={logical_ok} | native ok={native_ok} | "
                f"logical uncertain(raw)={logical_uncertain_raw} | "
                f"native uncertain(raw)={native_uncertain_raw} | "
                f"elapsed={elapsed:.2f}s | rate={rate:.2f} rows/s",
                flush=True,
            )

    out_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(raw_rows)], axis=1)
    out_df, resolve_stats = _resolve_dataframe_modes(
        out_df,
        default_fallback_mode=default_fallback_mode,
        assume_mode=assume_mode,
    )
    out_df = _build_aligned_result_columns(out_df, result_col=result_col)

    logical_unresolved_final = int((out_df["logical_best_endianness"] == "").sum())
    native_unresolved_final = int((out_df["native_best_endianness"] == "").sum())
    duration = time.time() - start

    stats = {
        "rows": total,
        "logical_ok": logical_ok,
        "native_ok": native_ok,
        "logical_failed": logical_failed,
        "native_failed": native_failed,
        "logical_uncertain_raw": logical_uncertain_raw,
        "native_uncertain_raw": native_uncertain_raw,
        "logical_unresolved_final": logical_unresolved_final,
        "native_unresolved_final": native_unresolved_final,
        "duration_sec": duration,
        **resolve_stats,
    }

    print(
        f"[INFO] Completed {label} | "
        f"rows={total} | logical ok={logical_ok}/{total} | "
        f"native ok={native_ok}/{total} | "
        f"logical uncertain(raw)={logical_uncertain_raw} | "
        f"native uncertain(raw)={native_uncertain_raw} | "
        f"logical unresolved(final)={logical_unresolved_final} | "
        f"native unresolved(final)={native_unresolved_final} | "
        f"duration={duration:.2f}s",
        flush=True,
    )

    return out_df, stats


# =====================================================================
# PARQUET STREAMING WRITER
# =====================================================================

class SafeParquetChunkWriter:
    """
    Append chunked pandas DataFrames into ONE final Parquet file.

    Why this helper exists
    ----------------------
    Later chunks must match the schema established by the first chunk.
    This class keeps that stable and avoids generating many temporary files.
    """

    def __init__(self, output_path: str, compression: str = "zstd") -> None:
        self.output_path = output_path
        self.compression = compression
        self.writer: Optional[pq.ParquetWriter] = None
        self.schema: Optional[pa.Schema] = None
        self.columns: Optional[List[str]] = None

    def _align_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align later chunks to the first-chunk schema.

        Missing columns are added as nulls, and column order is enforced.
        """
        assert self.columns is not None
        for col in self.columns:
            if col not in df.columns:
                df[col] = None
        return df[self.columns].copy()

    def write(self, df: pd.DataFrame) -> None:
        """
        Write one processed dataframe chunk.
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

    Parameters
    ----------
    input_path : str
        Input Parquet file path.
    batch_size : int
        Rows per batch.
    columns : Optional[List[str]]
        Restrict reading to these columns only.

    Yields
    ------
    pd.DataFrame
        One dataframe batch at a time.
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
    2. checks input schema
    3. processes the dataset in chunked or all-at-once mode
    4. writes the final verified Parquet
    5. prints a summary report
    """
    parser = argparse.ArgumentParser(
        description="Fast endianness verification using row-wise result vs ideal counts only."
    )
    parser.add_argument("--input", required=True, help="Input Parquet file")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional limit for testing")
    parser.add_argument("--chunksize", type=int, default=0, help="0 = process all rows at once")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N rows. Use 0 to disable periodic progress output.",
    )
    parser.add_argument(
        "--uncertainty-epsilon",
        type=float,
        default=0.001,
        help="If abs(l1_normal - l1_reversed) < epsilon, mark the raw endianness decision as uncertain.",
    )
    parser.add_argument(
        "--default-fallback-mode",
        choices=["normal", "reversed"],
        default="normal",
        help="Final fallback mode if no dominant confident mode exists.",
    )
    parser.add_argument(
        "--assume-mode",
        choices=["normal", "reversed"],
        default=None,
        help="Align every row with this bit order and ignore the detector's "
             "verdict (still recorded in *_best_endianness_raw). Use it whenever "
             "the platform's convention is known: these devices were confirmed on "
             "hardware to use Qiskit little-endian, so --assume-mode normal is "
             "correct. Leaving it unset re-enables per-row detection, which "
             "mislabels ~10%% of rows on low-fidelity circuits.",
    )
    parser.add_argument(
        "--result-col",
        default="result",
        help="Measured result column. Default: result",
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

    if args.uncertainty_epsilon < 0:
        raise ValueError("--uncertainty-epsilon must be >= 0")

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    requested_usecols = parse_csv_list_arg(args.usecols)
    required_cols = {args.result_col, "ideal_result_logical", "ideal_result_native"}

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
    effective_total_rows = (
        min(total_input_rows, args.max_rows)
        if args.max_rows is not None
        else total_input_rows
    )

    overall_start = time.time()

    total_rows_processed = 0
    total_logical_ok = 0
    total_native_ok = 0
    total_logical_uncertain_raw = 0
    total_native_uncertain_raw = 0
    total_logical_unresolved_final = 0
    total_native_unresolved_final = 0
    total_logical_fallback_count = 0
    total_native_fallback_count = 0
    chunk_count = 0

    logical_dominant_modes = []
    native_dominant_modes = []
    overall_dominant_modes = []

    compression = args.compression
    writer: Optional[SafeParquetChunkWriter] = None

    try:
        print("[INFO] Fast endianness verification started", flush=True)
        print(f"[INFO] Input file:               {args.input}", flush=True)
        print(f"[INFO] Output file:              {args.output}", flush=True)
        print(f"[INFO] Input rows:               {total_input_rows}", flush=True)
        print(f"[INFO] Rows to process:          {effective_total_rows}", flush=True)
        print(f"[INFO] Chunksize:                {args.chunksize if args.chunksize > 0 else 'all-at-once'}", flush=True)
        print(f"[INFO] Progress every:           {args.progress_every}", flush=True)
        print(f"[INFO] Uncertainty epsilon:      {args.uncertainty_epsilon}", flush=True)
        print(f"[INFO] Default fallback mode:    {args.default_fallback_mode}", flush=True)
        print(f"[INFO] Assume mode:              {args.assume_mode or '<detect per row>'}", flush=True)
        print(f"[INFO] Result column:            {args.result_col}", flush=True)
        print(f"[INFO] Compression:              {args.compression}", flush=True)

        # ---------------------------------------------------------
        # Chunked / streaming mode
        # ---------------------------------------------------------
        if args.chunksize and args.chunksize > 0:
            writer = SafeParquetChunkWriter(args.output, compression=compression)
            n_chunks = int(math.ceil(effective_total_rows / args.chunksize)) if effective_total_rows > 0 else 0

            rows_seen = 0
            for chunk_idx, chunk in enumerate(
                iter_parquet_batches(
                    args.input,
                    batch_size=args.chunksize,
                    columns=requested_usecols,
                ),
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
                    progress_every=args.progress_every,
                    label=label,
                    epsilon=args.uncertainty_epsilon,
                    default_fallback_mode=args.default_fallback_mode,
                    assume_mode=args.assume_mode,
                    result_col=args.result_col,
                )

                writer.write(out)

                rows_seen += len(chunk)
                total_rows_processed += stats["rows"]
                total_logical_ok += stats["logical_ok"]
                total_native_ok += stats["native_ok"]
                total_logical_uncertain_raw += stats["logical_uncertain_raw"]
                total_native_uncertain_raw += stats["native_uncertain_raw"]
                total_logical_unresolved_final += stats["logical_unresolved_final"]
                total_native_unresolved_final += stats["native_unresolved_final"]
                total_logical_fallback_count += stats["logical_fallback_count"]
                total_native_fallback_count += stats["native_fallback_count"]

                if stats["logical_dominant_mode"]:
                    logical_dominant_modes.append(stats["logical_dominant_mode"])
                if stats["native_dominant_mode"]:
                    native_dominant_modes.append(stats["native_dominant_mode"])
                if stats["overall_dominant_mode"]:
                    overall_dominant_modes.append(stats["overall_dominant_mode"])

                print(
                    f"[INFO] Written {label} to output | "
                    f"input rows read so far={rows_seen}/{effective_total_rows} | "
                    f"cumulative rows={total_rows_processed} | "
                    f"cumulative logical uncertain(raw)={total_logical_uncertain_raw} | "
                    f"cumulative native uncertain(raw)={total_native_uncertain_raw}",
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
                progress_every=args.progress_every,
                label="full dataset",
                epsilon=args.uncertainty_epsilon,
                default_fallback_mode=args.default_fallback_mode,
                assume_mode=args.assume_mode,
                result_col=args.result_col,
            )

            out.to_parquet(
                args.output,
                index=False,
                compression=None if compression == "none" else compression,
            )

            total_rows_processed = stats["rows"]
            total_logical_ok = stats["logical_ok"]
            total_native_ok = stats["native_ok"]
            total_logical_uncertain_raw = stats["logical_uncertain_raw"]
            total_native_uncertain_raw = stats["native_uncertain_raw"]
            total_logical_unresolved_final = stats["logical_unresolved_final"]
            total_native_unresolved_final = stats["native_unresolved_final"]
            total_logical_fallback_count = stats["logical_fallback_count"]
            total_native_fallback_count = stats["native_fallback_count"]
            chunk_count = 1

            if stats["logical_dominant_mode"]:
                logical_dominant_modes.append(stats["logical_dominant_mode"])
            if stats["native_dominant_mode"]:
                native_dominant_modes.append(stats["native_dominant_mode"])
            if stats["overall_dominant_mode"]:
                overall_dominant_modes.append(stats["overall_dominant_mode"])

        total_duration = time.time() - overall_start
        logical_fail = total_rows_processed - total_logical_ok
        native_fail = total_rows_processed - total_native_ok

        logical_confident_raw = total_logical_ok - total_logical_uncertain_raw
        native_confident_raw = total_native_ok - total_native_uncertain_raw

        final_logical_dominant = _dominant_mode_from_series(pd.Series(logical_dominant_modes))
        final_native_dominant = _dominant_mode_from_series(pd.Series(native_dominant_modes))
        final_overall_dominant = _dominant_mode_from_series(pd.Series(overall_dominant_modes))

        out_size = os.path.getsize(args.output) if os.path.exists(args.output) else 0

        print("\n========== FINAL REPORT ==========", flush=True)
        print(f"Saved output to:                 {args.output}", flush=True)
        print(f"Input file:                      {args.input}", flush=True)
        print(f"Rows processed:                  {total_rows_processed}", flush=True)
        print(f"Output size bytes:               {out_size}", flush=True)
        print(f"Chunks processed:                {chunk_count if args.chunksize and args.chunksize > 0 else 1}", flush=True)
        print(f"Chunksize used:                  {args.chunksize if args.chunksize > 0 else 'all-at-once'}", flush=True)
        print(f"Progress interval:               {args.progress_every}", flush=True)
        print(f"Uncertainty epsilon:             {args.uncertainty_epsilon}", flush=True)
        print(f"Default fallback mode:           {args.default_fallback_mode}", flush=True)
        print(f"Dominant logical confident mode: {final_logical_dominant or 'N/A'}", flush=True)
        print(f"Dominant native confident mode:  {final_native_dominant or 'N/A'}", flush=True)
        print(f"Dominant overall confident mode: {final_overall_dominant or 'N/A'}", flush=True)

        print("\n--- LOGICAL ENDIANNESS SUMMARY ---", flush=True)
        print(f"Logical verified:                {total_logical_ok}/{total_rows_processed}", flush=True)
        print(f"Logical failed:                  {logical_fail}/{total_rows_processed}", flush=True)
        print(f"Logical confident (raw):         {logical_confident_raw}/{total_rows_processed}", flush=True)
        print(f"Logical uncertain (raw):         {total_logical_uncertain_raw}/{total_rows_processed}", flush=True)
        print(f"Logical fallback used:           {total_logical_fallback_count}/{total_rows_processed}", flush=True)
        print(f"Logical unresolved (final):      {total_logical_unresolved_final}/{total_rows_processed}", flush=True)

        print("\n--- NATIVE ENDIANNESS SUMMARY ---", flush=True)
        print(f"Native verified:                 {total_native_ok}/{total_rows_processed}", flush=True)
        print(f"Native failed:                   {native_fail}/{total_rows_processed}", flush=True)
        print(f"Native confident (raw):          {native_confident_raw}/{total_rows_processed}", flush=True)
        print(f"Native uncertain (raw):          {total_native_uncertain_raw}/{total_rows_processed}", flush=True)
        print(f"Native fallback used:            {total_native_fallback_count}/{total_rows_processed}", flush=True)
        print(f"Native unresolved (final):       {total_native_unresolved_final}/{total_rows_processed}", flush=True)

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