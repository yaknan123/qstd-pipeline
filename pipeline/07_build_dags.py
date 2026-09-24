#!/usr/bin/env python3
"""
07_add_dags.py
==============

PURPOSE
-------
Build DAGs for normalized logical and native circuits, store compact DAG metrics
in the output Parquet, and persist a compact sidecar representation in SQLite.

UPDATED WORKFLOW ALIGNMENT
--------------------------
This version is aligned with the updated stage 01 and downstream stages.

In the updated workflow:
- stage 01 expands batch jobs BEFORE normalization
- each row should represent one sub-job / one circuit instance
- stage 02, 03, 04, and 06 operate row-wise
- list-like circuit payloads at this stage are treated as upstream errors

It now prefers per-row circuit sources in this order:

Logical:
1. circuit_best_for_sim
2. circuit_qasm2_norm
3. circuit

Native:
1. executed_circuit_best_for_sim
2. executed_circuit_qasm2_norm
3. executed_circuit_qasm2_norm_stripped
4. executed_circuit

It also supports:
- OpenQASM 2.0
- OpenQASM 3.0
- QASM2 custom instructions such as rxx / ryy / rzz / rzx / ecr

BEST-PRACTICE IMPROVEMENTS IN THIS VERSION
------------------------------------------
1. Parse-aware row-wise fallback:
   - logical source is chosen per row from logical candidates
   - native source is chosen per row from native candidates
   - selection is based on first successfully parsed/buildable DAG, not merely first non-empty text

2. Reduced multiprocessing payload:
   - workers receive only the small set of required columns

3. Worker-local memoization:
   - repeated circuit texts in the same worker are reused without reparsing/rebuilding

4. Better diagnostics:
   - source column used
   - number of attempts made
   - whether fallback was needed
   - normalized error category
   - row_drop_reason

5. Better SQLite write efficiency:
   - INSERT OR IGNORE with executemany in chunks

6. Safer invalid-row handling:
   --invalid-row-policy keep
   --invalid-row-policy drop_any_fail
   --invalid-row-policy drop_both_fail

7. Safer Parquet output:
   - no .part_* files
   - no end-of-run Parquet merge
   - output is written directly into one final Parquet file chunk by chunk

WHAT THIS SCRIPT ADDS
---------------------
For each row, this script adds two DAG result blocks:

Logical DAG block:
- logical_dag_source_col
- logical_dag_attempts
- logical_dag_used_fallback
- logical_dag_ok
- logical_dag_error_category
- logical_dag_error
- logical_dag_ref
- logical_dag_hash
- logical_dag_blob_bytes
- logical_dag_depth
- logical_dag_num_qubits
- logical_dag_num_clbits
- logical_dag_op_nodes
- logical_dag_twoq_ops
- logical_dag_measure_ops
- logical_dag_layers
- logical_dag_gate_histogram

Native DAG block:
- native_dag_source_col
- native_dag_attempts
- native_dag_used_fallback
- native_dag_ok
- native_dag_error_category
- native_dag_error
- native_dag_ref
- native_dag_hash
- native_dag_blob_bytes
- native_dag_depth
- native_dag_num_qubits
- native_dag_num_clbits
- native_dag_op_nodes
- native_dag_twoq_ops
- native_dag_measure_ops
- native_dag_layers
- native_dag_gate_histogram

It also adds:
- row_drop_reason

SQLITE SIDECAR DATABASE
-----------------------
This script stores compressed QPY blobs in SQLite so that the output Parquet
does not need to carry full reconstructed DAG payloads.

The SQLite table stores:
- dag_ref
- circuit_hash
- kind
- encoding
- qpy_gzip
- nbytes
- created_at

The Parquet file keeps only the compact references and summary metrics.

INVALID ROW POLICY
------------------
Rows can be handled in three ways:

1. keep
   Keep all rows even if logical/native DAG build fails.

2. drop_any_fail
   Drop row if logical OR native DAG construction fails.

3. drop_both_fail
   Drop row only if BOTH logical and native DAG construction fail.

COLUMN DROP OPTIONS
-------------------
You may optionally remove heavy columns from the OUTPUT Parquet:

--drop-result-cols
    Drop heavy result-related columns from the output dataset.

--drop-circuit-cols
    Drop heavy raw/normalized circuit text columns from the output dataset
    after DAG extraction succeeds.

IMPORTANT
---------
Protected identity columns are never dropped automatically, including:
- id
- parent_id
- sub_id
- batch_index
- batch_size
- is_batch_job
- executed_resource
- timestamp_scheduled
- timestamp_scheduled_utc
- timestamp_completed
- timestamp_completed_utc
- window_start_utc
- window_end_utc

USAGE
-----
Basic run:
python3 07_add_dags.py \
  --input /path/to/stage06_with_features.parquet \
  --output /path/to/stage07_with_dags.parquet \
  --dag-db /path/to/stage07_dag_storage.sqlite

Debug-first run:
python3 07_add_dags.py \
  --input /path/to/input.parquet \
  --output /path/to/output.parquet \
  --dag-db /path/to/dag_storage.sqlite \
  --jobs 16 \
  --chunksize 5000 \
  --progress-every 500 \
  --invalid-row-policy keep

Stricter run:
python3 07_add_dags.py \
  --input /path/to/stage06_with_features.parquet \
  --output /path/to/stage07_with_dags.parquet \
  --dag-db /path/to/stage07_dag_storage.sqlite \
  --jobs 24 \
  --chunksize 5000 \
  --progress-every 500 \
  --invalid-row-policy drop_any_fail

Keep all rows but drop heavy columns from output:
python3 07_add_dags.py \
  --input /path/to/stage06_with_features.parquet \
  --output /path/to/stage07_with_dags.parquet \
  --dag-db /path/to/stage07_dag_storage.sqlite \
  --jobs 24 \
  --chunksize 5000 \
  --progress-every 500 \
  --invalid-row-policy keep \
  --drop-result-cols \
  --drop-circuit-cols

Example for your workflow:
python3 07_add_dags.py \
  --input /path/to/stage06_with_features.parquet \
  --output /path/to/stage07_with_dags.parquet \
  --dag-db /path/to/stage07_dag_storage.sqlite \
  --jobs 24 \
  --chunksize 5000 \
  --progress-every 500 \
  --invalid-row-policy keep \
  --drop-result-cols \
  --drop-circuit-cols \
  --compression zstd
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import sqlite3
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# =====================================================================
# QISKIT IMPORTS
# =====================================================================
# qpy:
#   Used to serialize reconstructed circuits into a compact binary format
#   before storing them in SQLite.
#
# qasm2 / qasm3:
#   Used to parse row-level circuit text into QuantumCircuit objects.
#
# circuit_to_dag / dag_to_circuit:
#   Used to build DAGs and later convert them back to circuits for QPY storage.
from qiskit import qpy, qasm2
from qiskit.converters import circuit_to_dag, dag_to_circuit

try:
    from qiskit.qasm3 import loads as loads_qasm3
except Exception:
    loads_qasm3 = None

# =====================================================================
# OPTIONAL QASM2 CUSTOM-GATE SUPPORT
# =====================================================================
# To keep parsing aligned with earlier workflow stages, register commonly
# encountered custom instructions whenever the environment provides them.
_OPTIONAL_GATE_CLASSES: Dict[str, Any] = {}


def _try_import_gate(module_path: str, class_name: str) -> Optional[Any]:
    """
    Try to import a gate class safely.

    This keeps the script portable across environments with slightly different
    Qiskit installations.
    """
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception:
        return None


for _mod, _cls in [
    ("qiskit.circuit.library", "RXXGate"),
    ("qiskit.circuit.library", "RYYGate"),
    ("qiskit.circuit.library", "RZZGate"),
    ("qiskit.circuit.library", "RZXGate"),
    ("qiskit.circuit.library", "ECRGate"),
    ("qiskit.circuit.library", "iSwapGate"),
    ("qiskit.circuit.library", "DCXGate"),
]:
    gate_cls = _try_import_gate(_mod, _cls)
    if gate_cls is not None:
        _OPTIONAL_GATE_CLASSES[_cls] = gate_cls


def _build_qasm2_custom_instructions() -> Tuple[Any, ...]:
    """
    Build a broad OpenQASM 2 custom-instruction registry.

    This keeps DAG parsing aligned with the behavior already used in
    stages 01, 02, and 06.
    """
    customs = list(getattr(qasm2, "LEGACY_CUSTOM_INSTRUCTIONS", []))

    seen = set()
    for inst in customs:
        seen.add((inst.name, inst.num_params, inst.num_qubits))

    wanted_specs: List[Tuple[str, int, int, Optional[Any]]] = [
        ("rxx", 1, 2, _OPTIONAL_GATE_CLASSES.get("RXXGate")),
        ("ryy", 1, 2, _OPTIONAL_GATE_CLASSES.get("RYYGate")),
        ("rzz", 1, 2, _OPTIONAL_GATE_CLASSES.get("RZZGate")),
        ("rzx", 1, 2, _OPTIONAL_GATE_CLASSES.get("RZXGate")),
        ("ecr", 0, 2, _OPTIONAL_GATE_CLASSES.get("ECRGate")),
        ("iswap", 0, 2, _OPTIONAL_GATE_CLASSES.get("iSwapGate")),
        ("dcx", 0, 2, _OPTIONAL_GATE_CLASSES.get("DCXGate")),
    ]

    for name, num_params, num_qubits, ctor in wanted_specs:
        if ctor is None:
            continue
        key = (name, num_params, num_qubits)
        if key not in seen:
            customs.append(
                qasm2.CustomInstruction(
                    name=name,
                    num_params=num_params,
                    num_qubits=num_qubits,
                    constructor=ctor,
                    builtin=True,
                )
            )
            seen.add(key)

    return tuple(customs)


QASM2_CUSTOM_INSTRUCTIONS = _build_qasm2_custom_instructions()

# =====================================================================
# DEFAULT COLUMN-DROP POLICIES
# =====================================================================
# These sets define columns that can optionally be removed from the OUTPUT
# Parquet after DAG extraction has completed.

DEFAULT_RESULT_COLUMNS = {
    "result",
    "result_expanded",
    "result_parse_error",
    "ideal_result_logical",
    "ideal_result_native",
    "ideal_result_logical_error",
    "ideal_result_native_error",
    "logical_endianness_counts_used",
    "native_endianness_counts_used",
    "result_aligned_logical",
    "result_aligned_native",
}

DEFAULT_CIRCUIT_COLUMNS = {
    "circuit",
    "executed_circuit",
    "circuit_qasm2_norm",
    "executed_circuit_qasm2_norm",
    "executed_circuit_qasm2_norm_stripped",
    "circuit_best_for_sim",
    "executed_circuit_best_for_sim",
    "circuit_norm_text",
    "executed_circuit_norm_text",
    "circuit_parse_error",
    "executed_circuit_parse_error",
    "circuit_kind",
    "executed_circuit_kind",
    "circuit_n",
    "executed_circuit_n",
}

# Columns that should remain available for downstream merges and identity tracking.
# timestamp_completed / timestamp_completed_utc are protected because the upstream
# sensor extractor anchors its aggregation window on timestamp_completed.
PROTECTED_IDENTITY_COLUMNS = {
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
}

# Default row-wise circuit source preference.
DEFAULT_LOGICAL_CANDIDATES = (
    "circuit_best_for_sim",
    "circuit_qasm2_norm",
    "circuit",
)

DEFAULT_NATIVE_CANDIDATES = (
    "executed_circuit_best_for_sim",
    "executed_circuit_qasm2_norm",
    "executed_circuit_qasm2_norm_stripped",
    "executed_circuit",
)

# Compact metric names added for each DAG.
DAG_METRIC_NAMES = (
    "dag_depth",
    "dag_num_qubits",
    "dag_num_clbits",
    "dag_op_nodes",
    "dag_twoq_ops",
    "dag_measure_ops",
    "dag_layers",
    "dag_gate_histogram",
)

# =====================================================================
# WORKER-LOCAL MEMOIZATION CACHE
# =====================================================================
# Cache key:
#   (kind_prefix, circuit_hash)
#
# Cache value:
#   (
#       update_dict_for_output_columns,
#       optional_blob_record_for_sqlite
#   )
#
# This avoids repeated parse/build work for duplicate circuits seen by the
# same worker process.
_WORKER_CACHE: Dict[
    Tuple[str, str],
    Tuple[Dict[str, str], Optional[Tuple[str, str, str, bytes]]]
] = {}

# =====================================================================
# BASIC HELPERS
# =====================================================================

def parse_csv_list_arg(raw: Optional[str]) -> List[str]:
    """
    Parse a comma-separated CLI string into a Python list.

    Examples
    --------
    "a,b,c" -> ["a", "b", "c"]
    ""      -> []
    None    -> []
    """
    if raw is None:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def sha256_text(text: str) -> str:
    """
    Compute a stable SHA-256 hash for circuit text.

    This hash is used as:
    - a deduplication key
    - part of dag_ref
    - a lookup handle for SQLite storage
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def looks_like_qasm3(text: str) -> bool:
    """
    Lightweight QASM3 detection.

    This is only a heuristic. It does not guarantee that parsing will succeed.
    """
    if not text:
        return False
    t = text.lstrip().lower()
    return (
        t.startswith("openqasm 3")
        or "qubit[" in t
        or "bit[" in t
        or 'include "stdgates.inc"' in t
    )


def normalize_error_category(err: str) -> str:
    """
    Map raw errors into coarse categories for summary reporting.

    This is useful when reviewing large outputs and per-resource failure trends.
    """
    e = (err or "").strip().lower()
    if not e:
        return ""
    if "empty circuit text" in e:
        return "empty_text"
    if "list-like payload" in e or "list payload" in e:
        return "unexpected_list_payload"
    if "row has batch_len_mismatch" in e:
        return "row_batch_mismatch"
    if "parse_error" in e and "qasm2" in e:
        return "qasm2_parse_error"
    if "parse_error" in e and "qasm3" in e:
        return "qasm3_parse_error"
    if "parse_error" in e:
        return "parse_error"
    if "dag_error" in e:
        return "dag_build_error"
    return "other_error"


def shorten_error(err: str, max_len: int = 240) -> str:
    """
    Shorten long errors for compact storage in the output dataset.
    """
    if err is None:
        return ""
    err = str(err).strip().replace("\n", " ")
    if len(err) <= max_len:
        return err
    return err[: max_len - 3] + "..."


def _is_empty_value(x: Any) -> bool:
    """
    Return True for empty-like cell values.
    """
    if x is None:
        return True
    s = str(x).strip()
    return s == "" or s.lower() == "nan"


def assert_single_circuit_payload(text: Any, label: str) -> None:
    """
    Ensure that a circuit payload is a single circuit string.

    Since stage 01 now expands batch jobs before normalization, list-like
    circuit payloads reaching this stage indicate an upstream inconsistency.
    """
    if _is_empty_value(text):
        raise ValueError(f"{label}: empty circuit text")

    s = str(text).strip()
    if s.startswith("["):
        raise ValueError(f"{label}: list-like payload reached DAG stage after stage-01 expansion")


def assert_row_alignment(row_small: Dict[str, str]) -> None:
    """
    Assert row-level consistency for the post-expansion workflow.

    This function checks only lightweight row metadata. It does not parse
    circuits itself.
    """
    if "batch_len_mismatch_circuit_vs_executed_vs_result" in row_small:
        mismatch = str(row_small.get("batch_len_mismatch_circuit_vs_executed_vs_result", "")).strip().lower()
        if mismatch == "true":
            raise ValueError("row has batch_len_mismatch_circuit_vs_executed_vs_result=True")

    # Optional stage-01 raw diagnostics are kept as informational signals.
    for col in (
        "circuit_batch_raw_error",
        "executed_circuit_batch_raw_error",
        "result_batch_raw_error",
    ):
        _ = row_small.get(col, "")

# =====================================================================
# CIRCUIT PARSING AND DAG CONSTRUCTION
# =====================================================================

def parse_qasm_to_circuit(qasm_text: str):
    """
    Parse one circuit text payload into a QuantumCircuit.

    Parsing logic:
    1. reject list-like payloads
    2. guess QASM3 using a lightweight heuristic
    3. try the preferred parser first
    4. fall back to the alternative parser if needed
    """
    assert_single_circuit_payload(qasm_text, "parse_qasm_to_circuit")

    text = qasm_text.strip()
    errs: List[str] = []

    if looks_like_qasm3(text):
        if loads_qasm3 is not None:
            try:
                return loads_qasm3(text)
            except Exception as e:
                errs.append(f"qasm3: {e}")

        try:
            return qasm2.loads(text, custom_instructions=QASM2_CUSTOM_INSTRUCTIONS)
        except Exception as e:
            errs.append(f"qasm2-fallback: {e}")
    else:
        try:
            return qasm2.loads(text, custom_instructions=QASM2_CUSTOM_INSTRUCTIONS)
        except Exception as e:
            errs.append(f"qasm2: {e}")

        if loads_qasm3 is not None:
            try:
                return loads_qasm3(text)
            except Exception as e:
                errs.append(f"qasm3-fallback: {e}")

    raise ValueError(" ; ".join(errs) if errs else "no parser available")


def safe_build_dag(qasm_text: str):
    """
    Best-effort conversion:
        QASM text -> QuantumCircuit -> DAG

    Returns
    -------
    qc, dag, error_text

    Behavior
    --------
    - If parsing fails: returns (None, None, parse_error:...)
    - If DAG construction fails: returns (qc, None, dag_error:...)
    - On success: returns (qc, dag, "")
    """
    try:
        qc = parse_qasm_to_circuit(qasm_text)
    except Exception as e:
        return None, None, f"parse_error: {e}"

    try:
        dag = circuit_to_dag(qc)
        return qc, dag, ""
    except Exception as e:
        return qc, None, f"dag_error: {e}"


def dag_metrics(dag) -> Dict[str, Optional[object]]:
    """
    Compute compact DAG metrics for storage in the output dataset.

    The metrics are intentionally summary-level so they remain cheap to store
    and useful for downstream filtering and analysis.
    """
    if dag is None:
        return {
            "dag_depth": None,
            "dag_num_qubits": None,
            "dag_num_clbits": None,
            "dag_op_nodes": None,
            "dag_twoq_ops": None,
            "dag_measure_ops": None,
            "dag_layers": None,
            "dag_gate_histogram": None,
        }

    op_nodes = list(dag.op_nodes())
    gate_hist: Dict[str, int] = {}
    twoq_ops = 0
    measure_ops = 0

    for node in op_nodes:
        name = getattr(node, "name", None) or getattr(getattr(node, "op", None), "name", "unknown")
        gate_hist[name] = gate_hist.get(name, 0) + 1

        qargs = getattr(node, "qargs", ()) or ()
        if len(qargs) == 2:
            twoq_ops += 1
        if name == "measure":
            measure_ops += 1

    layer_count: Optional[int] = 0
    try:
        for _ in dag.layers():
            layer_count += 1
    except Exception:
        layer_count = None

    return {
        "dag_depth": dag.depth(),
        "dag_num_qubits": dag.num_qubits(),
        "dag_num_clbits": dag.num_clbits(),
        "dag_op_nodes": len(op_nodes),
        "dag_twoq_ops": twoq_ops,
        "dag_measure_ops": measure_ops,
        "dag_layers": layer_count,
        "dag_gate_histogram": json.dumps(gate_hist, sort_keys=True, separators=(",", ":")),
    }


def canonical_qpy_gz_from_dag(dag) -> bytes:
    """
    Convert a DAG back to a circuit and serialize it to compressed QPY bytes.

    Why store QPY instead of raw DAG objects?
    -----------------------------------------
    - DAG objects are not directly suitable for compact Parquet storage.
    - QPY provides a robust Qiskit-native serialized circuit format.
    - gzip reduces storage size further for SQLite sidecar storage.
    """
    qc2 = dag_to_circuit(dag)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        qpy.dump(qc2, gz)
    return buf.getvalue()

# =====================================================================
# SQLITE HELPERS
# =====================================================================

def init_db(conn: sqlite3.Connection) -> None:
    """
    Initialize the SQLite schema used for compressed DAG sidecar storage.
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dag_blobs (
            dag_ref TEXT PRIMARY KEY,
            circuit_hash TEXT NOT NULL,
            kind TEXT NOT NULL,
            encoding TEXT NOT NULL,
            qpy_gzip BLOB NOT NULL,
            nbytes INTEGER NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_dag_blobs_hash
        ON dag_blobs(circuit_hash)
    """)
    conn.commit()


def store_dag_blobs_batch(conn: sqlite3.Connection, records: List[Tuple[str, str, str, bytes]]) -> None:
    """
    Insert compressed DAG blobs into SQLite in batch mode.

    INSERT OR IGNORE is used because repeated circuits may produce the same
    dag_ref and hash.
    """
    if not records:
        return

    now = int(time.time())
    rows = [
        (dag_ref, circuit_hash, kind, "qpy+gzip", blob, len(blob), now)
        for dag_ref, circuit_hash, kind, blob in records
    ]

    conn.executemany("""
        INSERT OR IGNORE INTO dag_blobs
        (dag_ref, circuit_hash, kind, encoding, qpy_gzip, nbytes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)


def load_circuit_from_db(db_path: str, dag_ref: str):
    """
    Load a stored circuit from the SQLite sidecar using dag_ref.

    This is mainly a helper for later retrieval/debugging workflows.
    """
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT qpy_gzip FROM dag_blobs WHERE dag_ref = ?",
            (dag_ref,)
        ).fetchone()

        if row is None:
            raise KeyError(f"dag_ref not found: {dag_ref}")

        payload = row[0]
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as gz:
            programs = qpy.load(gz)

        if not programs:
            raise ValueError(f"QPY payload empty for dag_ref={dag_ref}")

        return programs[0]
    finally:
        conn.close()

# =====================================================================
# OUTPUT SCHEMA HELPERS
# =====================================================================

def build_drop_columns(
    input_fields: List[str],
    logical_candidates: List[str],
    native_candidates: List[str],
    drop_result_cols: bool,
    drop_circuit_cols: bool,
    extra_drop_cols: List[str],
    keep_cols: List[str],
) -> List[str]:
    """
    Build the final list of columns to drop from the OUTPUT dataset.

    Rules
    -----
    - optional heavy-column drop sets may be enabled
    - extra user-requested drop columns are included
    - protected identity columns are always preserved
    - keep-cols override drop choices
    """
    drops = set()

    if drop_result_cols:
        drops.update(DEFAULT_RESULT_COLUMNS)

    if drop_circuit_cols:
        drops.update(DEFAULT_CIRCUIT_COLUMNS)
        drops.update(logical_candidates)
        drops.update(native_candidates)

    drops.update(extra_drop_cols)

    drops.difference_update(PROTECTED_IDENTITY_COLUMNS)
    drops.difference_update(keep_cols)

    return [c for c in input_fields if c in drops]


def output_fieldnames(input_fields: List[str], drop_cols: List[str]) -> List[str]:
    """
    Build the final ordered output schema.

    The output keeps:
    - original kept columns
    - appended logical/native DAG metadata columns
    - row_drop_reason
    """
    extra = []
    for prefix in ("logical", "native"):
        extra.extend([
            f"{prefix}_dag_source_col",
            f"{prefix}_dag_attempts",
            f"{prefix}_dag_used_fallback",
            f"{prefix}_dag_ok",
            f"{prefix}_dag_error_category",
            f"{prefix}_dag_error",
            f"{prefix}_dag_ref",
            f"{prefix}_dag_hash",
            f"{prefix}_dag_blob_bytes",
            f"{prefix}_dag_depth",
            f"{prefix}_dag_num_qubits",
            f"{prefix}_dag_num_clbits",
            f"{prefix}_dag_op_nodes",
            f"{prefix}_dag_twoq_ops",
            f"{prefix}_dag_measure_ops",
            f"{prefix}_dag_layers",
            f"{prefix}_dag_gate_histogram",
        ])

    extra.append("row_drop_reason")

    drop_set = set(drop_cols)
    base_fields = [c for c in input_fields if c not in drop_set]
    existing = set(base_fields)

    return base_fields + [c for c in extra if c not in existing]


def slim_row_for_output(row: Dict[str, str], out_fields: List[str]) -> Dict[str, str]:
    """
    Keep only columns intended for final output.
    """
    return {k: row.get(k, "") for k in out_fields}

# =====================================================================
# DAG RESULT BUILDERS
# =====================================================================

def empty_dag_update(
    kind_prefix: str,
    source_col: str = "",
    attempts: int = 0,
    err: str = "empty circuit text",
) -> Dict[str, str]:
    """
    Build a default failure block for one DAG side.

    This is used when:
    - no usable text exists
    - parsing fails
    - DAG construction fails
    - row-level alignment checks fail before source attempts begin
    """
    upd = {
        f"{kind_prefix}_dag_source_col": source_col,
        f"{kind_prefix}_dag_attempts": str(attempts),
        f"{kind_prefix}_dag_used_fallback": "False",
        f"{kind_prefix}_dag_ok": "False",
        f"{kind_prefix}_dag_error_category": normalize_error_category(err),
        f"{kind_prefix}_dag_error": shorten_error(err),
        f"{kind_prefix}_dag_ref": "",
        f"{kind_prefix}_dag_hash": "",
        f"{kind_prefix}_dag_blob_bytes": "",
    }

    for k in DAG_METRIC_NAMES:
        upd[f"{kind_prefix}_{k}"] = ""

    return upd


def compute_dag_result_cached(
    qasm_text: str,
    kind_prefix: str,
    source_col: str,
) -> Tuple[Dict[str, str], Optional[Tuple[str, str, str, bytes]]]:
    """
    Build the DAG result for one circuit text using worker-local memoization.

    Returns
    -------
    update_dict, optional_blob_record

    The blob record is later inserted into SQLite in batch mode.
    """
    try:
        assert_single_circuit_payload(qasm_text, source_col or kind_prefix)
    except Exception as e:
        return empty_dag_update(
            kind_prefix,
            source_col=source_col,
            attempts=1,
            err=str(e),
        ), None

    text_hash = sha256_text(qasm_text)
    cache_key = (kind_prefix, text_hash)

    cached = _WORKER_CACHE.get(cache_key)
    if cached is not None:
        upd, blob_record = cached
        return dict(upd), blob_record

    qc, dag, err = safe_build_dag(qasm_text)

    if dag is None:
        upd = {
            f"{kind_prefix}_dag_source_col": source_col,
            f"{kind_prefix}_dag_attempts": "1",
            f"{kind_prefix}_dag_used_fallback": "False",
            f"{kind_prefix}_dag_ok": "False",
            f"{kind_prefix}_dag_error_category": normalize_error_category(err),
            f"{kind_prefix}_dag_error": shorten_error(err),
            f"{kind_prefix}_dag_ref": "",
            f"{kind_prefix}_dag_hash": text_hash,
            f"{kind_prefix}_dag_blob_bytes": "",
        }
        for k in DAG_METRIC_NAMES:
            upd[f"{kind_prefix}_{k}"] = ""

        result = (upd, None)
        _WORKER_CACHE[cache_key] = result
        return dict(upd), None

    metrics = dag_metrics(dag)
    dag_ref = f"{kind_prefix}:{text_hash}"
    blob = canonical_qpy_gz_from_dag(dag)

    upd = {
        f"{kind_prefix}_dag_source_col": source_col,
        f"{kind_prefix}_dag_attempts": "1",
        f"{kind_prefix}_dag_used_fallback": "False",
        f"{kind_prefix}_dag_ok": "True",
        f"{kind_prefix}_dag_error_category": "",
        f"{kind_prefix}_dag_error": "",
        f"{kind_prefix}_dag_ref": dag_ref,
        f"{kind_prefix}_dag_hash": text_hash,
        f"{kind_prefix}_dag_blob_bytes": str(len(blob)),
    }

    for k, v in metrics.items():
        upd[f"{kind_prefix}_{k}"] = "" if v is None else str(v)

    blob_record = (dag_ref, text_hash, kind_prefix, blob)
    result = (upd, blob_record)
    _WORKER_CACHE[cache_key] = result

    return dict(upd), blob_record


def try_candidate_columns(
    row_small: Dict[str, str],
    candidates: List[str],
    kind_prefix: str,
) -> Tuple[Dict[str, str], Optional[Tuple[str, str, str, bytes]]]:
    """
    Try source columns in priority order until one builds successfully.

    Behavior
    --------
    - empty columns are skipped
    - first success wins
    - attempts count reflects actual non-empty source tries
    - used_fallback becomes True if a later candidate succeeds
    - if all non-empty attempts fail, the last failure block is returned
    """
    attempts = 0
    first_nonempty_idx = None
    last_failure_upd = None

    for idx, col in enumerate(candidates):
        val = row_small.get(col, "")
        if not isinstance(val, str) or not val.strip():
            continue

        if first_nonempty_idx is None:
            first_nonempty_idx = idx

        attempts += 1
        upd, blob_record = compute_dag_result_cached(val, kind_prefix, col)

        if upd.get(f"{kind_prefix}_dag_ok") == "True":
            upd[f"{kind_prefix}_dag_attempts"] = str(attempts)
            upd[f"{kind_prefix}_dag_used_fallback"] = "True" if idx > 0 else "False"
            return upd, blob_record

        last_failure_upd = upd

    if last_failure_upd is not None:
        last_failure_upd[f"{kind_prefix}_dag_attempts"] = str(attempts)
        if first_nonempty_idx is not None and first_nonempty_idx > 0:
            last_failure_upd[f"{kind_prefix}_dag_used_fallback"] = "True"
        return last_failure_upd, None

    return empty_dag_update(kind_prefix, source_col="", attempts=0, err="empty circuit text"), None

# =====================================================================
# WORKER PAYLOAD HELPERS
# =====================================================================

def make_worker_row(
    row: Dict[str, str],
    logical_candidates: List[str],
    native_candidates: List[str],
    resource_col: str,
) -> Dict[str, str]:
    """
    Reduce the row payload sent to workers.

    This keeps multiprocessing overhead lower by sending only fields that
    may be required for:
    - source selection
    - diagnostics
    - invalid-row reasoning
    """
    needed_cols = set(logical_candidates) | set(native_candidates)

    if resource_col:
        needed_cols.add(resource_col)

    for identity_col in (
        "id",
        "parent_id",
        "sub_id",
        "batch_index",
        "batch_size",
        "is_batch_job",
        "batch_len_mismatch_circuit_vs_executed_vs_result",
        "circuit_batch_raw_kind",
        "circuit_batch_raw_n",
        "circuit_batch_raw_error",
        "executed_circuit_batch_raw_kind",
        "executed_circuit_batch_raw_n",
        "executed_circuit_batch_raw_error",
        "result_batch_raw_kind",
        "result_batch_raw_n",
        "result_batch_raw_error",
    ):
        if identity_col in row:
            needed_cols.add(identity_col)

    return {k: row.get(k, "") for k in needed_cols}


def process_row_payload(
    payload: Tuple[Dict[str, str], List[str], List[str], str]
) -> Tuple[Dict[str, str], List[Tuple[str, str, str, bytes]], bool, bool, str, str, str]:
    """
    Worker-side processing for one row.

    Returns
    -------
    (
        updated_fields,
        blob_records,
        logical_ok,
        native_ok,
        resource_value,
        logical_error_category,
        native_error_category
    )
    """
    row_small, logical_candidates, native_candidates, resource_col = payload

    blob_records: List[Tuple[str, str, str, bytes]] = []

    resource_value = (row_small.get(resource_col, "") or "").strip() if resource_col else ""
    if not resource_value:
        resource_value = "<missing>"

    try:
        assert_row_alignment(row_small)
        logical_upd, logical_blob = try_candidate_columns(row_small, logical_candidates, "logical")
        native_upd, native_blob = try_candidate_columns(row_small, native_candidates, "native")
    except Exception as e:
        err = str(e)
        logical_upd = empty_dag_update("logical", source_col="", attempts=0, err=err)
        native_upd = empty_dag_update("native", source_col="", attempts=0, err=err)
        logical_blob = None
        native_blob = None

    if logical_blob is not None:
        blob_records.append(logical_blob)
    if native_blob is not None:
        blob_records.append(native_blob)

    logical_ok = logical_upd.get("logical_dag_ok") == "True"
    native_ok = native_upd.get("native_dag_ok") == "True"

    updated_fields = {}
    updated_fields.update(logical_upd)
    updated_fields.update(native_upd)

    return (
        updated_fields,
        blob_records,
        logical_ok,
        native_ok,
        resource_value,
        logical_upd.get("logical_dag_error_category", ""),
        native_upd.get("native_dag_error_category", ""),
    )

# =====================================================================
# REPORTING HELPERS
# =====================================================================

def should_drop_row(policy: str, logical_ok: bool, native_ok: bool) -> bool:
    """
    Decide whether a row should be dropped under the selected invalid-row policy.
    """
    if policy == "keep":
        return False
    if policy == "drop_any_fail":
        return not (logical_ok and native_ok)
    if policy == "drop_both_fail":
        return not (logical_ok or native_ok)
    raise ValueError(f"Unknown invalid row policy: {policy}")


def compute_row_drop_reason(policy: str, logical_ok: bool, native_ok: bool) -> str:
    """
    Compute a readable drop reason for rows filtered by invalid-row policy.
    """
    if not should_drop_row(policy, logical_ok, native_ok):
        return ""
    if not logical_ok and not native_ok:
        return "both_failed"
    if not logical_ok:
        return "logical_failed"
    if not native_ok:
        return "native_failed"
    return "dropped"


def top_counter_items(counter: Counter, n: int = 5) -> List[Tuple[str, int]]:
    """
    Return the top-N counter items.
    """
    return counter.most_common(n)

# =====================================================================
# PARQUET STREAMING HELPERS
# =====================================================================

class SafeParquetChunkWriter:
    """
    Append chunked pandas DataFrames into ONE final Parquet file.

    The first non-empty chunk defines the schema. Later chunks are aligned to it.
    """

    def __init__(self, output_path: str, compression: str = "zstd") -> None:
        self.output_path = output_path
        self.compression = compression
        self.writer: Optional[pq.ParquetWriter] = None
        self.schema: Optional[pa.Schema] = None
        self.columns: Optional[List[str]] = None

    def _align_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Force later chunks to match the first chunk schema.
        """
        assert self.columns is not None
        for col in self.columns:
            if col not in df.columns:
                df[col] = None
        return df[self.columns].copy()

    def write(self, df: pd.DataFrame) -> None:
        """
        Write one output chunk into the final Parquet file.
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


def get_parquet_columns(input_path: str) -> List[str]:
    """
    Read Parquet schema column names without loading the full dataset.
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
    Stream Parquet rows in batches.

    This keeps memory usage stable for large input datasets.
    """
    pf = pq.ParquetFile(input_path)

    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()

# =====================================================================
# MAIN
# =====================================================================

def main():
    """
    CLI entry point.

    High-level flow
    ---------------
    1. read CLI settings
    2. validate candidate columns
    3. initialize SQLite sidecar
    4. stream input Parquet in chunks
    5. build logical/native DAG summaries row-wise
    6. batch-write QPY blobs into SQLite
    7. write final Parquet output directly
    """
    ap = argparse.ArgumentParser(
        description="Build, store, and reference DAGs for normalized logical and native circuits."
    )
    ap.add_argument("--input", required=True, help="Input Parquet")
    ap.add_argument("--output", required=True, help="Output Parquet with DAG columns")
    ap.add_argument("--dag-db", required=True, help="SQLite DB to store compressed DAG sidecar blobs")

    ap.add_argument(
        "--logical-candidates",
        default="circuit_best_for_sim,circuit_qasm2_norm,circuit",
        help="Comma-separated logical circuit source columns to try per row, in priority order",
    )
    ap.add_argument(
        "--native-candidates",
        default="executed_circuit_best_for_sim,executed_circuit_qasm2_norm,executed_circuit_qasm2_norm_stripped,executed_circuit",
        help="Comma-separated native circuit source columns to try per row, in priority order",
    )
    ap.add_argument("--resource-col", default="executed_resource", help="Resource column for diagnostics")
    ap.add_argument("--chunksize", type=int, default=2000, help="Rows per processing chunk")
    ap.add_argument("--progress-every", type=int, default=5000, help="Report progress every N rows")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel worker processes")
    ap.add_argument("--db-commit-every", type=int, default=5, help="Commit SQLite every N processed chunks")

    ap.add_argument(
        "--drop-result-cols",
        action="store_true",
        help="Drop large result-related columns from the OUTPUT Parquet",
    )
    ap.add_argument(
        "--drop-circuit-cols",
        action="store_true",
        help="Drop raw/normalized circuit text columns from the OUTPUT Parquet after DAG extraction",
    )
    ap.add_argument(
        "--extra-drop-cols",
        default="",
        help="Comma-separated extra columns to drop from the OUTPUT Parquet",
    )
    ap.add_argument(
        "--keep-cols",
        default="",
        help="Comma-separated columns to force-keep in the OUTPUT Parquet",
    )
    ap.add_argument(
        "--invalid-row-policy",
        choices=("keep", "drop_any_fail", "drop_both_fail"),
        default="keep",
        help=(
            "Policy for rows with DAG failures: "
            "keep=keep all rows, "
            "drop_any_fail=drop if logical OR native fails, "
            "drop_both_fail=drop only if both fail"
        ),
    )
    ap.add_argument(
        "--drop-invalid-rows",
        action="store_true",
        help="Backward-compatible alias for --invalid-row-policy drop_any_fail",
    )
    ap.add_argument(
        "--compression",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"],
        default="zstd",
        help="Parquet compression codec.",
    )

    args = ap.parse_args()

    if args.drop_invalid_rows:
        args.invalid_row_policy = "drop_any_fail"

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input Parquet not found: {args.input}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.dag_db)) or ".", exist_ok=True)

    logical_candidates = parse_csv_list_arg(args.logical_candidates) or list(DEFAULT_LOGICAL_CANDIDATES)
    native_candidates = parse_csv_list_arg(args.native_candidates) or list(DEFAULT_NATIVE_CANDIDATES)
    extra_drop_cols = parse_csv_list_arg(args.extra_drop_cols)
    keep_cols = parse_csv_list_arg(args.keep_cols)

    jobs = max(1, int(args.jobs))
    db_commit_every = max(1, int(args.db_commit_every))

    input_fields = get_parquet_columns(args.input)

    missing_logical = [c for c in logical_candidates if c not in input_fields]
    present_logical = [c for c in logical_candidates if c in input_fields]

    missing_native = [c for c in native_candidates if c not in input_fields]
    present_native = [c for c in native_candidates if c in input_fields]

    if not present_logical:
        raise ValueError("None of the logical candidate columns were found in the input Parquet.")
    if not present_native:
        raise ValueError("None of the native candidate columns were found in the input Parquet.")

    drop_cols = build_drop_columns(
        input_fields=input_fields,
        logical_candidates=present_logical,
        native_candidates=present_native,
        drop_result_cols=args.drop_result_cols,
        drop_circuit_cols=args.drop_circuit_cols,
        extra_drop_cols=extra_drop_cols,
        keep_cols=keep_cols,
    )

    out_fields = output_fieldnames(input_fields, drop_cols)
    total_input_rows = get_total_rows(args.input)

    print(f"[INFO] logical candidates    : {', '.join(present_logical)}", flush=True)
    print(f"[INFO] native candidates     : {', '.join(present_native)}", flush=True)
    print(f"[INFO] missing logical cols  : {', '.join(missing_logical) if missing_logical else '<none>'}", flush=True)
    print(f"[INFO] missing native cols   : {', '.join(missing_native) if missing_native else '<none>'}", flush=True)
    print(f"[INFO] resource column       : {args.resource_col}", flush=True)
    print(f"[INFO] worker processes      : {jobs}", flush=True)
    print(f"[INFO] invalid row policy    : {args.invalid_row_policy}", flush=True)
    print(f"[INFO] db commit every       : {db_commit_every} chunk(s)", flush=True)
    print(f"[INFO] output dropped cols   : {len(drop_cols)}", flush=True)
    print(f"[INFO] total input rows      : {total_input_rows}", flush=True)
    print(f"[INFO] compression           : {args.compression}", flush=True)
    print(f"[INFO] qasm3 available       : {loads_qasm3 is not None}", flush=True)
    print(f"[INFO] qasm2 custom count    : {len(QASM2_CUSTOM_INSTRUCTIONS)}", flush=True)
    if drop_cols:
        print(f"[INFO] dropped columns      : {', '.join(drop_cols)}", flush=True)

    conn = sqlite3.connect(args.dag_db)
    pool = None
    writer: Optional[SafeParquetChunkWriter] = None

    try:
        init_db(conn)

        if jobs > 1:
            pool = mp.Pool(processes=jobs)

        writer = SafeParquetChunkWriter(args.output, compression=args.compression)

        total_rows_read = 0
        total_rows_written = 0
        rows_dropped_invalid = 0
        logical_ok_total = 0
        native_ok_total = 0
        processed_chunks = 0

        per_resource = defaultdict(lambda: {
            "rows_read": 0,
            "rows_written": 0,
            "rows_dropped": 0,
            "logical_ok": 0,
            "native_ok": 0,
            "logical_error_categories": Counter(),
            "native_error_categories": Counter(),
        })

        pending_blob_records: List[Tuple[str, str, str, bytes]] = []

        for batch_df in iter_parquet_batches(args.input, batch_size=args.chunksize, columns=input_fields):
            batch = batch_df.to_dict(orient="records")

            payloads = [
                (
                    make_worker_row(row, present_logical, present_native, args.resource_col),
                    present_logical,
                    present_native,
                    args.resource_col,
                )
                for row in batch
            ]

            if pool is None:
                results = [process_row_payload(p) for p in payloads]
            else:
                results = pool.map(process_row_payload, payloads)

            out_rows: List[Dict[str, str]] = []

            for original_row, result in zip(batch, results):
                (
                    updated_fields,
                    blob_records,
                    row_logical_ok,
                    row_native_ok,
                    resource_value,
                    logical_err_cat,
                    native_err_cat,
                ) = result

                total_rows_read += 1
                per_resource[resource_value]["rows_read"] += 1

                pending_blob_records.extend(blob_records)

                if row_logical_ok:
                    logical_ok_total += 1
                    per_resource[resource_value]["logical_ok"] += 1
                elif logical_err_cat:
                    per_resource[resource_value]["logical_error_categories"][logical_err_cat] += 1

                if row_native_ok:
                    native_ok_total += 1
                    per_resource[resource_value]["native_ok"] += 1
                elif native_err_cat:
                    per_resource[resource_value]["native_error_categories"][native_err_cat] += 1

                row_drop_reason = compute_row_drop_reason(
                    args.invalid_row_policy,
                    row_logical_ok,
                    row_native_ok,
                )

                merged_row = {
                    k: ("" if original_row.get(k) is None else str(original_row.get(k)))
                    for k in input_fields
                }
                merged_row.update(updated_fields)
                merged_row["row_drop_reason"] = row_drop_reason

                if row_drop_reason:
                    rows_dropped_invalid += 1
                    per_resource[resource_value]["rows_dropped"] += 1
                    continue

                out_rows.append(slim_row_for_output(merged_row, out_fields))
                total_rows_written += 1
                per_resource[resource_value]["rows_written"] += 1

                if args.progress_every > 0 and total_rows_read % args.progress_every == 0:
                    db_rows = conn.execute("SELECT COUNT(*) FROM dag_blobs").fetchone()[0]
                    print(
                        f"[INFO] rows_read={total_rows_read} "
                        f"rows_written={total_rows_written} "
                        f"rows_dropped_invalid={rows_dropped_invalid} "
                        f"logical_ok={logical_ok_total} "
                        f"native_ok={native_ok_total} "
                        f"unique_dag_blobs={db_rows}",
                        flush=True,
                    )

            if out_rows:
                writer.write(pd.DataFrame(out_rows))

            if pending_blob_records:
                store_dag_blobs_batch(conn, pending_blob_records)
                pending_blob_records = []

            processed_chunks += 1
            if processed_chunks % db_commit_every == 0:
                conn.commit()

        if pending_blob_records:
            store_dag_blobs_batch(conn, pending_blob_records)
        conn.commit()

        db_rows = conn.execute("SELECT COUNT(*) FROM dag_blobs").fetchone()[0]

        print("[DONE] 07_add_dags.py finished", flush=True)
        print(f"[DONE] Input Parquet          : {args.input}", flush=True)
        print(f"[DONE] Output Parquet         : {args.output}", flush=True)
        print(f"[DONE] DAG DB                 : {args.dag_db}", flush=True)
        print(f"[DONE] Rows read              : {total_rows_read}", flush=True)
        print(f"[DONE] Rows written           : {total_rows_written}", flush=True)
        print(f"[DONE] Rows dropped           : {rows_dropped_invalid}", flush=True)
        print(f"[DONE] Logical DAG success    : {logical_ok_total}", flush=True)
        print(f"[DONE] Native DAG success     : {native_ok_total}", flush=True)
        print(f"[DONE] Unique stored DAG blobs: {db_rows}", flush=True)

        print("\n[SUMMARY] Per-resource diagnostics", flush=True)
        for resource in sorted(per_resource.keys()):
            stats = per_resource[resource]
            print(
                f"[SUMMARY] resource={resource} "
                f"rows_read={stats['rows_read']} "
                f"rows_written={stats['rows_written']} "
                f"rows_dropped={stats['rows_dropped']} "
                f"logical_ok={stats['logical_ok']} "
                f"native_ok={stats['native_ok']}",
                flush=True,
            )

            top_logical = top_counter_items(stats["logical_error_categories"], n=5)
            top_native = top_counter_items(stats["native_error_categories"], n=5)

            if top_logical:
                print(f"[SUMMARY]   top logical error categories for {resource}:", flush=True)
                for err, cnt in top_logical:
                    print(f"[SUMMARY]     {cnt}  {err}", flush=True)

            if top_native:
                print(f"[SUMMARY]   top native error categories for {resource}:", flush=True)
                for err, cnt in top_native:
                    print(f"[SUMMARY]     {cnt}  {err}", flush=True)

    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if writer is not None:
            writer.close()
        conn.close()


if __name__ == "__main__":
    main()