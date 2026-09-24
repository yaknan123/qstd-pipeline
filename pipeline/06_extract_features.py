#!/usr/bin/env python3
"""
06_add_circuit_features.py
==========================

PURPOSE
-------
Add circuit-structure features to the prepared workflow dataset.

This script is intended to run AFTER:

    01_parse_and_normalize_both_circuits.py
    02_simulate_ideal_result.py
    03_verify_result_endianness.py
    04_compute_two_hellinger_distances.py

UPDATED PIPELINE ALIGNMENT
--------------------------
In the updated workflow:
- stage 01 expands batch jobs BEFORE normalization
- each row after stage 01 should represent one sub-job / one circuit instance
- stage 02 simulates ideal results row-wise
- stage 03 verifies endianness row-wise
- stage 04 computes Hellinger row-wise

Because of that, this stage:
- prefers the stage-01 best simulation columns directly
- assumes one row = one logical circuit + one native circuit
- checks that result alignment is row-wise and single-payload
- treats list-like circuit/result payloads as upstream errors

WHAT THIS SCRIPT DOES
---------------------
For each row, it computes features for:

1. the logical circuit
   using:
       circuit_best_for_sim
   with fallback:
       circuit_qasm2_norm

2. the executed/native circuit
   using:
       executed_circuit_best_for_sim
   with fallback:
       executed_circuit_qasm2_norm

It also performs row-level alignment sanity checks so that the extracted
features correspond to the same circuit instance as the row-level result
and target values.

FEATURE GROUPS
--------------
Logical and native features include:
- n_qubits
- depth
- size
- n_1q_gates
- n_2q_gates
- n_measure
- n_reset
- n_barrier
- n_clbits
- n_parameters
- n_gate_types
- max_operand_size
- average_operand_size
- twoq_ratio
- measure_ratio
- connectedness
- liveliness
- density_2q
- component_count
- swap_count

Cross-circuit comparison features include:
- delta_depth
- delta_size
- delta_n_1q_gates
- delta_n_2q_gates
- delta_n_measure
- delta_n_qubits
- delta_swap_count
- ratio_depth
- ratio_gate_overhead
- ratio_twoq_overhead
- ratio_swap_overhead

UPDATED STORAGE / MEMORY DESIGN
-------------------------------
This version avoids the older pattern of:
- writing many parquet .part_* files
- combining them at the end

Instead, it:
1. reads Parquet input in streaming batches
2. processes each chunk independently
3. writes each processed chunk directly into ONE final Parquet file

OUTPUT
------
The output Parquet is the same input dataset plus:
- logical circuit feature columns
- native circuit feature columns
- comparison columns
- feature_error

DESIGN NOTES
------------
- One row is assumed to represent one already-expanded circuit instance.
- This script does NOT expand batch jobs again.
- This script uses normalized/best-for-sim columns from stage 01.
- If a row fails feature extraction, feature_error is populated and the script
  continues.

USAGE
-----
Basic run:
python3 06_add_circuit_features.py \
  --input /path/to/stage04_with_hellinger.parquet \
  --output /path/to/stage06_with_features.parquet

Sequential test run:
python3 06_add_circuit_features.py \
  --input /path/to/stage04_with_hellinger.parquet \
  --output /path/to/stage06_with_features.parquet \
  --max-rows 100 \
  --progress-every 20 \
  --jobs 1

Chunked run:
python3 06_add_circuit_features.py \
  --input /path/to/stage04_with_hellinger.parquet \
  --output /path/to/stage06_with_features.parquet \
  --chunksize 10000 \
  --progress-every 500

Parallel run:
python3 06_add_circuit_features.py \
  --input /path/to/stage04_with_hellinger.parquet \
  --output /path/to/stage06_with_features.parquet \
  --jobs 32

Parallel + chunked run:
python3 06_add_circuit_features.py \
  --input /path/to/stage04_with_hellinger.parquet \
  --output /path/to/stage06_with_features.parquet \
  --chunksize 10000 \
  --progress-every 500 \
  --jobs 32

Example for your workflow:
python3 06_add_circuit_features.py \
  --input /path/to/stage04_with_hellinger.parquet \
  --output /path/to/stage06_with_features.parquet \
  --chunksize 20000 \
  --progress-every 500 \
  --jobs 64 \
  --compression zstd
"""

from __future__ import annotations

import os

# =====================================================================
# THREAD CONTROL
# =====================================================================
# Limit nested threading from lower-level numeric libraries.
# This helps reduce oversubscription when multiprocessing is used.
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import ast
import json
import math
import time
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from qiskit import QuantumCircuit, qasm2

# =====================================================================
# OPTIONAL QASM3 SUPPORT
# =====================================================================
_qasm3_loads = None
try:
    from qiskit.qasm3 import loads as _qasm3_loads  # type: ignore
except Exception:
    _qasm3_loads = None

# =====================================================================
# OPTIONAL QASM2 CUSTOM-GATE SUPPORT
# =====================================================================
# Some normalized circuits may contain custom instructions such as:
# - rxx
# - ryy
# - rzz
# - rzx
# - ecr
#
# To keep stage 06 aligned with earlier stages, we register these
# instructions when the corresponding gate classes are available.
_OPTIONAL_GATE_CLASSES: Dict[str, Any] = {}


def _try_import_gate(module_path: str, class_name: str) -> Optional[Any]:
    """
    Try to import a gate class safely.

    This keeps the script portable across environments where some gate
    classes may not exist.
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

    This keeps stage 06 consistent with the parsing behavior used in
    earlier stages.
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
# BASIC HELPERS
# =====================================================================

def _is_empty(cell: Any) -> bool:
    """
    Return True if a value should be treated as empty.
    """
    if cell is None:
        return True
    if isinstance(cell, float) and np.isnan(cell):
        return True
    s = str(cell).strip()
    return s == "" or s.lower() == "nan"


def _safe_ratio(num: float, den: float) -> float:
    """
    Safe ratio helper.

    Returns NaN if the denominator is zero, missing, or invalid.
    """
    try:
        den = float(den)
        if den == 0.0:
            return float("nan")
        return float(num) / den
    except Exception:
        return float("nan")


def _safe_delta(a: Any, b: Any) -> float:
    """
    Safe numeric difference helper.

    Returns a - b when both values are numeric, otherwise NaN.
    """
    try:
        return float(a) - float(b)
    except Exception:
        return float("nan")


def _safe_json_loads(s: str) -> Tuple[bool, Any]:
    """
    Try JSON parsing without raising.
    """
    try:
        return True, json.loads(s)
    except Exception:
        return False, None


def _safe_literal_eval(s: str) -> Tuple[bool, Any]:
    """
    Try Python-literal parsing without raising.
    """
    try:
        return True, ast.literal_eval(s)
    except Exception:
        return False, None


def _loads_json_or_literal(s: str) -> Tuple[bool, Any]:
    """
    Try JSON first, then Python-literal parsing.
    """
    ok, obj = _safe_json_loads(s)
    if ok:
        return True, obj
    ok, obj = _safe_literal_eval(s)
    if ok:
        return True, obj
    return False, None


def _unwrap_json_repeatedly(s: str, max_depth: int = 6) -> Any:
    """
    Repeatedly unwrap JSON-encoded or Python-literal strings.

    This is useful for cells that have been encoded more than once.
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

        return t

    return obj


def parse_csv_list_arg(raw: Optional[str]) -> Optional[List[str]]:
    """
    Parse a comma-separated CLI value into a Python list.
    """
    if raw is None or str(raw).strip() == "":
        return None
    return [x.strip() for x in str(raw).split(",") if x.strip()]


# =====================================================================
# RESULT SANITY HELPERS
# =====================================================================

def _normalize_counts_keys(counts: Dict[str, int]) -> Dict[str, int]:
    """
    Normalize grouped count keys by removing spaces.

    Example
    -------
    "00010001 00000000" -> "0001000100000000"
    """
    out: Dict[str, int] = {}
    for k, v in counts.items():
        nk = str(k).replace(" ", "")
        out[nk] = out.get(nk, 0) + int(v)
    return out


def _parse_counts_cell(cell: Any) -> Dict[str, int]:
    """
    Parse a result cell and require it to be a SINGLE counts dictionary.

    This is part of the row-alignment sanity check. By stage 06, each row
    should already correspond to exactly one expanded result payload.
    """
    if _is_empty(cell):
        raise ValueError("result is empty")

    obj = _unwrap_json_repeatedly(str(cell).strip())

    if isinstance(obj, str):
        ok, parsed = _loads_json_or_literal(obj)
        if ok:
            obj = parsed
        else:
            raise ValueError("result string could not be decoded")

    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        obj = obj[0]

    if isinstance(obj, dict):
        return _normalize_counts_keys({str(k): int(v) for k, v in obj.items()})

    if isinstance(obj, list):
        raise ValueError("result is still a list; row was not fully expanded/aligned")

    raise ValueError("result is not a valid counts dictionary")


# =====================================================================
# CIRCUIT PAYLOAD GUARDS
# =====================================================================

def _assert_single_circuit_payload(text: Any, label: str) -> None:
    """
    Ensure that a circuit payload is a single circuit string, not a list-like
    batch payload.
    """
    if _is_empty(text):
        raise ValueError(f"{label} is empty")

    s = str(text).strip()
    if s.startswith("["):
        raise ValueError(f"{label} still contains list-like payload after stage-01 expansion")


# =====================================================================
# ROW ALIGNMENT / WORKFLOW CONSISTENCY CHECKS
# =====================================================================

def _assert_row_alignment(row: Dict[str, Any]) -> None:
    """
    Assert that the row is consistent with the updated expanded workflow.

    Checks
    ------
    - no unresolved batch mismatch flag
    - logical circuit exists and is single-payload
    - native circuit exists and is single-payload
    - at least one usable row-level result column exists and is single-payload
    """
    if "batch_len_mismatch_circuit_vs_executed_vs_result" in row:
        mismatch = row["batch_len_mismatch_circuit_vs_executed_vs_result"]
        if bool(mismatch):
            raise ValueError("row has batch_len_mismatch_circuit_vs_executed_vs_result=True")

    logical_text, _ = _choose_logical_circuit_text_and_format(row)
    native_text, _ = _choose_native_circuit_text_and_format(row)

    _assert_single_circuit_payload(logical_text, "logical circuit")
    _assert_single_circuit_payload(native_text, "native circuit")

    candidate_result_cols = [
        "result_aligned_logical",
        "result_aligned_native",
        "result",
    ]

    found_any = False
    for col in candidate_result_cols:
        if col in row and not _is_empty(row[col]):
            _parse_counts_cell(row[col])
            found_any = True
            break

    if not found_any:
        raise ValueError(
            "No usable row-level result column found among "
            "result_aligned_logical, result_aligned_native, result"
        )


# =====================================================================
# CIRCUIT PARSING
# =====================================================================

def _infer_qasm_format_from_text(qasm_text: str) -> str:
    """
    Infer whether circuit text is QASM2 or QASM3 when metadata is missing.
    """
    s = qasm_text.lstrip()

    if s.upper().startswith("OPENQASM 2"):
        return "qasm2"
    if s.upper().startswith("OPENQASM 3"):
        return "qasm3"

    s_lower = s.lower()
    if 'include "stdgates.inc"' in s_lower or "qubit[" in s_lower or "bit[" in s_lower:
        return "qasm3"
    if 'include "qelib1.inc"' in s_lower or "qreg " in s_lower or "creg " in s_lower:
        return "qasm2"

    return "unknown"


def _parse_qasm2_to_circuit(qasm_text: str) -> QuantumCircuit:
    """
    Parse OpenQASM 2 text into a QuantumCircuit using custom-instruction support.
    """
    return qasm2.loads(
        qasm_text,
        custom_instructions=QASM2_CUSTOM_INSTRUCTIONS,
    )


def _parse_qasm3_to_circuit(qasm_text: str) -> QuantumCircuit:
    """
    Parse OpenQASM 3 text into a QuantumCircuit.
    """
    if _qasm3_loads is None:
        raise ImportError("qiskit.qasm3.loads is not available in this environment.")
    return _qasm3_loads(qasm_text)


def _parse_circuit_text(qasm_text: str, declared_format: str) -> QuantumCircuit:
    """
    Parse one circuit text payload using QASM2/QASM3 routing.

    The function:
    1. checks that the payload is single-circuit
    2. uses declared format if trustworthy
    3. falls back across QASM2 and QASM3 parsers when needed
    """
    _assert_single_circuit_payload(qasm_text, "circuit text")

    fmt = declared_format.lower().strip() if declared_format else "unknown"
    if fmt not in {"qasm2", "qasm3"}:
        fmt = _infer_qasm_format_from_text(qasm_text)

    errs: List[str] = []

    if fmt == "qasm2":
        try:
            return _parse_qasm2_to_circuit(qasm_text)
        except Exception as e:
            errs.append(f"QASM2 parse failed: {e}")

        if _qasm3_loads is not None:
            try:
                return _parse_qasm3_to_circuit(qasm_text)
            except Exception as e:
                errs.append(f"QASM3 fallback failed: {e}")

        raise ValueError(" ; ".join(errs))

    if fmt == "qasm3":
        try:
            return _parse_qasm3_to_circuit(qasm_text)
        except Exception as e:
            errs.append(f"QASM3 parse failed: {e}")

        try:
            return _parse_qasm2_to_circuit(qasm_text)
        except Exception as e:
            errs.append(f"QASM2 fallback failed: {e}")

        raise ValueError(" ; ".join(errs))

    try:
        return _parse_qasm2_to_circuit(qasm_text)
    except Exception as e:
        errs.append(f"QASM2 guessed parse failed: {e}")

    if _qasm3_loads is not None:
        try:
            return _parse_qasm3_to_circuit(qasm_text)
        except Exception as e:
            errs.append(f"QASM3 guessed parse failed: {e}")

    raise ValueError(" ; ".join(errs))


def _choose_logical_circuit_text_and_format(row: Dict[str, Any]) -> Tuple[str, str]:
    """
    Select the logical circuit text and format metadata from the row.

    Preference
    ----------
    1. circuit_best_for_sim + circuit_best_for_sim_format
    2. circuit_qasm2_norm
    """
    if "circuit_best_for_sim" in row and not _is_empty(row["circuit_best_for_sim"]):
        text = str(row["circuit_best_for_sim"]).strip()
        _assert_single_circuit_payload(text, "circuit_best_for_sim")
        fmt = str(row.get("circuit_best_for_sim_format", "unknown")).strip().lower()
        return text, fmt

    if "circuit_qasm2_norm" in row and not _is_empty(row["circuit_qasm2_norm"]):
        text = str(row["circuit_qasm2_norm"]).strip()
        _assert_single_circuit_payload(text, "circuit_qasm2_norm")
        return text, "qasm2"

    raise ValueError("No usable logical circuit column found.")


def _choose_native_circuit_text_and_format(row: Dict[str, Any]) -> Tuple[str, str]:
    """
    Select the native circuit text and format metadata from the row.

    Preference
    ----------
    1. executed_circuit_best_for_sim + executed_circuit_best_for_sim_format
    2. executed_circuit_qasm2_norm
    """
    if "executed_circuit_best_for_sim" in row and not _is_empty(row["executed_circuit_best_for_sim"]):
        text = str(row["executed_circuit_best_for_sim"]).strip()
        _assert_single_circuit_payload(text, "executed_circuit_best_for_sim")
        fmt = str(row.get("executed_circuit_best_for_sim_format", "unknown")).strip().lower()
        return text, fmt

    if "executed_circuit_qasm2_norm" in row and not _is_empty(row["executed_circuit_qasm2_norm"]):
        text = str(row["executed_circuit_qasm2_norm"]).strip()
        _assert_single_circuit_payload(text, "executed_circuit_qasm2_norm")
        return text, "qasm2"

    raise ValueError("No usable executed/native circuit column found.")


# =====================================================================
# GRAPH / TOPOLOGY HELPERS
# =====================================================================

def _build_2q_graph(qc: QuantumCircuit) -> Tuple[Set[int], List[Tuple[int, int]]]:
    """
    Build an undirected interaction graph from multi-qubit operations.

    Nodes are qubit indices.
    Edges connect qubits that participate together in a multi-qubit operation.
    """
    qubit_to_index = {qb: i for i, qb in enumerate(qc.qubits)}
    nodes: Set[int] = set()
    edge_set: Set[Tuple[int, int]] = set()

    for inst in qc.data:
        qargs = inst.qubits
        if len(qargs) < 2:
            continue

        q_indices = [qubit_to_index[qb] for qb in qargs]
        nodes.update(q_indices)

        for i in range(len(q_indices)):
            for j in range(i + 1, len(q_indices)):
                u, v = sorted((q_indices[i], q_indices[j]))
                edge_set.add((u, v))

    return nodes, sorted(edge_set)


def _connected_components(nodes: Iterable[int], edges: Iterable[Tuple[int, int]]) -> int:
    """
    Count connected components in an undirected graph.
    """
    nodes = set(nodes)
    if not nodes:
        return 0

    adj: Dict[int, Set[int]] = {n: set() for n in nodes}
    for u, v in edges:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)

    visited: Set[int] = set()
    components = 0

    for start in nodes:
        if start in visited:
            continue

        components += 1
        stack = [start]
        visited.add(start)

        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)

    return components


# =====================================================================
# FEATURE EXTRACTION
# =====================================================================

def _extract_circuit_features(qc: QuantumCircuit, prefix: str) -> Dict[str, Any]:
    """
    Extract circuit-structure features for one circuit.

    Parameters
    ----------
    qc : QuantumCircuit
        Parsed circuit.
    prefix : str
        Either 'logical' or 'native'.

    Returns
    -------
    dict
        Feature-name to feature-value mapping with the chosen prefix.
    """
    features: Dict[str, Any] = {}

    n_qubits = qc.num_qubits
    n_clbits = qc.num_clbits
    depth = qc.depth()
    size = qc.size()

    ops = qc.count_ops()

    n_measure = int(ops.get("measure", 0))
    n_reset = int(ops.get("reset", 0))
    n_barrier = int(ops.get("barrier", 0))
    swap_count = int(ops.get("swap", 0))

    n_1q_gates = 0
    n_2q_gates = 0
    max_operand_size = 0
    operand_sizes: List[int] = []
    gate_types: Set[str] = set()

    active_qubits: Set[int] = set()
    qubit_to_index = {qb: i for i, qb in enumerate(qc.qubits)}

    for inst in qc.data:
        op_name = inst.operation.name
        qargs = inst.qubits
        nq = len(qargs)

        operand_sizes.append(nq)
        max_operand_size = max(max_operand_size, nq)

        for qb in qargs:
            active_qubits.add(qubit_to_index[qb])

        gate_types.add(op_name)

        if nq == 1:
            n_1q_gates += 1
        elif nq == 2:
            n_2q_gates += 1

    average_operand_size = float(np.mean(operand_sizes)) if operand_sizes else 0.0
    n_gate_types = len(gate_types)
    n_parameters = len(qc.parameters)

    graph_nodes, graph_edges = _build_2q_graph(qc)
    component_count = _connected_components(graph_nodes, graph_edges)

    max_possible_edges = n_qubits * (n_qubits - 1) / 2 if n_qubits >= 2 else 0.0
    connectedness = _safe_ratio(len(graph_edges), max_possible_edges)

    liveliness = _safe_ratio(len(active_qubits), n_qubits)
    density_2q = _safe_ratio(n_2q_gates, size)
    twoq_ratio = _safe_ratio(n_2q_gates, size)
    measure_ratio = _safe_ratio(n_measure, size)

    features[f"{prefix}_n_qubits"] = n_qubits
    features[f"{prefix}_n_clbits"] = n_clbits
    features[f"{prefix}_depth"] = depth
    features[f"{prefix}_size"] = size
    features[f"{prefix}_n_1q_gates"] = n_1q_gates
    features[f"{prefix}_n_2q_gates"] = n_2q_gates
    features[f"{prefix}_n_measure"] = n_measure
    features[f"{prefix}_n_reset"] = n_reset
    features[f"{prefix}_n_barrier"] = n_barrier
    features[f"{prefix}_n_parameters"] = n_parameters
    features[f"{prefix}_n_gate_types"] = n_gate_types
    features[f"{prefix}_max_operand_size"] = max_operand_size
    features[f"{prefix}_average_operand_size"] = average_operand_size
    features[f"{prefix}_twoq_ratio"] = twoq_ratio
    features[f"{prefix}_measure_ratio"] = measure_ratio
    features[f"{prefix}_connectedness"] = connectedness
    features[f"{prefix}_liveliness"] = liveliness
    features[f"{prefix}_density_2q"] = density_2q
    features[f"{prefix}_component_count"] = component_count
    features[f"{prefix}_swap_count"] = swap_count

    return features


def _extract_comparison_features(logical: Dict[str, Any], native: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute native-versus-logical comparison features.

    These features describe circuit overhead introduced between the logical
    circuit and the executed/native circuit.
    """
    out: Dict[str, Any] = {}

    out["delta_depth"] = _safe_delta(native.get("native_depth"), logical.get("logical_depth"))
    out["delta_size"] = _safe_delta(native.get("native_size"), logical.get("logical_size"))
    out["delta_n_1q_gates"] = _safe_delta(native.get("native_n_1q_gates"), logical.get("logical_n_1q_gates"))
    out["delta_n_2q_gates"] = _safe_delta(native.get("native_n_2q_gates"), logical.get("logical_n_2q_gates"))
    out["delta_n_measure"] = _safe_delta(native.get("native_n_measure"), logical.get("logical_n_measure"))
    out["delta_n_qubits"] = _safe_delta(native.get("native_n_qubits"), logical.get("logical_n_qubits"))
    out["delta_swap_count"] = _safe_delta(native.get("native_swap_count"), logical.get("logical_swap_count"))

    out["ratio_depth"] = _safe_ratio(native.get("native_depth"), logical.get("logical_depth"))
    out["ratio_gate_overhead"] = _safe_ratio(native.get("native_size"), logical.get("logical_size"))
    out["ratio_twoq_overhead"] = _safe_ratio(native.get("native_n_2q_gates"), logical.get("logical_n_2q_gates"))
    out["ratio_swap_overhead"] = _safe_ratio(native.get("native_swap_count"), logical.get("logical_swap_count"))

    return out


# =====================================================================
# ROW PROCESSING
# =====================================================================

def _process_row_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute all circuit features for one row.

    On failure, the function records the error in feature_error and returns
    a partial result so the full dataset can still be processed.
    """
    out: Dict[str, Any] = {"feature_error": ""}

    try:
        _assert_row_alignment(row)

        logical_text, logical_fmt = _choose_logical_circuit_text_and_format(row)
        native_text, native_fmt = _choose_native_circuit_text_and_format(row)

        logical_qc = _parse_circuit_text(logical_text, logical_fmt)
        native_qc = _parse_circuit_text(native_text, native_fmt)

        logical_features = _extract_circuit_features(logical_qc, prefix="logical")
        native_features = _extract_circuit_features(native_qc, prefix="native")
        comparison_features = _extract_comparison_features(logical_features, native_features)

        out.update(logical_features)
        out.update(native_features)
        out.update(comparison_features)

    except Exception as exc:
        out["feature_error"] = f"{type(exc).__name__}: {exc}"

    return out


def _worker_process_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Multiprocessing worker wrapper for per-row feature extraction.
    """
    return _process_row_dict(row)


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
    progress_every: int,
    label: str,
    jobs: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Process one dataframe batch and append feature columns.

    This function:
    1. validates that required circuit columns exist
    2. computes features row by row
    3. appends results to the input dataframe
    4. returns both the updated dataframe and summary statistics
    """
    has_new_logical = "circuit_best_for_sim" in df.columns
    has_old_logical = "circuit_qasm2_norm" in df.columns
    has_new_native = "executed_circuit_best_for_sim" in df.columns
    has_old_native = "executed_circuit_qasm2_norm" in df.columns

    if not (has_new_logical or has_old_logical):
        raise ValueError(
            "Missing logical circuit input column. Expected either "
            "'circuit_best_for_sim' or 'circuit_qasm2_norm'."
        )

    if not (has_new_native or has_old_native):
        raise ValueError(
            "Missing native circuit input column. Expected either "
            "'executed_circuit_best_for_sim' or 'executed_circuit_qasm2_norm'."
        )

    total = len(df)
    start = time.time()
    print(f"[INFO] Starting feature extraction for {label}: {total} row(s)", flush=True)

    row_dicts = df.to_dict(orient="records")

    if jobs <= 1:
        feature_rows: List[Dict[str, Any]] = []
        success_count = 0
        fail_count = 0

        for pos, row in enumerate(row_dicts, start=1):
            feature_row = _process_row_dict(row)
            feature_rows.append(feature_row)

            if feature_row.get("feature_error", "") == "":
                success_count += 1
            else:
                fail_count += 1

            if progress_every > 0 and (pos % progress_every == 0 or pos == total):
                elapsed = time.time() - start
                rate = pos / elapsed if elapsed > 0 else 0.0
                print(
                    f"[INFO] {label}: processed {pos}/{total} rows | "
                    f"success={success_count} | failed={fail_count} | "
                    f"elapsed={elapsed:.2f}s | rate={rate:.2f} rows/s",
                    flush=True,
                )
    else:
        print(f"[INFO] Using {jobs} worker process(es)", flush=True)
        feature_rows = []
        success_count = 0
        fail_count = 0

        with Pool(processes=jobs) as pool:
            for pos, feature_row in enumerate(
                pool.imap(_worker_process_row, row_dicts, chunksize=_pool_chunksize(total, jobs)),
                start=1,
            ):
                feature_rows.append(feature_row)

                if feature_row.get("feature_error", "") == "":
                    success_count += 1
                else:
                    fail_count += 1

                if progress_every > 0 and (pos % progress_every == 0 or pos == total):
                    elapsed = time.time() - start
                    rate = pos / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[INFO] {label}: processed {pos}/{total} rows | "
                        f"success={success_count} | failed={fail_count} | "
                        f"elapsed={elapsed:.2f}s | rate={rate:.2f} rows/s",
                        flush=True,
                    )

    features_df = pd.DataFrame(feature_rows)

    if "feature_error" not in features_df.columns:
        features_df["feature_error"] = ""

    out = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

    duration = time.time() - start
    stats = {
        "rows": total,
        "success": success_count,
        "failed": fail_count,
        "duration_sec": duration,
    }

    print(
        f"[INFO] Completed {label} | "
        f"rows={total} | success={success_count}/{total} | "
        f"failed={fail_count}/{total} | duration={duration:.2f}s",
        flush=True,
    )

    return out, stats


# =====================================================================
# PARQUET STREAMING WRITER
# =====================================================================

class SafeParquetChunkWriter:
    """
    Append chunked pandas DataFrames into ONE final Parquet file.

    The schema is fixed by the first non-empty chunk and reused for all later
    chunks. This avoids many temporary files and a final merge step.
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
    1. validates input configuration
    2. processes the dataset chunk by chunk or all at once
    3. writes one final Parquet output
    4. prints processing and success/failure summaries
    """
    parser = argparse.ArgumentParser(
        description="Add logical/native circuit features using normalized expanded stage-01 columns."
    )
    parser.add_argument("--input", required=True, help="Input Parquet file")
    parser.add_argument("--output", required=True, help="Output Parquet file")
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
        default=500,
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

    input_cols = set(get_input_columns(args.input))

    logical_ok = (
        "circuit_best_for_sim" in input_cols or
        "circuit_qasm2_norm" in input_cols
    )
    native_ok = (
        "executed_circuit_best_for_sim" in input_cols or
        "executed_circuit_qasm2_norm" in input_cols
    )

    if not logical_ok:
        raise ValueError(
            "Missing logical circuit input column. Expected either "
            "'circuit_best_for_sim' or 'circuit_qasm2_norm'."
        )

    if not native_ok:
        raise ValueError(
            "Missing native circuit input column. Expected either "
            "'executed_circuit_best_for_sim' or 'executed_circuit_qasm2_norm'."
        )

    if requested_usecols is not None:
        requested_set = set(requested_usecols)
        if not (
            {"circuit_best_for_sim"} <= requested_set or
            {"circuit_qasm2_norm"} <= requested_set
        ):
            raise SystemExit(
                "--usecols must include either 'circuit_best_for_sim' or 'circuit_qasm2_norm'."
            )

        if not (
            {"executed_circuit_best_for_sim"} <= requested_set or
            {"executed_circuit_qasm2_norm"} <= requested_set
        ):
            raise SystemExit(
                "--usecols must include either 'executed_circuit_best_for_sim' or "
                "'executed_circuit_qasm2_norm'."
            )

    total_input_rows = get_total_rows(args.input)
    effective_total_rows = min(total_input_rows, args.max_rows) if args.max_rows is not None else total_input_rows

    overall_start = time.time()
    total_rows_processed = 0
    total_success = 0
    total_failed = 0
    chunk_count = 0

    writer: Optional[SafeParquetChunkWriter] = None
    compression = args.compression

    print("[INFO] Circuit feature extraction started", flush=True)
    print(f"[INFO] Input file:                {args.input}", flush=True)
    print(f"[INFO] Output file:               {args.output}", flush=True)
    print(f"[INFO] Chunksize:                 {args.chunksize if args.chunksize > 0 else 'all-at-once'}", flush=True)
    print(f"[INFO] Max rows:                  {args.max_rows if args.max_rows is not None else 'all'}", flush=True)
    print(f"[INFO] Progress every:            {args.progress_every}", flush=True)
    print(f"[INFO] Worker processes (--jobs): {args.jobs}", flush=True)
    print(f"[INFO] Compression:               {args.compression}", flush=True)
    print(f"[INFO] QASM3 available:           {_qasm3_loads is not None}", flush=True)
    print(f"[INFO] QASM2 custom count:        {len(QASM2_CUSTOM_INSTRUCTIONS)}", flush=True)
    print(f"[INFO] Input rows:                {total_input_rows}", flush=True)
    print(f"[INFO] Rows to process:           {effective_total_rows}", flush=True)

    try:
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
                    progress_every=args.progress_every,
                    label=label,
                    jobs=args.jobs,
                )

                writer.write(out)

                rows_seen += len(chunk)
                total_rows_processed += stats["rows"]
                total_success += stats["success"]
                total_failed += stats["failed"]

                print(
                    f"[INFO] Written {label} to output | "
                    f"input rows read so far={rows_seen}/{effective_total_rows} | "
                    f"cumulative rows={total_rows_processed} | "
                    f"cumulative success={total_success} | "
                    f"cumulative failed={total_failed}",
                    flush=True,
                )

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
                jobs=args.jobs,
            )

            out.to_parquet(
                args.output,
                index=False,
                compression=None if compression == "none" else compression,
            )

            total_rows_processed = stats["rows"]
            total_success = stats["success"]
            total_failed = stats["failed"]
            chunk_count = 1

        total_duration = time.time() - overall_start
        out_size = os.path.getsize(args.output) if os.path.exists(args.output) else 0

        print("\n========== FINAL REPORT ==========", flush=True)
        print(f"Saved output to:                 {args.output}", flush=True)
        print(f"Input file:                      {args.input}", flush=True)
        print(f"Rows processed:                  {total_rows_processed}", flush=True)
        print(f"Output size bytes:               {out_size}", flush=True)
        print(f"Chunks processed:                {chunk_count if args.chunksize and args.chunksize > 0 else 1}", flush=True)
        print(f"Feature extraction succeeded:    {total_success}/{total_rows_processed}", flush=True)
        print(f"Feature extraction failed:       {total_failed}/{total_rows_processed}", flush=True)

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