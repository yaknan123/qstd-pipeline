#!/usr/bin/env python3
"""
02_simulate_ideal_result.py
===========================

PURPOSE
-------
Simulate ideal results for both:

1. the logical circuit
2. the native executed circuit

using the already-normalized, already-expanded output from:

    01_parse_and_normalize_both_circuits.py

UPDATED ALIGNMENT
-----------------
Stage 01 now expands batch jobs BEFORE normalization.

Therefore stage 02 now assumes:

- one row = one logical circuit
- one row = one native circuit
- one row = one result payload

This means:

- *_best_for_sim fields must be SINGLE circuit strings
- *_qasm2_norm fallback fields must also represent SINGLE circuit strings
- list-like payloads are treated as an upstream error
- stage 02 is now a strictly row-wise simulation stage

IMPORTANT DESIGN CHOICE
-----------------------
This stage still follows the safer normalized simulation workflow.

Therefore:

- DO NOT strip or rename native custom gates
- DO parse the best normalized text from stage 01
- DO repeatedly decompose custom gates
- DO transpile to AerSimulator
- DO simulate the resulting Aer-compatible circuit

WHY THIS STAGE EXISTS
---------------------
Later analysis requires ideal probability/count distributions for both:

- the logical circuit representation
- the native executed circuit representation

These ideal outputs are needed for:

- endianness verification
- Hellinger distance computation
- downstream feature analysis
- ML dataset preparation

EXPECTED INPUT
--------------
Required base columns:
- id
- shots

Preferred logical input:
- circuit_best_for_sim
- circuit_best_for_sim_format

Fallback logical input:
- circuit_qasm2_norm

Preferred native input:
- executed_circuit_best_for_sim
- executed_circuit_best_for_sim_format

Fallback native input:
- executed_circuit_qasm2_norm

OPTIONAL COMMON COLUMNS
-----------------------
These are not required by this stage, but often exist in the pipeline:
- parent_id
- sub_id
- batch_index
- batch_size
- is_batch_job
- timestamp_scheduled
- executed_resource
- result

OUTPUT COLUMNS ADDED
--------------------
Logical:
- ideal_result_logical
- ideal_result_logical_error
- logical_num_qubits
- logical_depth_before_decompose
- logical_depth_after_decompose
- logical_sim_status

Native:
- ideal_result_native
- ideal_result_native_error
- native_num_qubits
- native_depth_before_decompose
- native_depth_after_decompose
- native_sim_status

STORAGE / MEMORY DESIGN
-----------------------
This version avoids the older pattern of:
- writing many temporary .part_* parquet files
- merging all parts at the end

Instead, it:

1. reads input Parquet in streaming batches
2. processes one chunk at a time
3. writes each processed chunk directly into ONE final output Parquet file

This is safer for large datasets and reduces end-of-run memory pressure.

SIMULATION NOTES
----------------
- QASM2 custom instructions such as rxx / ryy / rzz / rzx / ecr are supported
  when available in the installed Qiskit version.
- QASM3 parsing is used when available and when the text is identified as QASM3.
- If a circuit has no measurements and --measure-all-if-missing is enabled,
  measurements are added before simulation.
- Count keys are normalized by removing spaces between classical register groups.

USAGE
-----
Basic run:
python3 02_simulate_ideal_result.py \
  --input /path/to/stage01_normalized.parquet \
  --output /path/to/stage02_with_ideal.parquet \
  --measure-all-if-missing

Sequential run:
python3 02_simulate_ideal_result.py \
  --input /path/to/stage01_normalized.parquet \
  --output /path/to/stage02_with_ideal.parquet \
  --measure-all-if-missing \
  --jobs 1

Parallel run with chunking:
python3 02_simulate_ideal_result.py \
  --input /path/to/stage01_normalized.parquet \
  --output /path/to/stage02_with_ideal.parquet \
  --measure-all-if-missing \
  --chunksize 500 \
  --jobs 4

Larger parallel run:
python3 02_simulate_ideal_result.py \
  --input /path/to/stage01_normalized.parquet \
  --output /path/to/stage02_with_ideal.parquet \
  --measure-all-if-missing \
  --chunksize 5000 \
  --jobs 24 \
  --compression zstd

All-at-once run on a small dataset:
python3 02_simulate_ideal_result.py \
  --input /path/to/stage01_normalized.parquet \
  --output /path/to/stage02_with_ideal.parquet \
  --measure-all-if-missing \
  --chunksize 0

Test run on the first 100 rows:
python3 02_simulate_ideal_result.py \
  --input /path/to/stage01_normalized.parquet \
  --output /path/to/stage02_test.parquet \
  --measure-all-if-missing \
  --max-rows 100 \
  --jobs 4

Example for your workflow:
python3 02_simulate_ideal_result.py \
  --input /path/to/stage01_normalized.parquet \
  --output /path/to/stage02_with_ideal.parquet \
  --measure-all-if-missing \
  --chunksize 5000 \
  --jobs 24 \
  --compression zstd

Restricted-column run:
python3 02_simulate_ideal_result.py \
  --input /path/to/stage01_normalized.parquet \
  --output /path/to/stage02_with_ideal.parquet \
  --measure-all-if-missing \
  --usecols id,shots,circuit_best_for_sim,circuit_best_for_sim_format,executed_circuit_best_for_sim,executed_circuit_best_for_sim_format
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------
# Limit nested threading from libraries used inside worker processes.
# This reduces oversubscription and improves stability on large nodes.
# ---------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# Optional QASM3 parser support.
# If unavailable, QASM3 rows may fail unless they can be recovered by
# fallback parsing.
# ---------------------------------------------------------------------
_qasm3_loads = None
try:
    from qiskit.qasm3 import loads as _qasm3_loads  # type: ignore
except Exception:
    _qasm3_loads = None

# ---------------------------------------------------------------------
# Optional gate-class imports for broad QASM2 custom-instruction support.
# These imports are performed safely so the script remains compatible
# across different Qiskit installations.
# ---------------------------------------------------------------------
_OPTIONAL_GATE_CLASSES: Dict[str, Any] = {}


def _try_import_gate(module_path: str, class_name: str) -> Optional[Any]:
    """
    Try to import a gate class safely.

    Returns
    -------
    Optional[Any]
        The gate class if import succeeds, otherwise None.
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
    Build a broad QASM2 custom-instruction registry.

    This keeps stage 02 aligned with stage 01 so both stages can parse the
    same family of normalized QASM2 circuits.
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

Counts = Dict[str, int]

# ---------------------------------------------------------------------
# Global worker-local state used only in multiprocessing mode.
# Each worker gets its own AerSimulator instance.
# ---------------------------------------------------------------------
_WORKER_SIM: AerSimulator | None = None
_WORKER_MEASURE_ALL_IF_MISSING: bool = False

# ---------------------------------------------------------------------
# Output column groups used for dtype normalization before Parquet write.
# ---------------------------------------------------------------------
STRING_OUTPUT_COLUMNS = [
    "ideal_result_logical",
    "ideal_result_native",
    "ideal_result_logical_error",
    "ideal_result_native_error",
    "logical_sim_status",
    "native_sim_status",
]

INT_OUTPUT_COLUMNS = [
    "logical_num_qubits",
    "logical_depth_before_decompose",
    "logical_depth_after_decompose",
    "native_num_qubits",
    "native_depth_before_decompose",
    "native_depth_after_decompose",
]


# =====================================================================
# BASIC HELPERS
# =====================================================================

def _is_empty(cell: Any) -> bool:
    """
    Return True if a cell should be treated as empty.
    """
    if cell is None:
        return True
    if isinstance(cell, float) and np.isnan(cell):
        return True

    s = str(cell).strip()
    return s == "" or s.lower() == "nan"


def _safe_parse_shots(value: Any) -> int:
    """
    Parse and validate the shots value.

    A valid simulation row must have shots > 0.
    """
    shots = int(value)
    if shots <= 0:
        raise ValueError("shots must be > 0")
    return shots


def parse_csv_list_arg(raw: Optional[str]) -> Optional[List[str]]:
    """
    Parse a comma-separated CLI string into a Python list.
    """
    if raw is None or str(raw).strip() == "":
        return None
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _recommended_jobs(requested_jobs: int) -> int:
    """
    Choose a safer default worker count when the user passes --jobs 0.

    This intentionally avoids using all CPUs because Aer simulation can be
    heavy and nested threading may still exist elsewhere in the stack.
    """
    if requested_jobs > 0:
        return requested_jobs

    detected = cpu_count()
    if detected >= 256:
        return 24
    if detected >= 128:
        return 20
    if detected >= 64:
        return 16
    return max(1, min(8, detected))


def _validate_required_columns(df: pd.DataFrame) -> None:
    """
    Validate that the dataframe contains the minimum required columns.

    Stage 02 accepts either:
    - new stage-01 best-for-sim columns
    - old qasm2_norm fallback columns
    """
    required_base = ["id", "shots"]
    missing_base = [col for col in required_base if col not in df.columns]
    if missing_base:
        raise ValueError(f"Missing required column(s): {missing_base}")

    has_new_logical = "circuit_best_for_sim" in df.columns
    has_old_logical = "circuit_qasm2_norm" in df.columns
    if not (has_new_logical or has_old_logical):
        raise ValueError(
            "Missing logical simulation input column. Expected either "
            "'circuit_best_for_sim' or 'circuit_qasm2_norm'."
        )

    has_new_native = "executed_circuit_best_for_sim" in df.columns
    has_old_native = "executed_circuit_qasm2_norm" in df.columns
    if not (has_new_native or has_old_native):
        raise ValueError(
            "Missing native simulation input column. Expected either "
            "'executed_circuit_best_for_sim' or 'executed_circuit_qasm2_norm'."
        )


# =====================================================================
# QASM FORMAT RESOLUTION
# =====================================================================

def _infer_qasm_format_from_text(qasm_text: str) -> str:
    """
    Infer whether circuit text looks like QASM2 or QASM3.

    This is used when explicit metadata is missing or unreliable.
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


def _assert_single_circuit_payload(circuit_text: str, prefix: str) -> None:
    """
    Enforce the updated pipeline rule that each row must contain one circuit.

    If stage 01 expanded batch jobs correctly, stage 02 should never receive
    list-like circuit payloads here.
    """
    s = str(circuit_text).strip()
    if s.startswith("["):
        raise ValueError(
            f"{prefix} still contains a list-like payload. "
            "Stage 02 now expects stage 01 to expand batch jobs before normalization."
        )


def _get_circuit_text_and_format(
    row: pd.Series | Dict[str, Any],
    prefix: str,
) -> Tuple[str, str]:
    """
    Resolve the best circuit text and declared format for one circuit family.

    Parameters
    ----------
    row : pd.Series or dict
        Row data.
    prefix : str
        Either:
        - 'circuit'
        - 'executed_circuit'

    Priority
    --------
    1. <prefix>_best_for_sim
    2. <prefix>_qasm2_norm

    Returns
    -------
    (text, format)
        format is one of:
        - qasm2
        - qasm3
        - unknown
    """
    best_col = f"{prefix}_best_for_sim"
    best_fmt_col = f"{prefix}_best_for_sim_format"
    old_col = f"{prefix}_qasm2_norm"

    if best_col in row and not _is_empty(row[best_col]):
        text = str(row[best_col]).strip()
        _assert_single_circuit_payload(text, prefix)

        fmt = str(row.get(best_fmt_col, "unknown")).strip().lower()
        if fmt in {"qasm2", "qasm3"}:
            return text, fmt
        return text, "unknown"

    if old_col in row and not _is_empty(row[old_col]):
        text = str(row[old_col]).strip()
        _assert_single_circuit_payload(text, prefix)
        return text, "qasm2"

    raise ValueError(f"No usable circuit text found for prefix '{prefix}'.")


# =====================================================================
# CIRCUIT PARSING
# =====================================================================

def _parse_qasm2_to_circuit(qasm_text: str) -> QuantumCircuit:
    """
    Parse OpenQASM 2 text using the broad custom-instruction set.
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
    Parse circuit text using declared format first, then conservative fallbacks.

    Strategy
    --------
    1. Trust declared qasm2/qasm3 if available
    2. Otherwise infer from the text
    3. Fall back between QASM2 and QASM3 if needed
    """
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


# =====================================================================
# CIRCUIT PREPARATION FOR SIMULATION
# =====================================================================

def _has_measurements(qc: QuantumCircuit) -> bool:
    """
    Return True if the circuit already contains at least one measurement.
    """
    return any(inst.operation.name == "measure" for inst in qc.data)


def _fully_decompose(qc: QuantumCircuit, max_rounds: int = 20) -> QuantumCircuit:
    """
    Repeatedly decompose the circuit until it stops changing.

    This is the key step that makes many backend/native custom constructions
    Aer-compatible without renaming the original gate names.
    """
    current = qc
    for _ in range(max_rounds):
        nxt = current.decompose()
        if nxt == current:
            break
        current = nxt
    return current


def _prepare_measured_circuit(
    qc: QuantumCircuit,
    measure_all_if_missing: bool,
) -> QuantumCircuit:
    """
    Ensure the circuit has measurements if requested.

    Existing measurements are preserved.
    """
    qc2 = qc.copy()

    if _has_measurements(qc2):
        return qc2

    if not measure_all_if_missing:
        return qc2

    qc2.measure_all()
    return qc2


def _prepare_circuit_for_simulation(
    qc: QuantumCircuit,
    sim: AerSimulator,
    measure_all: bool,
    seed: int | None = None,
) -> tuple[QuantumCircuit, int | None, int | None]:
    """
    Prepare a circuit for Aer simulation.

    Steps
    -----
    1. record original depth
    2. decompose repeatedly
    3. record post-decomposition depth
    4. add measurements if requested
    5. transpile to AerSimulator
    """
    depth_before = qc.depth()

    qc2 = _fully_decompose(qc, max_rounds=20)
    depth_after = qc2.depth()

    qc2 = _prepare_measured_circuit(qc2, measure_all_if_missing=measure_all)
    # Seeded with the sampling seed when --seed is given: some transpiler passes
    # are randomised, and a different circuit samples differently.
    qc2 = transpile(qc2, sim) if seed is None else transpile(qc2, sim, seed_transpiler=seed)

    return qc2, depth_before, depth_after


# =====================================================================
# SIMULATION HELPERS
# =====================================================================

def _normalize_counts_keys(counts: Counts) -> Counts:
    """
    Normalize count keys by removing spaces between classical register groups.
    """
    normalized: Counts = {}
    for key, value in counts.items():
        new_key = key.replace(" ", "")
        normalized[new_key] = normalized.get(new_key, 0) + int(value)
    return normalized


SEED_ENV = "QSTD_SIM_SEED"


def _row_seed(row: pd.Series | Dict[str, Any], side: str) -> int | None:
    """Deterministic per-circuit seed from --seed, or None (unseeded, the default).

    The ideal distribution is SAMPLED with the job's shot count, so without a
    seed every run of this stage draws a different sample (the original QSTD
    build was unseeded). With --seed the sample depends only on the seed, the
    circuit and the side, so reruns reproduce it; the distribution is the same.
    Passed through the environment so pool workers inherit it.
    """
    base = os.environ.get(SEED_ENV)
    if base is None:
        return None
    key = row.get("sub_id") or row.get("id")
    digest = hashlib.blake2b(f"{base}:{key}:{side}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def _simulate_qc(
    qc: QuantumCircuit,
    shots: int,
    measure_all: bool,
    sim: AerSimulator,
    seed: int | None = None,
) -> tuple[Counts, int | None, int | None]:
    """
    Simulate one already-parsed circuit.
    """
    qc2, depth_before, depth_after = _prepare_circuit_for_simulation(
        qc=qc,
        sim=sim,
        measure_all=measure_all,
        seed=seed,
    )

    job = (sim.run(qc2, shots=int(shots)) if seed is None
           else sim.run(qc2, shots=int(shots), seed_simulator=seed))
    result = job.result()
    counts = dict(result.get_counts(qc2))
    counts = _normalize_counts_keys(counts)

    return counts, depth_before, depth_after


def _simulate_single_circuit_cell(
    circuit_text: str,
    circuit_format: str,
    shots: int,
    measure_all: bool,
    sim: AerSimulator,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Simulate one circuit text cell and return compact row-wise outputs.
    """
    if _is_empty(circuit_text):
        raise ValueError("Circuit cell is empty")

    _assert_single_circuit_payload(circuit_text, "circuit_cell")
    qc = _parse_circuit_text(circuit_text, circuit_format)

    counts, depth_before, depth_after = _simulate_qc(
        qc=qc,
        shots=shots,
        measure_all=measure_all,
        sim=sim,
        seed=seed,
    )

    return {
        "result_json": json.dumps(counts, ensure_ascii=False, sort_keys=True),
        "num_qubits": qc.num_qubits,
        "depth_before_decompose": depth_before,
        "depth_after_decompose": depth_after,
    }


# =====================================================================
# ROW PROCESSING
# =====================================================================

def _simulate_row(
    row: pd.Series | Dict[str, Any],
    measure_all_if_missing: bool,
    sim: AerSimulator,
) -> Dict[str, Any]:
    """
    Simulate both logical and native circuits for one row.

    Any logical/native failure is recorded independently so one side can still
    succeed even if the other fails.
    """
    outputs: Dict[str, Any] = {
        "ideal_result_logical": "",
        "ideal_result_native": "",
        "ideal_result_logical_error": "",
        "ideal_result_native_error": "",
        "logical_num_qubits": pd.NA,
        "logical_depth_before_decompose": pd.NA,
        "logical_depth_after_decompose": pd.NA,
        "logical_sim_status": "FAILED",
        "native_num_qubits": pd.NA,
        "native_depth_before_decompose": pd.NA,
        "native_depth_after_decompose": pd.NA,
        "native_sim_status": "FAILED",
    }

    # -------------------------------------------------------------
    # Validate shots once. If shots are invalid, both sides fail.
    # -------------------------------------------------------------
    try:
        shots = _safe_parse_shots(row["shots"])
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        outputs["ideal_result_logical_error"] = msg
        outputs["ideal_result_native_error"] = msg
        return outputs

    # -------------------------------------------------------------
    # Logical circuit simulation
    # -------------------------------------------------------------
    try:
        logical_text, logical_format = _get_circuit_text_and_format(row, "circuit")
        logical_info = _simulate_single_circuit_cell(
            circuit_text=logical_text,
            circuit_format=logical_format,
            shots=shots,
            measure_all=measure_all_if_missing,
            sim=sim,
            seed=_row_seed(row, "logical"),
        )
        outputs["ideal_result_logical"] = logical_info["result_json"]
        outputs["logical_num_qubits"] = logical_info["num_qubits"]
        outputs["logical_depth_before_decompose"] = logical_info["depth_before_decompose"]
        outputs["logical_depth_after_decompose"] = logical_info["depth_after_decompose"]
        outputs["logical_sim_status"] = "SUCCESS"
    except Exception as exc:
        outputs["ideal_result_logical_error"] = f"{type(exc).__name__}: {exc}"

    # -------------------------------------------------------------
    # Native circuit simulation
    # -------------------------------------------------------------
    try:
        native_text, native_format = _get_circuit_text_and_format(row, "executed_circuit")
        native_info = _simulate_single_circuit_cell(
            circuit_text=native_text,
            circuit_format=native_format,
            shots=shots,
            measure_all=measure_all_if_missing,
            sim=sim,
            seed=_row_seed(row, "native"),
        )
        outputs["ideal_result_native"] = native_info["result_json"]
        outputs["native_num_qubits"] = native_info["num_qubits"]
        outputs["native_depth_before_decompose"] = native_info["depth_before_decompose"]
        outputs["native_depth_after_decompose"] = native_info["depth_after_decompose"]
        outputs["native_sim_status"] = "SUCCESS"
    except Exception as exc:
        outputs["ideal_result_native_error"] = f"{type(exc).__name__}: {exc}"

    return outputs


# =====================================================================
# MULTIPROCESSING HELPERS
# =====================================================================

def _init_worker(method: str, measure_all_if_missing: bool) -> None:
    """
    Initialize one AerSimulator instance per worker process.
    """
    global _WORKER_SIM, _WORKER_MEASURE_ALL_IF_MISSING
    _WORKER_SIM = AerSimulator(method=method)
    _WORKER_MEASURE_ALL_IF_MISSING = measure_all_if_missing


def _worker_simulate_row(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker-side wrapper for row simulation.
    """
    global _WORKER_SIM, _WORKER_MEASURE_ALL_IF_MISSING
    if _WORKER_SIM is None:
        raise RuntimeError("Worker AerSimulator was not initialized.")
    return _simulate_row(
        row=row_dict,
        measure_all_if_missing=_WORKER_MEASURE_ALL_IF_MISSING,
        sim=_WORKER_SIM,
    )


def _iter_row_dicts(df: pd.DataFrame) -> Iterable[Dict[str, Any]]:
    """
    Convert dataframe rows into plain dictionaries for multiprocessing.
    """
    for _, row in df.iterrows():
        yield row.to_dict()


def _compute_pool_chunksize(n_rows: int, jobs: int) -> int:
    """
    Choose a moderate multiprocessing chunksize.
    """
    if n_rows <= 0:
        return 1
    approx = n_rows // max(1, jobs * 8)
    return max(1, min(64, approx if approx > 0 else 1))


# =====================================================================
# OUTPUT DTYPE NORMALIZATION
# =====================================================================

def _normalize_output_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize output dtypes so all chunks share the same schema.
    """
    out = df.copy()

    for col in STRING_OUTPUT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("")

    for col in INT_OUTPUT_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    return out


# =====================================================================
# CHUNK PROCESSING
# =====================================================================

def _process_chunk_sequential(
    df: pd.DataFrame,
    measure_all_if_missing: bool,
    method: str,
) -> pd.DataFrame:
    """
    Sequential processing path for one dataframe chunk.
    """
    _validate_required_columns(df)

    sim = AerSimulator(method=method)

    row_outputs: List[Dict[str, Any]] = []
    n_rows = len(df)
    logical_ok = 0
    native_ok = 0

    for _, row in df.iterrows():
        result_fields = _simulate_row(
            row=row,
            measure_all_if_missing=measure_all_if_missing,
            sim=sim,
        )
        row_outputs.append(result_fields)

        if result_fields["logical_sim_status"] == "SUCCESS":
            logical_ok += 1
        if result_fields["native_sim_status"] == "SUCCESS":
            native_ok += 1

    out = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(row_outputs).reset_index(drop=True)],
        axis=1,
    )
    out = _normalize_output_dtypes(out)

    print(
        f"[INFO] Chunk summary | "
        f"logical success: {logical_ok}/{n_rows} | "
        f"native success: {native_ok}/{n_rows}",
        flush=True,
    )

    return out


def _process_chunk_parallel(
    df: pd.DataFrame,
    measure_all_if_missing: bool,
    method: str,
    jobs: int,
) -> pd.DataFrame:
    """
    Parallel processing path for one dataframe chunk.
    """
    _validate_required_columns(df)

    df = df.copy()
    row_dicts = list(_iter_row_dicts(df))
    n_rows = len(row_dicts)

    if n_rows == 0:
        return _normalize_output_dtypes(df)

    print(f"[INFO] Starting parallel chunk processing with {jobs} worker(s)...", flush=True)

    with Pool(
        processes=jobs,
        initializer=_init_worker,
        initargs=(method, measure_all_if_missing),
        maxtasksperchild=50,
    ) as pool:
        results = list(
            pool.imap(
                _worker_simulate_row,
                row_dicts,
                chunksize=_compute_pool_chunksize(n_rows, jobs),
            )
        )

    results_df = pd.DataFrame(results)
    out = pd.concat(
        [df.reset_index(drop=True), results_df.reset_index(drop=True)],
        axis=1,
    )
    out = _normalize_output_dtypes(out)

    logical_ok = int((out["logical_sim_status"] == "SUCCESS").sum())
    native_ok = int((out["native_sim_status"] == "SUCCESS").sum())

    print(
        f"[INFO] Chunk summary | "
        f"logical success: {logical_ok}/{n_rows} | "
        f"native success: {native_ok}/{n_rows}",
        flush=True,
    )

    return out


def _process_chunk(
    df: pd.DataFrame,
    measure_all_if_missing: bool,
    method: str,
    jobs: int,
) -> pd.DataFrame:
    """
    Dispatch one chunk to sequential or parallel processing.
    """
    if jobs <= 1:
        return _process_chunk_sequential(
            df=df,
            measure_all_if_missing=measure_all_if_missing,
            method=method,
        )

    return _process_chunk_parallel(
        df=df,
        measure_all_if_missing=measure_all_if_missing,
        method=method,
        jobs=jobs,
    )


# =====================================================================
# PARQUET STREAMING WRITER
# =====================================================================

class SafeParquetChunkWriter:
    """
    Append chunked pandas DataFrames into ONE final Parquet file.

    This avoids temporary part files and avoids a final global merge.
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
                df[col] = pd.NA
        return df[self.columns].copy()

    def write(self, df: pd.DataFrame) -> None:
        """
        Write one processed chunk into the final Parquet file.
        """
        if df is None or df.empty:
            return

        df = _normalize_output_dtypes(df)

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
        df2 = _normalize_output_dtypes(df2)
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
    3. chooses chunked or all-at-once mode
    4. simulates logical and native circuits
    5. writes the final Parquet output
    6. prints a summary report
    """
    parser = argparse.ArgumentParser(
        description=(
            "Simulate ideal logical and native results from a normalized, "
            "already-expanded Parquet circuit dataset."
        )
    )
    parser.add_argument("--input", required=True, help="Input Parquet file")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    parser.add_argument(
        "--measure-all-if-missing",
        action="store_true",
        help="Auto-insert measurements if a circuit has none.",
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
        "--method",
        default="automatic",
        choices=["automatic", "statevector", "density_matrix", "matrix_product_state", "stabilizer"],
        help="Aer simulation method",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Number of worker processes. Use 0 for automatic safe choice.",
    )
    parser.add_argument(
        "--usecols",
        default=None,
        help=(
            "Optional restricted input columns to load. Must still include id, shots, "
            "and either updated best_for_sim fields or older qasm2_norm fields."
        ),
    )
    parser.add_argument(
        "--compression",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"],
        default="zstd",
        help="Parquet compression codec.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed the ideal-distribution sampling per circuit (reproducible "
             "reruns). Default: unseeded, as in the original build.",
    )
    args = parser.parse_args()
    if args.seed is not None:
        os.environ[SEED_ENV] = str(args.seed)

    jobs = _recommended_jobs(args.jobs)

    if jobs < 1:
        raise ValueError("--jobs must resolve to >= 1")

    detected_cpus = cpu_count()
    if jobs > detected_cpus:
        print(
            f"[INFO] Requested jobs ({jobs}) exceed detected CPU count "
            f"({detected_cpus}).",
            flush=True,
        )

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    requested_usecols = parse_csv_list_arg(args.usecols)
    required_base = {"id", "shots"}

    if requested_usecols is not None:
        if not required_base.issubset(set(requested_usecols)):
            raise SystemExit(
                f"--usecols is missing required base column(s): "
                f"{sorted(required_base - set(requested_usecols))}"
            )

    input_cols = set(get_input_columns(args.input))

    missing_base = [col for col in required_base if col not in input_cols]
    if missing_base:
        raise ValueError(f"Missing required base column(s): {missing_base}")

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

    total_input_rows = get_total_rows(args.input)
    effective_total_rows = min(total_input_rows, args.max_rows) if args.max_rows is not None else total_input_rows

    start_time = time.time()
    total_rows_processed = 0
    total_logical_ok = 0
    total_native_ok = 0

    compression = None if args.compression == "none" else args.compression
    writer: Optional[SafeParquetChunkWriter] = None

    try:
        print("[INFO] Ideal simulation started", flush=True)
        print(f"[INFO] Input file:                {args.input}", flush=True)
        print(f"[INFO] Output file:               {args.output}", flush=True)
        print(f"[INFO] Input rows:                {total_input_rows}", flush=True)
        print(f"[INFO] Rows to process:           {effective_total_rows}", flush=True)
        print(f"[INFO] Chunksize:                 {args.chunksize if args.chunksize > 0 else 'all-at-once'}", flush=True)
        print(f"[INFO] Simulation method:         {args.method}", flush=True)
        print(f"[INFO] Worker processes (--jobs): {jobs}", flush=True)
        print(f"[INFO] Compression:               {args.compression}", flush=True)
        print(f"[INFO] QASM3 available:           {_qasm3_loads is not None}", flush=True)
        print(f"[INFO] QASM2 custom count:        {len(QASM2_CUSTOM_INSTRUCTIONS)}", flush=True)

        # ---------------------------------------------------------
        # Chunked / streaming mode
        # ---------------------------------------------------------
        if args.chunksize and args.chunksize > 0:
            writer = SafeParquetChunkWriter(args.output, compression=args.compression)
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

                out = _process_chunk(
                    df=chunk.copy(),
                    measure_all_if_missing=args.measure_all_if_missing,
                    method=args.method,
                    jobs=jobs,
                )

                writer.write(out)

                n_rows = len(out)
                rows_seen += len(chunk)
                total_rows_processed += n_rows
                total_logical_ok += int((out["logical_sim_status"] == "SUCCESS").sum())
                total_native_ok += int((out["native_sim_status"] == "SUCCESS").sum())

                print(
                    f"[INFO] Chunk {chunk_idx}/{n_chunks} done | "
                    f"input rows read so far: {rows_seen}/{effective_total_rows} | "
                    f"written rows so far: {total_rows_processed} | "
                    f"logical ok so far: {total_logical_ok}/{total_rows_processed} | "
                    f"native ok so far: {total_native_ok}/{total_rows_processed}",
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

            if args.max_rows is not None:
                df = df.head(args.max_rows).copy()

            out = _process_chunk(
                df=df.copy(),
                measure_all_if_missing=args.measure_all_if_missing,
                method=args.method,
                jobs=jobs,
            )
            out = _normalize_output_dtypes(out)

            out.to_parquet(
                args.output,
                index=False,
                compression=compression,
            )

            total_rows_processed = len(out)
            total_logical_ok = int((out["logical_sim_status"] == "SUCCESS").sum())
            total_native_ok = int((out["native_sim_status"] == "SUCCESS").sum())

        duration = time.time() - start_time
        out_size = os.path.getsize(args.output) if os.path.exists(args.output) else 0

        print("\n========== FINAL SUMMARY ==========", flush=True)
        print(f"Saved output to:           {args.output}", flush=True)
        print(f"Rows processed:            {total_rows_processed}", flush=True)
        print(f"Logical simulations ok:    {total_logical_ok}/{total_rows_processed}", flush=True)
        print(f"Native simulations ok:     {total_native_ok}/{total_rows_processed}", flush=True)
        print(f"Output size bytes:         {out_size}", flush=True)
        print(f"Execution time (s):        {duration:.2f}", flush=True)
        print("Stage 02 now assumes stage 01 already expanded batch jobs.", flush=True)
        print("Each row is expected to contain one logical circuit and one native circuit.", flush=True)
        print("===================================\n", flush=True)

    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()