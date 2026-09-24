#!/usr/bin/env python3
"""
01_parse_and_normalize_both_circuits.py
=======================================

PURPOSE
-------
Parse and normalize BOTH:
    - circuit
    - executed_circuit

from a Parquet file, and also align/expand the real-system:
    - result

UPDATED DESIGN
--------------
This version expands batch jobs BEFORE normalization.

That means the workflow is now:

1. Read one raw parent row
2. Decode raw batch structure for:
      - circuit
      - executed_circuit
      - result
3. Align singleton vs batch payloads
4. Expand into one row per sub-job
5. Normalize each expanded single circuit row
6. Normalize each expanded single result row
7. Write directly into one Parquet file

WHY THIS CHANGE
---------------
This is better for downstream stages because:

- stage 02 should receive exactly one circuit per row
- stage 03, 04, 06, 07, and later ML stages also become simpler
- trapped_ion_20q/OpenQASM3 batch rows are easier to debug after early expansion
- row lineage is preserved early with:
      parent_id, sub_id, batch_index, batch_size, is_batch_job

IMPORTANT
---------
- This script does NOT simulate
- This script does NOT compute ideal results
- This script DOES expand raw batch rows before normalization

DOWNSTREAM RECOMMENDATION
-------------------------
Later stages should prefer:
    <circuit_col>_best_for_sim
    <executed_col>_best_for_sim

This script guarantees these are single-circuit fields per row.

OUTPUT COLUMNS ADDED
--------------------
For logical circuit:
- circuit_format
- circuit_parse_ok
- circuit_parse_error
- circuit_kind
- circuit_n
- circuit_source_syntax
- circuit_parser_used
- circuit_norm_format
- circuit_qasm2_norm
- circuit_norm_text
- circuit_best_for_sim
- circuit_best_for_sim_format

For executed circuit:
- executed_circuit_format
- executed_circuit_parse_ok
- executed_circuit_parse_error
- executed_circuit_kind
- executed_circuit_n
- executed_circuit_source_syntax
- executed_circuit_parser_used
- executed_circuit_norm_format
- executed_circuit_qasm2_norm
- executed_circuit_norm_text
- executed_circuit_best_for_sim
- executed_circuit_best_for_sim_format

For real-system result:
- result_kind
- result_n
- result_expanded
- result_parse_error

Batch expansion:
- parent_id
- sub_id
- batch_index
- batch_size
- is_batch_job
- batch_len_mismatch_circuit_vs_executed_vs_result

Raw batch split diagnostics:
- circuit_batch_raw_kind
- circuit_batch_raw_n
- circuit_batch_raw_error
- executed_circuit_batch_raw_kind
- executed_circuit_batch_raw_n
- executed_circuit_batch_raw_error
- result_batch_raw_kind
- result_batch_raw_n
- result_batch_raw_error

USAGE
-----
python3 01_parse_and_normalize_both_circuits.py \
  --input /path/to/input.parquet \
  --output /path/to/output.parquet \
  --chunksize 2000 \
  --jobs 8 \
  --drop-mismatch-rows \
  --compression zstd
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from qiskit import QuantumCircuit, qasm2
from qiskit.qasm2 import dumps as qasm2_dumps

# ---------------------------------------------------------------------
# Optional QASM3 support
# ---------------------------------------------------------------------
_qasm3_loads = None
_qasm3_dumps = None
try:
    from qiskit.qasm3 import loads as _qasm3_loads  # type: ignore
except Exception:
    _qasm3_loads = None

try:
    from qiskit.qasm3 import dumps as _qasm3_dumps  # type: ignore
except Exception:
    _qasm3_dumps = None

# ---------------------------------------------------------------------
# Optional gate-class imports for broad QASM2 custom-instruction support
# ---------------------------------------------------------------------
_OPTIONAL_GATE_CLASSES: Dict[str, Any] = {}


def _try_import_gate(module_path: str, class_name: str) -> Optional[Any]:
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

# ---------------------------------------------------------------------
# Syntax-detection regexes and heuristics
# ---------------------------------------------------------------------
_QASM2_HEADER_RE = re.compile(r"^\s*OPENQASM\s+2(\.0)?\s*;\s*", re.IGNORECASE)
_QASM3_HEADER_RE = re.compile(r"^\s*OPENQASM\s+3(\.0)?\s*;\s*", re.IGNORECASE)

_QASM3_HINTS = (
    'include "stdgates.inc"',
    "qubit[",
    "bit[",
    "input ",
    "output ",
    "let ",
    "const ",
    "def ",
)

_QASM2_HINTS = (
    'include "qelib1.inc"',
    "qreg ",
    "creg ",
    "measure ",
    "gate ",
    "opaque ",
    "barrier ",
    "if(",
    "reset ",
)

_EXTRA_QASM2_GATE_DEFS = r"""
// --- injected explicit gate definitions for downstream re-import safety ---
gate rxx(theta) a,b {
    h a;
    h b;
    cx a,b;
    rz(theta) b;
    cx a,b;
    h a;
    h b;
}

gate ryy(theta) a,b {
    rx(pi/2) a;
    rx(pi/2) b;
    cx a,b;
    rz(theta) b;
    cx a,b;
    rx(-pi/2) a;
    rx(-pi/2) b;
}

gate rzz(theta) a,b {
    cx a,b;
    rz(theta) b;
    cx a,b;
}

gate rzx(theta) a,b {
    h b;
    cx a,b;
    rz(theta) b;
    cx a,b;
    h b;
}
"""

# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
def _is_nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x))


def _safe_json_loads(s: str) -> Tuple[bool, Any]:
    try:
        return True, json.loads(s)
    except Exception:
        return False, None


def _safe_literal_eval(s: str) -> Tuple[bool, Any]:
    try:
        return True, ast.literal_eval(s)
    except Exception:
        return False, None


def _loads_json_or_literal(s: str) -> Tuple[bool, Any]:
    ok, obj = _safe_json_loads(s)
    if ok:
        return True, obj
    ok, obj = _safe_literal_eval(s)
    if ok:
        return True, obj
    return False, None


def _extract_payload(obj: Any) -> Any:
    if isinstance(obj, dict):
        for k in (
            "qasm",
            "openqasm",
            "circuit",
            "executed_circuit",
            "data",
            "payload",
            "program",
            "text",
            "source",
            "body",
        ):
            if k in obj:
                return obj[k]
        for k in ("circuits", "items", "list", "programs"):
            if k in obj:
                return obj[k]
    return obj


def _coerce_singleton_list_payload(obj: Any) -> Any:
    if isinstance(obj, list) and len(obj) == 1:
        return obj[0]
    return obj


def _unwrap_json_repeatedly(obj: Any, max_depth: int = 10) -> Any:
    cur = obj

    for _ in range(max_depth):
        if not isinstance(cur, str):
            return cur

        s = cur.strip()
        if not s:
            return ""

        if s[0] in ['"', "{", "["]:
            ok, parsed = _safe_json_loads(s)
            if ok:
                cur = parsed
                continue

        if s[0] in ["{", "["]:
            ok, parsed = _safe_literal_eval(s)
            if ok:
                cur = parsed
                continue

        return s

    return cur


def _stringify_payload(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False)
    return str(obj)


def parse_csv_list_arg(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None or str(raw).strip() == "":
        return None
    return [x.strip() for x in str(raw).split(",") if x.strip()]


# ---------------------------------------------------------------------
# Format / syntax detection
# ---------------------------------------------------------------------
def looks_like_qasm3_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip().lower()
    if not t:
        return False
    if _QASM3_HEADER_RE.match(t):
        return True
    return any(h in t for h in _QASM3_HINTS)


def looks_like_qasm2_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip().lower()
    if not t:
        return False
    if _QASM2_HEADER_RE.match(t):
        return True
    return any(h in t for h in _QASM2_HINTS)


def detect_format(raw: Any) -> str:
    if _is_nan(raw):
        return "empty"

    s = str(raw).strip()
    if not s:
        return "empty"

    if s[0] == "[":
        ok, _ = _safe_json_loads(s)
        if ok:
            return "json_list"
        ok, _ = _safe_literal_eval(s)
        if ok:
            return "py_list"
        return "bracketed_unknown"

    if s[0] == "{":
        ok, _ = _safe_json_loads(s)
        if ok:
            return "json_dict"
        ok, _ = _safe_literal_eval(s)
        if ok:
            return "py_dict"
        return "braced_unknown"

    if s.startswith("gAS"):
        return "b64_pickle_like"

    t = s.lstrip()
    if _QASM3_HEADER_RE.match(t):
        return "qasm3_direct"
    if _QASM2_HEADER_RE.match(t):
        return "qasm2_direct"

    if looks_like_qasm3_text(t):
        return "qasm3_like"

    if looks_like_qasm2_text(t):
        return "qasm2_like"

    return "other_string"


def infer_source_syntax(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "empty"
    if looks_like_qasm3_text(text):
        return "qasm3"
    if looks_like_qasm2_text(text):
        return "qasm2"
    return "unknown"


# ---------------------------------------------------------------------
# QASM2 custom-instruction registry
# ---------------------------------------------------------------------
def _build_qasm2_custom_instructions() -> Tuple[Any, ...]:
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


def _qasm2_loads_with_customs(text: str, extra_customs: Optional[List[Any]] = None) -> QuantumCircuit:
    customs = list(QASM2_CUSTOM_INSTRUCTIONS)
    if extra_customs:
        customs.extend(extra_customs)
    return qasm2.loads(text, custom_instructions=tuple(customs))


def _normalize_qasm2_includes_for_retry(qasm_text: str) -> str:
    text = qasm_text

    if not _QASM2_HEADER_RE.match(text.lstrip()) and looks_like_qasm2_text(text):
        text = "OPENQASM 2.0;\n" + text

    text = re.sub(
        r'^\s*include\s+"[^"]+"\s*;\s*$',
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    lines = text.splitlines(True)
    out = []
    inserted = False

    for line in lines:
        out.append(line)
        if not inserted and _QASM2_HEADER_RE.match(line.strip()):
            out.append('include "qelib1.inc";\n')
            inserted = True

    if not inserted:
        return 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' + text

    return "".join(out)


def _needs_qasm2_retry_cleanup(qasm_text: str) -> bool:
    t = qasm_text.lower()
    return any(tok in t for tok in (
        "rxx(",
        "ryy(",
        "rzz(",
        "rzx(",
        'include "stdgates.inc"',
        'include "',
    ))


def _extract_undefined_gate_names(parse_error: str) -> List[str]:
    if not isinstance(parse_error, str):
        return []
    return re.findall(r"'([A-Za-z][A-Za-z0-9_]*)'\s+is not defined in this scope", parse_error)


def _build_extra_customs_for_undefined_names(names: List[str]) -> List[Any]:
    out: List[Any] = []

    name_to_spec = {
        "rxx": ("rxx", 1, 2, _OPTIONAL_GATE_CLASSES.get("RXXGate")),
        "ryy": ("ryy", 1, 2, _OPTIONAL_GATE_CLASSES.get("RYYGate")),
        "rzz": ("rzz", 1, 2, _OPTIONAL_GATE_CLASSES.get("RZZGate")),
        "rzx": ("rzx", 1, 2, _OPTIONAL_GATE_CLASSES.get("RZXGate")),
        "ecr": ("ecr", 0, 2, _OPTIONAL_GATE_CLASSES.get("ECRGate")),
        "iswap": ("iswap", 0, 2, _OPTIONAL_GATE_CLASSES.get("iSwapGate")),
        "dcx": ("dcx", 0, 2, _OPTIONAL_GATE_CLASSES.get("DCXGate")),
    }

    for name in names:
        spec = name_to_spec.get(name.lower())
        if spec is None:
            continue

        gate_name, num_params, num_qubits, ctor = spec
        if ctor is None:
            continue

        out.append(
            qasm2.CustomInstruction(
                name=gate_name,
                num_params=num_params,
                num_qubits=num_qubits,
                constructor=ctor,
                builtin=True,
            )
        )

    return out


# ---------------------------------------------------------------------
# Raw batch expansion helpers
# ---------------------------------------------------------------------
@dataclass
class RawBatchInfo:
    kind: str
    n: int
    items: List[str]
    error: str


def _unwrap_payload_for_batch(raw: Any) -> Any:
    if _is_nan(raw):
        return None

    cur = _unwrap_json_repeatedly(raw)

    for _ in range(10):
        nxt = _extract_payload(cur)
        nxt = _unwrap_json_repeatedly(nxt)
        if nxt is cur:
            break
        cur = nxt

    return cur


def _raw_circuit_cell_to_items(raw: Any) -> RawBatchInfo:
    if _is_nan(raw):
        return RawBatchInfo("empty", 0, [], "Empty/NaN cell")

    try:
        payload = _unwrap_payload_for_batch(raw)

        if payload is None:
            return RawBatchInfo("empty", 0, [], "Empty/NaN cell")

        if isinstance(payload, list):
            items: List[str] = []
            for i, item in enumerate(payload):
                item_payload = _unwrap_payload_for_batch(item)
                item_payload = _coerce_singleton_list_payload(item_payload)

                if isinstance(item_payload, list):
                    raise ValueError(f"Nested list encountered at item {i}")

                text = _stringify_payload(item_payload).strip()
                if not text:
                    raise ValueError(f"Empty batch item at index {i}")
                items.append(text)

            if not items:
                raise ValueError("Decoded list is empty")

            return RawBatchInfo("list", len(items), items, "")

        text = _stringify_payload(_coerce_singleton_list_payload(payload)).strip()
        if not text:
            raise ValueError("Empty string payload")

        return RawBatchInfo("single", 1, [text], "")

    except Exception as e:
        s = "" if _is_nan(raw) else str(raw)
        if s.strip():
            return RawBatchInfo("single", 1, [s], f"{type(e).__name__}: {e}")
        return RawBatchInfo("unknown", 0, [], f"{type(e).__name__}: {e}")


def _parse_counts_dict(obj: Dict[Any, Any]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in obj.items()}


def _raw_result_cell_to_items(raw: Any) -> RawBatchInfo:
    if _is_nan(raw):
        return RawBatchInfo("empty", 0, [], "Empty/NaN result")

    try:
        payload = _unwrap_payload_for_batch(raw)
        payload = _coerce_singleton_list_payload(payload)

        if isinstance(payload, str):
            ok, parsed = _loads_json_or_literal(payload)
            if ok:
                payload = parsed

        if isinstance(payload, dict):
            d = _parse_counts_dict(payload)
            return RawBatchInfo("single", 1, [json.dumps(d, ensure_ascii=False, sort_keys=True)], "")

        if isinstance(payload, list):
            if len(payload) == 1 and isinstance(payload[0], dict):
                d = _parse_counts_dict(payload[0])
                return RawBatchInfo("single", 1, [json.dumps(d, ensure_ascii=False, sort_keys=True)], "")

            items: List[str] = []
            for i, item in enumerate(payload):
                if isinstance(item, str):
                    ok, parsed = _loads_json_or_literal(item)
                    if ok:
                        item = parsed

                if not isinstance(item, dict):
                    raise ValueError(f"Result list contains non-dict element at index {i}")

                d = _parse_counts_dict(item)
                items.append(json.dumps(d, ensure_ascii=False, sort_keys=True))

            if not items:
                raise ValueError("Decoded result list is empty")

            return RawBatchInfo("list", len(items), items, "")

        raise ValueError("Result cell must decode to dict or list[dict]")

    except Exception as e:
        s = "" if _is_nan(raw) else str(raw)
        if s.strip():
            return RawBatchInfo("single", 1, [s], f"{type(e).__name__}: {e}")
        return RawBatchInfo("unknown", 0, [], f"{type(e).__name__}: {e}")


def _expand_raw_chunk(
    df: pd.DataFrame,
    circuit_col: str,
    executed_col: str,
    result_col: str,
) -> pd.DataFrame:
    expanded_rows: List[Dict[str, Any]] = []
    cols = list(df.columns)

    for row_tuple in df.itertuples(index=False, name=None):
        row = dict(zip(cols, row_tuple))
        parent_id = row.get("id", "unknown_id")

        logical_raw = _raw_circuit_cell_to_items(row.get(circuit_col))
        native_raw = _raw_circuit_cell_to_items(row.get(executed_col))
        result_raw = _raw_result_cell_to_items(row.get(result_col))

        row["circuit_batch_raw_kind"] = logical_raw.kind
        row["circuit_batch_raw_n"] = logical_raw.n
        row["circuit_batch_raw_error"] = logical_raw.error

        row["executed_circuit_batch_raw_kind"] = native_raw.kind
        row["executed_circuit_batch_raw_n"] = native_raw.n
        row["executed_circuit_batch_raw_error"] = native_raw.error

        row["result_batch_raw_kind"] = result_raw.kind
        row["result_batch_raw_n"] = result_raw.n
        row["result_batch_raw_error"] = result_raw.error

        sizes = []
        if logical_raw.n > 1:
            sizes.append(logical_raw.n)
        if native_raw.n > 1:
            sizes.append(native_raw.n)
        if result_raw.n > 1:
            sizes.append(result_raw.n)

        mismatch = len(set(sizes)) > 1 if sizes else False
        row["batch_len_mismatch_circuit_vs_executed_vs_result"] = mismatch

        if mismatch:
            new_row = dict(row)
            new_row["parent_id"] = parent_id
            new_row["sub_id"] = f"{parent_id}_0"
            new_row["batch_index"] = 0
            new_row["batch_size"] = max(
                logical_raw.n,
                native_raw.n,
                result_raw.n,
                1,
            )
            new_row["is_batch_job"] = (
                logical_raw.kind == "list" or
                native_raw.kind == "list" or
                result_raw.kind == "list"
            )
            expanded_rows.append(new_row)
            continue

        batch_size = max(logical_raw.n, native_raw.n, result_raw.n, 1)
        is_batch_job = batch_size > 1

        logical_items = list(logical_raw.items)
        native_items = list(native_raw.items)
        result_items = list(result_raw.items)

        if logical_items and len(logical_items) == 1 and batch_size > 1:
            logical_items = logical_items * batch_size
        if native_items and len(native_items) == 1 and batch_size > 1:
            native_items = native_items * batch_size
        if result_items and len(result_items) == 1 and batch_size > 1:
            result_items = result_items * batch_size

        valid_lengths = [len(x) for x in (logical_items, native_items, result_items) if x]
        if any(n != batch_size for n in valid_lengths):
            new_row = dict(row)
            new_row["batch_len_mismatch_circuit_vs_executed_vs_result"] = True
            new_row["parent_id"] = parent_id
            new_row["sub_id"] = f"{parent_id}_0"
            new_row["batch_index"] = 0
            new_row["batch_size"] = batch_size
            new_row["is_batch_job"] = is_batch_job
            expanded_rows.append(new_row)
            continue

        for i in range(batch_size):
            new_row = dict(row)
            new_row["parent_id"] = parent_id
            new_row["sub_id"] = f"{parent_id}_{i}"
            new_row["batch_index"] = i
            new_row["batch_size"] = batch_size
            new_row["is_batch_job"] = is_batch_job

            if logical_items:
                new_row[circuit_col] = logical_items[i]
            if native_items:
                new_row[executed_col] = native_items[i]
            if result_items:
                new_row[result_col] = result_items[i]

            expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)


# ---------------------------------------------------------------------
# Circuit parsing after expansion
# ---------------------------------------------------------------------
def _try_unpickle_base64(s: str) -> QuantumCircuit:
    data = base64.b64decode(s.encode("ascii"), validate=False)
    obj = pickle.loads(data)
    if not isinstance(obj, QuantumCircuit):
        raise TypeError(f"Unpickled object is not QuantumCircuit (got {type(obj)})")
    return obj


def _parse_qasm_auto(qasm_text: str) -> Tuple[QuantumCircuit, str, str]:
    if not isinstance(qasm_text, str) or not qasm_text.strip():
        raise ValueError("Empty circuit text")

    text = qasm_text.strip()
    source_syntax = infer_source_syntax(text)
    errs: List[str] = []

    if source_syntax == "qasm3":
        if _qasm3_loads is not None:
            try:
                return _qasm3_loads(text), "qasm3", source_syntax
            except Exception as e:
                errs.append(f"QASM3 parse failed: {e}")
        else:
            errs.append("QASM3 parser unavailable")

        try:
            cleaned = _normalize_qasm2_includes_for_retry(text)
            return _qasm2_loads_with_customs(cleaned), "qasm2_fallback_from_qasm3", source_syntax
        except Exception as e:
            errs.append(f"QASM2 fallback failed: {e}")

        raise ValueError(" ; ".join(errs))

    if source_syntax == "qasm2":
        try:
            return _qasm2_loads_with_customs(text), "qasm2_customs", source_syntax
        except Exception as e1:
            errs.append(f"QASM2 custom parse failed: {e1}")

            missing_names = _extract_undefined_gate_names(str(e1))
            if missing_names:
                try:
                    return _qasm2_loads_with_customs(
                        text,
                        extra_customs=_build_extra_customs_for_undefined_names(missing_names),
                    ), "qasm2_customs_dynamic_retry", source_syntax
                except Exception as e_dynamic:
                    errs.append(f"QASM2 dynamic-custom retry failed: {e_dynamic}")

        if _needs_qasm2_retry_cleanup(text):
            try:
                cleaned = _normalize_qasm2_includes_for_retry(text)
                return _qasm2_loads_with_customs(cleaned), "qasm2_customs_retry_cleaned", source_syntax
            except Exception as e2:
                errs.append(f"QASM2 cleaned retry failed: {e2}")

        if _qasm3_loads is not None:
            try:
                return _qasm3_loads(text), "qasm3_fallback", source_syntax
            except Exception as e3:
                errs.append(f"QASM3 fallback failed: {e3}")

        raise ValueError(" ; ".join(errs))

    try:
        return _qasm2_loads_with_customs(text), "qasm2_customs_guess", source_syntax
    except Exception as e1:
        errs.append(f"QASM2 custom guessed parse failed: {e1}")

        missing_names = _extract_undefined_gate_names(str(e1))
        if missing_names:
            try:
                return _qasm2_loads_with_customs(
                    text,
                    extra_customs=_build_extra_customs_for_undefined_names(missing_names),
                ), "qasm2_customs_dynamic_guess_retry", source_syntax
            except Exception as e_dynamic:
                errs.append(f"QASM2 dynamic-custom guessed retry failed: {e_dynamic}")

    if _needs_qasm2_retry_cleanup(text) or looks_like_qasm2_text(text):
        try:
            cleaned = _normalize_qasm2_includes_for_retry(text)
            return _qasm2_loads_with_customs(cleaned), "qasm2_customs_guess_cleaned", source_syntax
        except Exception as e2:
            errs.append(f"QASM2 cleaned guessed retry failed: {e2}")

    if _qasm3_loads is not None:
        try:
            return _qasm3_loads(text), "qasm3_guess", source_syntax
        except Exception as e3:
            errs.append(f"QASM3 guessed parse failed: {e3}")

    raise ValueError(" ; ".join(errs) if errs else "No parser succeeded")


def parse_cell(raw: Any, trust_level: str) -> Tuple[QuantumCircuit, str, str]:
    if _is_nan(raw):
        raise ValueError("Empty/NaN cell")

    raw_unwrapped = _unwrap_json_repeatedly(raw)
    fmt_payload = _extract_payload(raw_unwrapped)
    fmt_payload = _coerce_singleton_list_payload(fmt_payload)

    if isinstance(fmt_payload, list):
        raise ValueError("List payload reached post-expansion normalization stage")

    if isinstance(fmt_payload, dict):
        fmt_payload = _extract_payload(fmt_payload)
        fmt_payload = _unwrap_json_repeatedly(fmt_payload)
        fmt_payload = _coerce_singleton_list_payload(fmt_payload)
        if isinstance(fmt_payload, list):
            raise ValueError("List payload reached post-expansion normalization stage")

    if isinstance(fmt_payload, str):
        s = fmt_payload.strip()
    else:
        s = _stringify_payload(fmt_payload).strip()

    if not s:
        raise ValueError("Empty string")

    if s.startswith("gAS"):
        if trust_level != "trusted":
            raise ValueError("Looks like base64 pickle; refused in safe mode (use --trust-level trusted)")
        qc = _try_unpickle_base64(s)
        return qc, "pickle_base64", "pickle_base64"

    return _parse_qasm_auto(s)


# ---------------------------------------------------------------------
# Circuit normalization
# ---------------------------------------------------------------------
@dataclass
class NormResult:
    parse_ok: bool
    parse_error: str
    fmt: str
    kind: str
    n: int
    source_syntax: str
    parser_used: str
    norm_format: str
    qasm2_norm: str
    norm_text: str
    best_for_sim: str
    best_for_sim_format: str


def _inject_extra_defs_into_qasm2_export(qasm_text: str) -> str:
    if not isinstance(qasm_text, str) or not qasm_text.strip():
        return qasm_text

    needed = False
    for gate_name in ("rxx", "ryy", "rzz", "rzx"):
        uses_gate = re.search(rf"\b{gate_name}\s*\(", qasm_text) is not None
        has_gate_def = re.search(rf"^\s*gate\s+{gate_name}\s*\(", qasm_text, flags=re.MULTILINE) is not None
        if uses_gate and not has_gate_def:
            needed = True
            break

    if not needed:
        return qasm_text

    lines = qasm_text.splitlines(True)
    out = []
    inserted = False

    for line in lines:
        out.append(line)
        if not inserted and re.match(r'^\s*include\s+"qelib1\.inc"\s*;\s*$', line.strip(), flags=re.IGNORECASE):
            out.append(_EXTRA_QASM2_GATE_DEFS + "\n")
            inserted = True

    if not inserted:
        out = []
        for line in lines:
            out.append(line)
            if not inserted and _QASM2_HEADER_RE.match(line.strip()):
                out.append(_EXTRA_QASM2_GATE_DEFS + "\n")
                inserted = True

    if not inserted:
        return _EXTRA_QASM2_GATE_DEFS + "\n" + qasm_text

    return "".join(out)


def _serialize_single_circuit(qc: QuantumCircuit, source_syntax: str) -> Tuple[str, str, str, str]:
    q2_text = ""
    q3_text = ""

    try:
        q2_text = qasm2_dumps(qc)
        q2_text = _inject_extra_defs_into_qasm2_export(q2_text)
    except Exception:
        q2_text = ""

    if _qasm3_dumps is not None:
        try:
            q3_text = _qasm3_dumps(qc)
        except Exception:
            q3_text = ""

    # Safer downstream policy after early expansion:
    # prefer QASM2 whenever export succeeds; keep QASM3 only as fallback.
    if q2_text:
        return "qasm2", q2_text, q2_text, "qasm2"
    if q3_text:
        return "qasm3", q3_text, "", "qasm3"

    return "unknown", "", "", "unknown"


def normalize_cell(raw: Any, trust_level: str) -> NormResult:
    fmt = detect_format(raw)

    if fmt == "empty":
        return NormResult(False, "Empty/NaN", fmt, "empty", 0, "empty", "", "unknown", "", "", "", "unknown")

    try:
        qc, parser_used, source_syntax = parse_cell(raw, trust_level=trust_level)
        norm_format, norm_text, qasm2_norm, best_for_sim_format = _serialize_single_circuit(qc, source_syntax)
        return NormResult(
            True,
            "",
            fmt,
            "single",
            1,
            source_syntax,
            parser_used,
            norm_format,
            qasm2_norm,
            norm_text,
            norm_text,
            best_for_sim_format,
        )
    except Exception as e:
        return NormResult(False, str(e), fmt, "unknown", 0, "unknown", "", "unknown", "", "", "", "unknown")


# ---------------------------------------------------------------------
# Result parsing / normalization
# ---------------------------------------------------------------------
@dataclass
class ResultNorm:
    kind: str
    n: int
    expanded: str


def _normalize_result_cell(raw: Any) -> ResultNorm:
    if _is_nan(raw):
        return ResultNorm("empty", 0, "")

    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return ResultNorm("empty", 0, "")

    obj = _unwrap_json_repeatedly(s)
    obj = _coerce_singleton_list_payload(obj)

    if isinstance(obj, str):
        ok, parsed = _loads_json_or_literal(obj)
        if ok:
            obj = parsed

    if isinstance(obj, dict):
        parsed = _parse_counts_dict(obj)
        return ResultNorm("single", 1, json.dumps(parsed, ensure_ascii=False, sort_keys=True))

    if isinstance(obj, list):
        if len(obj) == 1 and isinstance(obj[0], dict):
            parsed = _parse_counts_dict(obj[0])
            return ResultNorm("single", 1, json.dumps(parsed, ensure_ascii=False, sort_keys=True))

        raise ValueError("List payload reached post-expansion result normalization stage")

    raise ValueError("Result cell must decode to dict")


# ---------------------------------------------------------------------
# Parallel helpers
# ---------------------------------------------------------------------
def _normalize_cell_worker(args: Tuple[Any, str]) -> NormResult:
    raw, trust_level = args
    return normalize_cell(raw, trust_level=trust_level)


def _normalize_result_cell_worker(raw: Any) -> Tuple[ResultNorm, str]:
    try:
        return _normalize_result_cell(raw), ""
    except Exception as exc:
        return ResultNorm("unknown", 0, ""), f"{type(exc).__name__}: {exc}"


def _parallel_map(items: List[Any], func, jobs: int) -> List[Any]:
    if jobs <= 1 or len(items) == 0:
        return [func(x) for x in items]

    with Pool(processes=jobs) as pool:
        chunksize = max(1, min(128, len(items) // max(1, jobs * 4) or 1))
        return list(pool.map(func, items, chunksize=chunksize))


# ---------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------
def process_chunk(
    df: pd.DataFrame,
    circuit_col: str,
    executed_col: str,
    result_col: str,
    trust_level: str,
    jobs: int,
) -> pd.DataFrame:
    # Step 1: raw batch expansion before normalization
    expanded = _expand_raw_chunk(
        df.copy(),
        circuit_col=circuit_col,
        executed_col=executed_col,
        result_col=result_col,
    )

    # Step 2: normalize logical single rows
    circuit_items = [(x, trust_level) for x in expanded[circuit_col].tolist()]
    c = _parallel_map(circuit_items, _normalize_cell_worker, jobs)
    expanded[f"{circuit_col}_format"] = [r.fmt for r in c]
    expanded[f"{circuit_col}_parse_ok"] = [r.parse_ok for r in c]
    expanded[f"{circuit_col}_parse_error"] = [r.parse_error for r in c]
    expanded[f"{circuit_col}_kind"] = [r.kind for r in c]
    expanded[f"{circuit_col}_n"] = [r.n for r in c]
    expanded[f"{circuit_col}_source_syntax"] = [r.source_syntax for r in c]
    expanded[f"{circuit_col}_parser_used"] = [r.parser_used for r in c]
    expanded[f"{circuit_col}_norm_format"] = [r.norm_format for r in c]
    expanded[f"{circuit_col}_qasm2_norm"] = [r.qasm2_norm for r in c]
    expanded[f"{circuit_col}_norm_text"] = [r.norm_text for r in c]
    expanded[f"{circuit_col}_best_for_sim"] = [r.best_for_sim for r in c]
    expanded[f"{circuit_col}_best_for_sim_format"] = [r.best_for_sim_format for r in c]

    # Step 3: normalize native single rows
    executed_items = [(x, trust_level) for x in expanded[executed_col].tolist()]
    e = _parallel_map(executed_items, _normalize_cell_worker, jobs)
    expanded[f"{executed_col}_format"] = [r.fmt for r in e]
    expanded[f"{executed_col}_parse_ok"] = [r.parse_ok for r in e]
    expanded[f"{executed_col}_parse_error"] = [r.parse_error for r in e]
    expanded[f"{executed_col}_kind"] = [r.kind for r in e]
    expanded[f"{executed_col}_n"] = [r.n for r in e]
    expanded[f"{executed_col}_source_syntax"] = [r.source_syntax for r in e]
    expanded[f"{executed_col}_parser_used"] = [r.parser_used for r in e]
    expanded[f"{executed_col}_norm_format"] = [r.norm_format for r in e]
    expanded[f"{executed_col}_qasm2_norm"] = [r.qasm2_norm for r in e]
    expanded[f"{executed_col}_norm_text"] = [r.norm_text for r in e]
    expanded[f"{executed_col}_best_for_sim"] = [r.best_for_sim for r in e]
    expanded[f"{executed_col}_best_for_sim_format"] = [r.best_for_sim_format for r in e]

    # Step 4: normalize already-expanded single results
    result_pairs = _parallel_map(expanded[result_col].tolist(), _normalize_result_cell_worker, jobs)
    result_norms = [p[0] for p in result_pairs]
    result_errors = [p[1] for p in result_pairs]

    expanded[f"{result_col}_kind"] = [r.kind for r in result_norms]
    expanded[f"{result_col}_n"] = [r.n for r in result_norms]
    expanded[f"{result_col}_expanded"] = [r.expanded for r in result_norms]
    expanded[f"{result_col}_parse_error"] = result_errors

    return expanded


def _drop_mismatch_rows_if_requested(
    df: pd.DataFrame,
    drop_mismatch_rows: bool,
    label: str,
) -> tuple[pd.DataFrame, int]:
    if not drop_mismatch_rows:
        return df, 0

    if "batch_len_mismatch_circuit_vs_executed_vs_result" not in df.columns:
        return df, 0

    before = len(df)
    mask = df["batch_len_mismatch_circuit_vs_executed_vs_result"].fillna(False).astype(bool)
    out = df.loc[~mask].copy()
    removed = before - len(out)

    print(f"[INFO] Dropped mismatch rows in {label}: {removed}", flush=True)
    return out, removed


# ---------------------------------------------------------------------
# Parquet streaming output writer
# ---------------------------------------------------------------------
class SafeParquetChunkWriter:
    def __init__(self, output_path: str, compression: str = "zstd") -> None:
        self.output_path = output_path
        self.compression = compression
        self.writer: Optional[pq.ParquetWriter] = None
        self.schema: Optional[pa.Schema] = None
        self.columns: Optional[List[str]] = None

    def _align_df(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.columns is not None
        for col in self.columns:
            if col not in df.columns:
                df[col] = None
        return df[self.columns].copy()

    def write(self, df: pd.DataFrame) -> None:
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
        if self.writer is not None:
            self.writer.close()
            self.writer = None


# ---------------------------------------------------------------------
# Parquet input helpers
# ---------------------------------------------------------------------
def get_input_columns(input_path: str) -> List[str]:
    pf = pq.ParquetFile(input_path)
    return pf.schema_arrow.names


def get_total_rows(input_path: str) -> int:
    pf = pq.ParquetFile(input_path)
    return pf.metadata.num_rows


def iter_parquet_batches(
    input_path: str,
    batch_size: int,
    columns: Optional[List[str]] = None,
) -> Iterable[pd.DataFrame]:
    pf = pq.ParquetFile(input_path)

    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input Parquet file")
    ap.add_argument("--output", required=True, help="Output normalized Parquet file")
    ap.add_argument("--circuit-col", default="circuit", help="Logical circuit column name")
    ap.add_argument("--executed-col", default="executed_circuit", help="Executed circuit column name")
    ap.add_argument("--result-col", default="result", help="Real-system result column name")
    ap.add_argument("--chunksize", type=int, default=2000, help="0 = process all rows at once")
    ap.add_argument(
        "--trust-level",
        choices=["safe", "trusted"],
        default="safe",
        help="safe refuses pickle; trusted allows base64 pickle decoding",
    )
    ap.add_argument(
        "--drop-mismatch-rows",
        action="store_true",
        help="Drop rows where circuit/executed/result batch lengths mismatch.",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes for normalization. Use 1 for sequential mode.",
    )
    ap.add_argument(
        "--usecols",
        default=None,
        help="Optional comma-separated input columns to load. Must include required columns.",
    )
    ap.add_argument(
        "--compression",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"],
        default="zstd",
        help="Parquet compression codec.",
    )
    args = ap.parse_args()

    if _qasm3_loads is None:
        print("WARNING: qiskit.qasm3.loads not found; QASM3 rows may fail.", flush=True)
    else:
        print("[INFO] qiskit.qasm3.loads is available.", flush=True)

    print(f"[INFO] Number of built QASM2 custom instructions: {len(QASM2_CUSTOM_INSTRUCTIONS)}", flush=True)

    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")

    if args.jobs > cpu_count():
        print(f"WARNING: requested --jobs {args.jobs} exceeds detected CPU count {cpu_count()}.", flush=True)

    if not os.path.exists(args.input):
        raise SystemExit(f"Input Parquet not found: {args.input}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    requested_usecols = parse_csv_list_arg(args.usecols)
    required_cols = {args.circuit_col, args.executed_col, args.result_col}

    if requested_usecols is not None:
        missing_required = required_cols - set(requested_usecols)
        if missing_required:
            raise SystemExit(f"--usecols is missing required column(s): {sorted(missing_required)}")

    input_cols = get_input_columns(args.input)
    for required in (args.circuit_col, args.executed_col, args.result_col):
        if required not in input_cols:
            raise SystemExit(f"Missing column in input Parquet: {required}")

    total_input_rows = get_total_rows(args.input)
    processed_input_total = 0
    processed_total = 0
    total_dropped_mismatch = 0

    compression = args.compression
    writer: Optional[SafeParquetChunkWriter] = None

    try:
        print("[INFO] Parse/normalize stage started", flush=True)
        print(f"[INFO] Input file:                {args.input}", flush=True)
        print(f"[INFO] Output file:               {args.output}", flush=True)
        print(f"[INFO] Input rows:                {total_input_rows}", flush=True)
        print(f"[INFO] Chunksize:                 {args.chunksize if args.chunksize > 0 else 'all-at-once'}", flush=True)
        print(f"[INFO] Worker processes (--jobs): {args.jobs}", flush=True)
        print(f"[INFO] Drop mismatch rows:        {args.drop_mismatch_rows}", flush=True)
        print(f"[INFO] Usecols:                   {requested_usecols if requested_usecols is not None else 'all columns'}", flush=True)
        print(f"[INFO] Compression:               {args.compression}", flush=True)

        t0 = time.time()

        if args.chunksize and args.chunksize > 0:
            writer = SafeParquetChunkWriter(args.output, compression=compression)
            n_chunks = (total_input_rows + args.chunksize - 1) // args.chunksize if total_input_rows > 0 else 0

            for chunk_idx, chunk in enumerate(
                iter_parquet_batches(args.input, batch_size=args.chunksize, columns=requested_usecols),
                start=1,
            ):
                out = process_chunk(
                    chunk.copy(),
                    args.circuit_col,
                    args.executed_col,
                    args.result_col,
                    args.trust_level,
                    args.jobs,
                )

                out, dropped = _drop_mismatch_rows_if_requested(
                    out,
                    drop_mismatch_rows=args.drop_mismatch_rows,
                    label=f"chunk {chunk_idx}",
                )
                total_dropped_mismatch += dropped

                writer.write(out)

                processed_input_total += len(chunk)
                processed_total += len(out)
                print(
                    f"[INFO] Chunk {chunk_idx}/{n_chunks} done. "
                    f"Input rows read: {processed_input_total} / {total_input_rows}. "
                    f"Expanded rows written so far: {processed_total}",
                    flush=True,
                )

        else:
            df = next(
                iter_parquet_batches(
                    args.input,
                    batch_size=max(total_input_rows, 1),
                    columns=requested_usecols,
                )
            )

            out = process_chunk(
                df.copy(),
                args.circuit_col,
                args.executed_col,
                args.result_col,
                args.trust_level,
                args.jobs,
            )

            out, dropped = _drop_mismatch_rows_if_requested(
                out,
                drop_mismatch_rows=args.drop_mismatch_rows,
                label="full dataset",
            )
            total_dropped_mismatch += dropped

            out.to_parquet(args.output, index=False, compression=None if compression == "none" else compression)
            processed_total = len(out)

            print(f"Processed expanded rows: {len(out)}", flush=True)
            print(f"{args.circuit_col} parse OK rate: {out[f'{args.circuit_col}_parse_ok'].mean()*100.0:.2f}%", flush=True)
            print(f"{args.executed_col} parse OK rate: {out[f'{args.executed_col}_parse_ok'].mean()*100.0:.2f}%", flush=True)
            if "batch_len_mismatch_circuit_vs_executed_vs_result" in out.columns:
                print(
                    "batch length mismatches retained: "
                    f"{int(out['batch_len_mismatch_circuit_vs_executed_vs_result'].fillna(False).astype(bool).sum())}",
                    flush=True,
                )

        elapsed = time.time() - t0
        out_size = os.path.getsize(args.output) if os.path.exists(args.output) else 0

        if args.drop_mismatch_rows:
            print(f"[INFO] Mismatch rows were removed from the dataset. Total dropped: {total_dropped_mismatch}", flush=True)
        else:
            print("[INFO] Mismatch rows were retained in the dataset.", flush=True)

        print(f"[DONE] Normalized and expanded Parquet written: {args.output}", flush=True)
        print(f"[DONE] Output expanded rows:                    {processed_total}", flush=True)
        print(f"[DONE] Output size bytes:                       {out_size}", flush=True)
        print(f"[DONE] Elapsed sec:                             {elapsed:.2f}", flush=True)

    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()