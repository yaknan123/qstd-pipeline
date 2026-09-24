#!/usr/bin/env python3
"""
00_data_preprocessing.py
========================

PURPOSE
-------
Stage-00 preprocessing for the raw dataset.

This version:
1. Reads the input CSV or Parquet
2. Applies basic preprocessing filters
3. Writes the cleaned result as Parquet by default

DEFAULT CLEANING
----------------
By default, this script can:

1. Drop rows where executed_resource is NULL or empty      (if --drop-na)
2. Exclude selected executed_resource values               (operator-supplied codes)

WHY THIS VERSION
----------------
The first stage may start from CSV or Parquet, but later stages should use
Parquet for faster IO, smaller storage, and cleaner downstream processing.

USAGE
-----
1. Basic preprocessing from CSV to Parquet:
python3 00_data_preprocessing.py \
  --input /path/to/raw_input.csv \
  --output /path/to/cleaned_output.parquet \
  --drop-na \
  --exclude-resources <codes> \
  --threads 16

2. Basic preprocessing from Parquet to Parquet:
python3 00_data_preprocessing.py \
  --input /path/to/circuit_raw_data.parquet \
  --output /path/to/stage00_cleaned.parquet \
  --drop-na \
  --exclude-resources <codes> \
  --threads 16

3. Quick sample test:
python3 00_data_preprocessing.py \
  --input /path/to/raw_input.csv \
  --output /path/to/sample_cleaned.parquet \
  --drop-na \
  --exclude-resources <codes> \
  --sample-rows 10000 \
  --threads 8

4. Write CSV instead, if needed:
python3 00_data_preprocessing.py \
  --input /path/to/raw_input.parquet \
  --output /path/to/cleaned_output.csv \
  --drop-na \
  --exclude-resources <codes>
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional

import duckdb


def parse_csv_list_arg(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_escape_string(s: str) -> str:
    return s.replace("'", "''")


def detect_output_format(path: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit.lower()
    low = path.lower()
    if low.endswith(".parquet"):
        return "parquet"
    if low.endswith(".csv"):
        return "csv"
    raise ValueError("Could not infer output format. Use --format parquet or --format csv.")


def detect_input_format(path: str, explicit: Optional[str]) -> str:
    if explicit:
        fmt = explicit.lower()
        if fmt not in {"csv", "parquet"}:
            raise ValueError("--input-format must be csv or parquet")
        return fmt

    low = path.lower()
    if low.endswith(".csv"):
        return "csv"
    if low.endswith(".parquet"):
        return "parquet"

    raise ValueError("Could not infer input format. Use --input-format csv or --input-format parquet.")


def get_input_columns(
    con: duckdb.DuckDBPyConnection,
    input_path: str,
    input_format: str,
    header: bool,
    delim: str,
) -> List[str]:
    escaped = sql_escape_string(input_path)
    delim_esc = sql_escape_string(delim)

    if input_format == "csv":
        sql = f"""
        DESCRIBE
        SELECT *
        FROM read_csv_auto(
            '{escaped}',
            header={str(header).lower()},
            delim='{delim_esc}'
        )
        """
    elif input_format == "parquet":
        sql = f"""
        DESCRIBE
        SELECT *
        FROM read_parquet('{escaped}')
        """
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    rows = con.execute(sql).fetchall()
    return [r[0] for r in rows]


def build_source_sql(
    input_path: str,
    input_format: str,
    header: bool,
    delim: str,
) -> str:
    escaped = sql_escape_string(input_path)

    if input_format == "csv":
        delim_esc = sql_escape_string(delim)
        header_sql = str(header).lower()
        return f"""
            read_csv_auto(
                '{escaped}',
                header={header_sql},
                delim='{delim_esc}'
            )
        """

    if input_format == "parquet":
        return f"read_parquet('{escaped}')"

    raise ValueError(f"Unsupported input format: {input_format}")


def build_where_sql(
    resource_col: str,
    drop_na: bool,
    exclude_resources: List[str],
    extra_where: Optional[str],
) -> str:
    clauses: List[str] = []

    if extra_where and extra_where.strip():
        clauses.append(f"({extra_where.strip()})")

    if drop_na:
        clauses.append(
            f"LENGTH(TRIM(COALESCE(CAST({quote_ident(resource_col)} AS VARCHAR), ''))) > 0"
        )

    if exclude_resources:
        excluded = ", ".join(f"'{sql_escape_string(x)}'" for x in exclude_resources)
        clauses.append(f"{quote_ident(resource_col)} NOT IN ({excluded})")

    if not clauses:
        return ""

    return "WHERE " + " AND ".join(clauses)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage-00 preprocessing: read CSV or Parquet, clean rows, and write Parquet or CSV."
    )
    ap.add_argument("--input", required=True, help="Input CSV or Parquet file")
    ap.add_argument("--output", required=True, help="Output file (.parquet or .csv)")
    ap.add_argument("--input-format", choices=["csv", "parquet"], default=None, help="Optional explicit input format")
    ap.add_argument("--format", choices=["parquet", "csv"], default=None, help="Output format")
    ap.add_argument(
        "--drop-na",
        action="store_true",
        help="Drop rows where executed_resource is NULL or empty"
    )
    ap.add_argument(
        "--exclude-resources",
        default="",
        help="Comma-separated executed_resource values to exclude"
    )
    ap.add_argument(
        "--resource-col",
        default="executed_resource",
        help="Resource column name"
    )
    ap.add_argument(
        "--where",
        default=None,
        help="Optional extra SQL WHERE clause"
    )
    ap.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="Optional LIMIT for quick testing"
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="DuckDB worker threads. 0 means default"
    )
    ap.add_argument(
        "--memory-limit",
        default=None,
        help="Optional DuckDB memory limit, e.g. 64GB"
    )
    ap.add_argument(
        "--csv-header",
        action="store_true",
        default=True,
        help="Input CSV has a header row (default: true)"
    )
    ap.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter (default: ,)"
    )

    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    input_format = detect_input_format(args.input, args.input_format)
    out_format = detect_output_format(args.output, args.format)
    exclude_resources = parse_csv_list_arg(args.exclude_resources)

    con = duckdb.connect()
    try:
        if args.threads and args.threads > 0:
            con.execute(f"PRAGMA threads={int(args.threads)}")
        if args.memory_limit:
            con.execute(f"PRAGMA memory_limit='{sql_escape_string(args.memory_limit)}'")

        input_cols = get_input_columns(
            con=con,
            input_path=args.input,
            input_format=input_format,
            header=args.csv_header,
            delim=args.delimiter,
        )

        if args.resource_col not in input_cols:
            raise ValueError(
                f"Resource column '{args.resource_col}' not found in input {input_format.upper()}."
            )

        source_sql = build_source_sql(
            input_path=args.input,
            input_format=input_format,
            header=args.csv_header,
            delim=args.delimiter,
        )

        total_rows_sql = f"SELECT COUNT(*) FROM {source_sql}"
        total_rows = con.execute(total_rows_sql).fetchone()[0]

        where_sql = build_where_sql(
            resource_col=args.resource_col,
            drop_na=args.drop_na,
            exclude_resources=exclude_resources,
            extra_where=args.where,
        )

        filtered_count_sql = f"""
            SELECT COUNT(*)
            FROM {source_sql}
            {where_sql}
        """
        kept_rows = con.execute(filtered_count_sql).fetchone()[0]
        removed_rows = total_rows - kept_rows

        limit_sql = f"LIMIT {int(args.sample_rows)}" if args.sample_rows and args.sample_rows > 0 else ""

        select_sql = f"""
            SELECT *
            FROM {source_sql}
            {where_sql}
            {limit_sql}
        """

        print("[INFO] Stage-00 preprocessing started", flush=True)
        print(f"[INFO] Input file          : {args.input}", flush=True)
        print(f"[INFO] Input format        : {input_format}", flush=True)
        print(f"[INFO] Output file         : {args.output}", flush=True)
        print(f"[INFO] Output format       : {out_format}", flush=True)
        print(f"[INFO] Resource column     : {args.resource_col}", flush=True)
        print(f"[INFO] Drop NA             : {args.drop_na}", flush=True)
        print(f"[INFO] Excluded resources  : {exclude_resources if exclude_resources else '<none>'}", flush=True)
        print(f"[INFO] Extra WHERE         : {args.where if args.where else '<none>'}", flush=True)
        if input_format == "csv":
            print(f"[INFO] Delimiter           : {args.delimiter}", flush=True)
        if args.sample_rows:
            print(f"[INFO] Sample rows         : {args.sample_rows}", flush=True)
        if args.threads:
            print(f"[INFO] Threads             : {args.threads}", flush=True)
        if args.memory_limit:
            print(f"[INFO] Memory limit        : {args.memory_limit}", flush=True)

        print(f"[INFO] Input rows          : {total_rows}", flush=True)
        print(f"[INFO] Removed rows        : {removed_rows}", flush=True)
        print(f"[INFO] Output rows         : {min(kept_rows, args.sample_rows) if args.sample_rows else kept_rows}", flush=True)

        t0 = time.time()
        escaped_output = sql_escape_string(args.output)

        if out_format == "parquet":
            copy_sql = f"""
                COPY (
                    {select_sql}
                )
                TO '{escaped_output}'
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        else:
            copy_sql = f"""
                COPY (
                    {select_sql}
                )
                TO '{escaped_output}'
                (FORMAT CSV, HEADER true)
            """

        con.execute(copy_sql)

        elapsed = time.time() - t0
        out_size = os.path.getsize(args.output) if os.path.exists(args.output) else 0

        print("[DONE] Preprocessing finished", flush=True)
        print(f"[DONE] Output file         : {args.output}", flush=True)
        print(f"[DONE] Output size bytes   : {out_size}", flush=True)
        print(f"[DONE] Elapsed sec         : {elapsed:.2f}", flush=True)

    finally:
        con.close()


if __name__ == "__main__":
    main()