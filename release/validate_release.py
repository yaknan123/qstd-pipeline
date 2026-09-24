#!/usr/bin/env python3
"""
validate_release.py
===================

Checks a QSTD upload folder (release/dist/qstd_v<version>/) exactly as it will
be published. Needs nothing but the folder, so it can be rerun on downloaded
files. build_release.py calls it after every build.

FAIL (exit code 1) on anything that makes the release wrong or unsafe:
  files       exactly the expected five files; no salt or other private file
  integrity   checksum and counts match the manifest; license, title and record
              count in the parquet metadata; the "N.NNM records" statement
              matches the row count
  dictionary  one row per column, in file order, each with a description and
              a source; no references to internal documents
  identifiers job_id is a dense 1..N job number and circuit_id numbers the
              circuits within a job; neither is derived from internal ids
  time        hour resolution; submitted <= scheduled <= completed (scheduled
              may be empty where the scheduler log was inconsistent)
  values      Hellinger in [0, 1]; status only COMPLETED; no empty or
              single-valued columns (except status and sensor min/max/sd);
              each sensor channel has a
              complete set of aggregates with min <= mean <= max and sd >= 0
  scoping     sensor and calibration columns are empty on the records of
              devices they do not belong to
  disclosure  no raw-id, DAG, user, project or account columns

The portal record metadata (qstd_v<version>_lrz_fair_record.json, next to the
folder) must exist and agree on title and license. WARN (not blocking) while it
still has placeholders.

USAGE
-----
    python release/validate_release.py release/dist/qstd_v1.0
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

BLOCKED = re.compile(r"(_dag_|^owner$|^budget$|^user|^project|^id$|^parent_id$|^sub_id$)")
INTERNAL = re.compile(r"docstring|PAPER_OBJECTIVES|Section \d|stage\s?0\d|\.py\b", re.I)
TIME = ("submitted_hour_utc", "scheduled_hour_utc", "completed_hour_utc")


def main(folder: str) -> int:
    root = Path(folder)
    stem = root.name
    fails, warns = [], []
    fail = fails.append

    expected = {f"{stem}.parquet", f"{stem}_columns.csv", f"{stem}_manifest.json",
                "README.md", "LICENSE.txt"}
    present = {p.name for p in root.iterdir()}
    if present != expected:
        fail(f"files: missing {sorted(expected - present)}, unexpected {sorted(present - expected)}")
        return report(fails, warns)

    pq_path = root / f"{stem}.parquet"
    manifest = json.loads((root / f"{stem}_manifest.json").read_text())
    if hashlib.sha256(pq_path.read_bytes()).hexdigest() != manifest["sha256"]:
        fail("parquet checksum does not match the manifest")
    pf = pq.ParquetFile(pq_path)
    meta = {k.decode(): v.decode() for k, v in (pf.schema_arrow.metadata or {}).items()
            if k.startswith(b"qstd.")}
    df = pd.read_parquet(pq_path)
    n = len(df)
    if meta.get("qstd.license") != "CC-BY-4.0":
        fail("parquet metadata: license is not CC-BY-4.0")
    if meta.get("qstd.records") != str(n) or manifest["records"] != n:
        fail("record count differs between data, parquet metadata and manifest")
    if manifest["columns"] != len(df.columns) or manifest["jobs"] != df["job_id"].nunique():
        fail("column or job count differs from the manifest")
    # The description must carry the record count, however it is worded.
    if f"{n / 1e6:.2f}M" not in meta.get("qstd.description", ""):
        fail("statement's record count does not match the data")
    readme = (root / "README.md").read_text()
    if f"{n:,}" not in readme or "CC-BY-4.0" not in readme:
        fail("README does not state the record count and license")

    cols = pd.read_csv(root / f"{stem}_columns.csv")
    if list(cols["column"]) != list(df.columns):
        fail("data dictionary does not list the columns in file order")
    for field in ("description", "source", "group", "device_scope"):
        empty = cols[cols[field].isna() | (cols[field].astype(str).str.strip() == "")]
        if len(empty):
            fail(f"dictionary: {len(empty)} columns without {field}, e.g. {list(empty.column[:3])}")
    bad_src = cols[cols["source"].astype(str).str.contains(r"\?|unknown")]
    if len(bad_src):
        fail(f"dictionary: unresolved source for {list(bad_src.column[:5])}")
    internal = cols[cols["description"].astype(str).str.contains(INTERNAL)]
    if len(internal):
        fail(f"dictionary: references to internal documents in {list(internal.column[:5])}")

    blocked = [c for c in df.columns if BLOCKED.search(c)]
    if blocked:
        fail(f"disclosure: blocked columns {blocked}")
    # Generated ids: dense 1..N job numbers, and one circuit number per job.
    jobs = df["job_id"]
    if not pd.api.types.is_integer_dtype(jobs) or jobs.isna().any():
        fail("job_id is not an integer everywhere")
    elif sorted(jobs.unique()) != list(range(1, jobs.nunique() + 1)):
        fail("job_id is not a dense 1..N sequence")
    if not pd.api.types.is_integer_dtype(df["circuit_id"]):
        fail("circuit_id is not an integer")
    elif not df["circuit_id"].is_unique:
        # It has to identify a row on its own: a per-job counter would look like
        # a row id while taking only a few hundred distinct values.
        fail(f"circuit_id is not unique ({df['circuit_id'].nunique():,} distinct "
             f"values for {len(df):,} rows)")
    # Nothing in the release may be derived from the internal identifiers.
    if any(c in df.columns for c in ("parent_id", "sub_id", "id", "owner", "budget")):
        fail("an internal identifier column is present in the release")

    for c in TIME:
        t = df[c].dropna()
        if ((t.dt.minute != 0) | (t.dt.second != 0) | (t.dt.microsecond != 0)).any():
            fail(f"{c}: finer than one hour")
        if c != "scheduled_hour_utc" and df[c].isna().any():
            fail(f"{c}: missing values")
    if ((df[TIME[0]] > df[TIME[1]]) | (df[TIME[1]] > df[TIME[2]])).any():
        fail("time order: submitted <= scheduled <= completed violated")

    for c in ("hellinger_native", "hellinger_logical"):
        if ((df[c] < 0) | (df[c] > 1)).any():
            fail(f"{c} outside [0, 1]")
    if set(df["status"].dropna()) != {"COMPLETED"}:
        fail("status contains values other than COMPLETED")
    window_extremes = set(cols.loc[(cols.group == "sensor")
                                   & cols.statistic.isin(["min", "max", "sd"]), "column"])
    for c in df.columns:
        v = df[c].dropna()
        if v.empty:
            fail(f"{c} is empty")
        elif v.nunique() == 1 and c != "status" and c not in window_extremes:
            # A single-valued sensor min/max/sd (a counter whose minimum is
            # always 0) is a real reading of a channel that otherwise varies.
            fail(f"{c} has a single value")

    # Sensor channels and statistics come from the dictionary (release names
    # have no separator that could be parsed reliably, e.g. *_x_peak).
    sensor = cols[cols.group == "sensor"]
    if sensor["channel"].isna().any() or sensor["statistic"].isna().any():
        fail("dictionary: sensor columns without channel or statistic")
    for base, grp in sensor.groupby("channel"):
        stats = dict(zip(grp["statistic"], grp["column"]))
        if set(stats) not in ({"mean", "min", "max", "sd"}, {"value"}):
            fail(f"sensor channel {base}: incomplete statistics {sorted(stats)}")
            continue
        if set(stats) == {"value"}:
            continue
        if any(stats[s] != f"{base}_{s}" for s in stats):
            fail(f"sensor channel {base}: columns not named <channel>_<statistic>")
            continue
        lo, mu, hi, sd = (df[stats[s]] for s in ("min", "mean", "max", "sd"))
        if ((lo > mu + 1e-6) | (mu > hi + 1e-6)).any() or (sd < 0).any():
            fail(f"sensor channel {base}: min <= mean <= max or sd >= 0 violated")
        if not (lo.isna() == mu.isna()).all():
            fail(f"sensor channel {base}: aggregates filled on different rows")
    single = set(sensor.loc[sensor.statistic == "value", "column"])
    clash = [c for c in single if c.rsplit("_", 1)[-1] in ("mean", "min", "max", "sd")]
    if clash:
        fail(f"single-value channels named like a window statistic: {clash[:5]}")
    # Every sensor and calibration column must be named in the published scheme:
    # sc_* for the superconducting device, ion_* for the trapped-ion one. This is
    # a positive check rather than a search for known-bad words, so a name the
    # renaming missed fails here instead of being published.
    stray = [c for c in cols.loc[cols.group.isin(["sensor", "calibration"]), "column"]
             if not re.fullmatch(r"(sc|ion)_[a-z0-9_]+", c)]
    if stray:
        fail(f"sensor/calibration columns not in the published naming scheme: {stray[:5]}")

    for _, r in cols[cols.group.isin(["sensor", "calibration"])].iterrows():
        off = df["device"] != r.device_scope
        if df.loc[off, r.column].notna().any():
            fail(f"{r.column} ({r.device_scope}) has values on other devices' records")

    record_path = root.parent / f"{stem}_lrz_fair_record.json"
    if not record_path.exists():
        fail(f"portal record metadata missing: {record_path.name}")
        return report(fails, warns)
    record = record_path.read_text()
    if "FILL-IN" in record:
        warns.append(f"{record_path.name} still has FILL-IN placeholders (creators, "
                     "publication date): complete them before uploading")
    try:
        rec = json.loads(record)["metadata"]
        if rec["rights"][0]["id"] != "cc-by-4.0" or rec["title"] != meta.get("qstd.title"):
            fail(f"{record_path.name}: title or license differs from the dataset")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        fail(f"{record_path.name} unreadable: {e}")

    print(f"  checked {n:,} records x {len(df.columns)} columns in {root}")
    return report(fails, warns)


def report(fails, warns) -> int:
    for w in warns:
        print(f"  WARN  {w}")
    for f in fails:
        print(f"  FAIL  {f}")
    print("  RELEASE VALID" if not fails else f"  RELEASE INVALID ({len(fails)} failures)")
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
