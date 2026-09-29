# QSTD — pipeline and analysis code

Code for the **Quantum Sensor-aligned Telemetry Dataset (QSTD)**: how the dataset
was built, and how to reproduce every number in the paper from it.

The dataset is published separately on the
[LRZ FAIR Data Portal](https://rdm.lab.lrz.de/) under CC-BY-4.0. This code is
Apache-2.0.

---

## The dataset

One record is one quantum circuit execution: what was run, how well it ran, and
what the facility's sensors and the device's last calibration read at the moment
it finished.

| | |
|---|---|
| Records | 1,555,121 — one per executed circuit |
| Jobs | 214,517 — a job is one submission of one or more circuits |
| Period | 2024-05-24 to 2026-05-10 (UTC) |
| Columns | 313: 182 sensor, 100 calibration, 20 circuit, 3 job, 3 time, 3 identifier, 2 target |
| Devices | `superconducting_20q` (1,372,537 records), `trapped_ion_20q` (176,697), three other backends (5,887) |
| File | `qstd_v1.0.parquet`, 57 MB on disk, about 4 GB as a pandas frame |

**The target is `hellinger_native`**: the Hellinger distance between the measured
outcome distribution and a noise-free simulation of the same circuit at the same
shot count. 0 means identical, 1 means no overlap. `hellinger_logical` is the
same distance against the circuit as submitted rather than as transpiled; the two
correlate at r = 0.999, so use one or the other, never one to predict the other.

Sensor coverage depends on the month and calibration coverage on the device — the
dataset's own `README.md` carries the month-by-month table, and
`analysis/data_availability_report.py` regenerates it. That unevenness is why
there are three analysis sets rather than one; see below.

---

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/). It creates the
virtual environment, installs the pinned versions from `uv.lock`, and fetches a
suitable Python if the system one is too old — there is no `pip` step and nothing
else to install.

**1. Install uv** (once, if you do not have it):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS / Linux
# Windows PowerShell:
#   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Create the environment**, from the project root:

```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock` and builds `.venv/` with exactly the
versions the published results were produced with. Python 3.10 or newer is
required; uv installs one if needed. The default set is enough to reproduce every
table.

Optional extras:

```bash
uv sync --extra notebook    # Jupyter, for analysis/qstd_overview.ipynb
uv sync --extra build       # Qiskit, only needed to rebuild the dataset itself
```

**3. Get the dataset.** Download `qstd_v1.0.parquet` (57 MB) from the portal into
`data/`. The code reads it directly; there is no conversion step.

**4. Check it works:**

```bash
uv run python -c "import lightgbm, pandas, pyarrow; print('ready')"
ls data/qstd_v1.0.parquet
```

Every command below goes through `uv run`, which uses the locked environment
without you activating anything.

---

## Reproduce the paper's results

```bash
bash scripts/reproduce_all.sh
```

About two hours on 8 cores, CPU only — no GPU, no database, no cluster access.

| Paper | Script | Output |
|---|---|---|
| Table III — tiers per device | `experiments/train_tiers.py` | `task3_per_device.csv` |
| Table IV — shot distribution | `experiments/shot_histogram.py` | `task4_shot_distribution.csv` |
| Table V — feature categories | `experiments/tier2a_sensor_only.py` | `task5_decomposition.csv` |
| Table VI — leave-one-device-out | `experiments/lodo.py` | `task6_lodo.csv` |
| Table VIII — class shift | `experiments/class_shift.py` | `task8_class_shift.json` |
| Table IX — ΔR² over five splits | `experiments/split_repeats.py` | `task9_split_summary.csv` |
| Table X — forward-in-time split | `experiments/forward_split.py` | `task10_forward_split.csv` |
| Analysis sets side by side | `experiments/compare_pools.py` | `pool_comparison.csv` |

Three scripts are not in the driver, because they answer a question rather than
fill a table: `experiments/nn_devemb.py` is the Tier 1 neural baseline and prints
its scores, `analysis/data_availability_report.py` regenerates the sensor and
calibration coverage tables, and `release/validate_release.py` checks a built
release against its manifest.

Results land in `experiments/results/`, and non-default analysis sets in
`experiments/results/pool_<name>/`. To run one on its own:

```bash
uv run python experiments/train_tiers.py
uv run python experiments/split_repeats.py --pools A B AVAIL
uv run jupyter lab analysis/qstd_overview.ipynb     # needs --extra notebook
```

**On the alignment window.** `experiments/window_ablation.py` is included but is
not one of the paper's tables. It compares 2, 5, 10 and 20-minute sensor
alignment windows, and each window is a separate build of the same records. Only
the 5-minute build is published — widening the window changes the telemetry
contribution by at most +0.0012 R² against a baseline of 0.91, so the other three
would add 229 MB to the release for no difference in what the data supports. Run
against the published dataset the script reports the 5-minute column alone. See
`data/README.md`.

---

## What is in here

```
experiments/   the models and evaluations behind every table
pipeline/      stages 00-10: raw job records -> the training dataset
release/       the validator that checks a built release
analysis/      qstd_overview.ipynb, a short tour of the data; coverage figures
scripts/       reproduce_all.sh and check_results.py
data/          where the dataset goes (not committed)
```

**Reproducible from the published dataset:** everything in `experiments/`,
`analysis/qstd_overview.ipynb` and `analysis/data_availability_report.py`.

**Included to be read, not run:** `pipeline/` documents how the dataset was
produced from raw job records. It needs the job database, which is not public. It
is here so that the alignment rule, the calibration recency bound and the
cleaning decisions can be checked against what the paper claims.

**Deliberately not included:** the sensor and calibration extraction, its
configuration, and the release builder. Those are written around a production
facility's internal monitoring paths, room names and system names. What they do
is described below and in the paper; the dataset they produced is what is being
published.

---

## How the dataset was built

Raw job records in, training dataset out. Each stage reads the previous stage's
Parquet file and writes its own, so any intermediate step can be inspected.

| Stage | Script | What it does |
|---|---|---|
| 00 | `00_preprocess.py` | Reads the raw job export, drops records with no result, puts every timestamp in UTC. |
| 01 | `01_normalize_circuits.py` | Parses each circuit (OpenQASM 2 or 3) and expands a batch job into one row per circuit, keeping the job id so the batch stays traceable. |
| 02 | `02_simulate_ideal.py` | Simulates every circuit noise-free at the shot count the device used. Sampling is seeded per circuit, so the reference distribution is reproducible rather than a fresh draw. This is the stage that takes hours. |
| 03 | `03_verify_endianness.py` | Aligns the measured bitstrings to the simulator's convention. |
| 04 | `04_compute_hellinger.py` | The target: `hellinger_native` against the transpiled circuit, `hellinger_logical` against the circuit as submitted. |
| 06 | `06_extract_features.py` | 21 structural features per circuit, computed twice — as submitted (`logical_*`) and as transpiled onto the device (`native_*`). |
| 07 | `07_build_dags.py` | Builds each circuit's DAG into a side database, then drops the circuit and result text from the row. |
| 08 | `08_merge_sensor_data.py` | Joins the aligned sensor and calibration table onto the job id. |
| 09 | `09_filter_rows.py` | Drops the per-stage diagnostic columns and any row missing both targets. |
| 10 | `10_clean_dataset.py` | Drops cancelled jobs, classifies every remaining column by an explicit rule, and removes any feature that is empty, constant, or present on under 1% of rows. |

There is no stage 05 — the number belongs to a single-job inspection tool, not to
the chain. The release step that follows stage 10 gives the columns their public
names, truncates timestamps to the hour and generates the identifiers; it is not
included here, for the reason given above.

### The three rules that matter

Everything else is bookkeeping. These three are the substantive choices, and each
is the one a reader should check against the code.

**Bit ordering.** Every backend on the platform was confirmed on hardware to
return Qiskit's little-endian convention — bit n-1 leftmost, bit 0 rightmost,
the same convention the simulator uses. Stage 03 therefore
runs with `--assume-mode normal` and applies that ordering to every record. It
does not infer the ordering per record: on a circuit that ran badly the measured
distribution is close to noise, the two orderings score within a hair of each
other, and a per-record detector picks the wrong one often enough to matter. The
detector still runs and still records what it would have chosen, in
`*_best_endianness_raw`, so the exposure stays measurable.

**Sensor alignment.** Readings come from the window **ending at the completion
timestamp**, aggregated as mean, min, max and standard deviation. Never the
submission timestamp: a job can sit in a queue for hours, and readings taken then
describe an unrelated moment. The published dataset uses a 5-minute window.

**Calibration recency.** The most recent calibration published within **48 hours**
before the job finished. Anything older is left empty, never carried forward, so
a stale reading can never pass as a fresh one.

### Anonymisation

`job_id` and `circuit_id` are numbers generated for this dataset: a fixed-seed
permutation, not a hash of the internal job numbers. There is no salt and no
secret to keep, and the ids cannot be inverted because they are not derived from
anything. Rebuilding from the same input reproduces them exactly. Timestamps are
published at hour resolution. Devices are identified by technology and size
(`superconducting_20q`, `trapped_ion_20q`), never by an internal name. No user,
project or account information and no circuit source code is published.

---

## Analysis sets

Sensor coverage depends on the month, so the sensor tiers are not trained on
every record: training one on months when the sensors were not logging would let
missingness stand in for time. Three sets are reported.

| Set | superconducting_20q | trapped_ion_20q |
|---|---|---|
| **A** (default) | 2025-04 onward, excluding 2025-09 | 2025-03 onward |
| **B** | 2025-06 onward, excluding 2025-09 | 2025-03 onward |
| **AVAIL** | rows carrying sensor data and calibration at most 48 h old | rows carrying sensor data |

Circuit-only models (Tier 1, leave-one-device-out, the shot histogram) use every
record.

```bash
uv run python experiments/train_tiers.py --pool B --tier 2
uv run python experiments/compare_pools.py --pools A B AVAIL
```

---

## Reproducibility

Seed 42 throughout (`experiments/common.py: set_all_seeds`), covering Python,
NumPy, PyTorch and the model parameters. Splits are grouped by job, so circuits
from one submission never straddle train and test. `uv.lock` pins every package
version.

**Why ΔR² is reported as mean ± sd.** The effects measured here are a few
thousandths of R², and a single 80/20 split does not resolve a difference that
small: which jobs land in the holdout moves a ΔR² by about as much as the ΔR² is.
`experiments/split_repeats.py` repeats each comparison over five different group
splits and reports the mean, the standard deviation and the range, with a
`resolved` column stating whether mean ± 1 sd excludes zero.

**Why there is a forward-in-time split.** A random split lets a model recognise
the period a holdout circuit belongs to and predict the fidelity typical of that
period. `experiments/forward_split.py` trains on the earliest 80% of the timeline
and tests on the latest 20%, which removes that shortcut.

The neural baseline runs on CPU; on a GPU it may differ in the last decimal.

---

## Citation

Please cite the accompanying paper. Contact: yaknan.gambo@lrz.de
