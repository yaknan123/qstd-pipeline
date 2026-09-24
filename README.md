# QSTD — pipeline and analysis code

Code for the **Quantum Sensor-aligned Telemetry Dataset (QSTD)**: how the dataset
was built, and how to reproduce every number in the paper from it.

The dataset is published separately on the
[LRZ FAIR Data Portal](https://rdm.lab.lrz.de/) under CC-BY-4.0. This code is
Apache-2.0.

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

**3. Get the dataset.** Download `qstd_v1.0.parquet` (53 MB) from the portal into
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
| Table VII — alignment window | `experiments/window_ablation.py` | `task7_window_ablation.csv` |
| Table VIII — class shift | `experiments/class_shift.py` | `task8_class_shift.json` |
| Table IX — ΔR² over five splits | `experiments/split_repeats.py` | `task9_split_summary.csv` |
| Table X — forward-in-time split | `experiments/forward_split.py` | `task10_forward_split.csv` |
| Analysis sets side by side | `experiments/compare_pools.py` | `pool_comparison.csv` |

Results land in `experiments/results/`, and non-default analysis sets in
`experiments/results/pool_<name>/`. To run one on its own:

```bash
uv run python experiments/train_tiers.py
uv run python experiments/split_repeats.py --pools A B AVAIL
uv run jupyter lab analysis/qstd_overview.ipynb     # needs --extra notebook
```

**One exception.** Table VII compares 2, 5, 10 and 20-minute sensor alignment
windows. Each is a separate build of the same records, and the published dataset
is the 5-minute one. Without the other three the ablation reports the 5-minute
column only. See `data/README.md`.

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

1. **Job records** — every circuit execution on the platform, with the submitted
   circuit, the transpiled circuit and the measured outcome counts.
2. **Ideal simulation** (`pipeline/02_simulate_ideal.py`) — each circuit is
   simulated noise-free, and the Hellinger distance between the measured and the
   ideal distribution is the target. Sampling is seeded per circuit, so the
   target is reproducible to 1e-16.
3. **Circuit features** (`06`, `07`) — 21 structural features, for the circuit as
   submitted (`logical_*`) and after transpilation onto the device (`native_*`).
4. **Result alignment** (`03`) — the measured bitstrings are aligned to the
   simulator's convention. The devices were confirmed on hardware to use Qiskit's
   little-endian ordering, bit n-1 leftmost and bit 0 rightmost, so the stage runs
   with that convention fixed rather than inferring it per record.
5. **Sensor alignment** — for each execution, facility sensor readings from the
   window **ending at the completion timestamp**, aggregated as mean, min, max and
   standard deviation. Never the submission timestamp: a job can wait in a queue
   for hours, and readings taken then describe an unrelated moment.
6. **Calibration** — the most recent calibration published within **48 hours**
   before the job finished. Anything older is left empty, never carried forward.
7. **Cleaning** (`pipeline/10_clean_dataset.py`) — cancelled jobs dropped, every
   column classified by an explicit rule, and any feature that is empty, constant,
   or present on under 1% of rows removed.
8. **Release** — public column names, hour-resolution timestamps, generated
   identifiers.

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
