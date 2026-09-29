# Dataset

The dataset is not in this repository. Download `qstd_v1.0.parquet` (57 MB) from
the LRZ FAIR Data Portal (https://rdm.lab.lrz.de/) and put it here. The code
reads it directly — there is no conversion step.

    ls data/qstd_v1.0.parquet

Both the parquet and anything derived from it are gitignored.

## Alignment windows

Sensor readings are aggregated over a window ending at each job's completion
timestamp, and the published dataset uses **5 minutes**.

The dataset was also built at 2, 10 and 20 minutes to check whether that choice
matters. It does not, materially: the telemetry contribution changes by at most
**+0.0012 R² against a baseline of 0.91** on the superconducting device, and is
not resolvable at all on the trapped-ion one. Those three builds are therefore
**not published** — they are 74-78 MB each, and would multiply the release size
five-fold to no purpose.

`experiments/window_ablation.py` is still included, because it documents how the
check was made. Given only the published dataset it reports the 5-minute column
and says the others are absent; that is expected, not a missing file. If you have
built the other windows yourself from the raw job records, pass them with
`--window 2=<path>` and so on.
