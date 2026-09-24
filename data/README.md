# Dataset

The dataset is not in this repository. Download `qstd_v1.0.parquet` (53 MB) from
the LRZ FAIR Data Portal (https://rdm.lab.lrz.de/) and put it here. The code
reads it directly — there is no conversion step.

    ls data/qstd_v1.0.parquet

Both the parquet and anything derived from it are gitignored.

## Alignment windows

Table VII compares 2, 5, 10 and 20-minute sensor alignment windows. The published
dataset is the 5-minute build. If the other three are published alongside it, put
them in `data/windows/` as `window_02.parquet`, `window_10.parquet` and
`window_20.parquet`; `scripts/reproduce_all.sh` picks them up automatically and
skips that table when they are absent.
