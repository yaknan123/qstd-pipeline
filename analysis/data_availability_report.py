#!/usr/bin/env python3
"""Where sensor and calibration data exist, and how irregular the cells are.

Reads the full dataset and draws where sensor and calibration data exist, with
every channel labelled by its QSTD release name (release/column_names.py). The
older plot_sensor_onset.py / plot_sensor_timeline.py drew the sensor side from a
probe sample; their figures are kept in figures/probe_sample/.

OUTPUTS (analysis/figures/ unless --outdir)
------------------------------------------
  data_availability.png/.pdf      per month, the share of each device's jobs that
                                  carry sensors and fresh calibration, sensors
                                  only, calibration only, or neither; and the
                                  share of Q-Exa calibration that is stale
  calibration_onset.png/.pdf      circuit jobs per month vs calibration channels
                                  returning data (the calibration twin of
                                  sensor_onset)
  sensor_onset.png/.pdf           circuit jobs per month vs sensor channels
                                  returning data (full data; supersedes the
                                  probe-sample version in figures/probe_sample/)
  sensor_timeline.png/.pdf        per-channel sensor coverage, release names;
                                  channels dropped before training in grey
  analysis_sets.md/.csv           what each --pool keeps, with row counts
  avail_row_selection.csv         step-by-step why pool AVAIL keeps its rows,
                                  for both devices (also in analysis_sets.md)
  calibration_timeline.png/.pdf   per-qubit / per-pair coverage of every
                                  calibration metric, one panel per metric (the
                                  twin of sensor_timeline)
  data_availability.csv/.md       the report table: per device and month, jobs,
                                  row and cell coverage, calibration age, and
                                  the rows in pool A and in pool AVAIL
  sensor_channel_coverage.csv     per channel and month, share of jobs with a reading
  calibration_channel_coverage.csv  per metric, qubit/pair and month, share with a
                                  value and share with a value other than 0 or 1

SOURCES
-------
Row states and the table use the stage 10 training data, so they match what the
models see and what experiments/common.py pool AVAIL keeps. Channel coverage uses
stage 09, which still holds the channels stage 10 drops as constant, dead or
always zero -- dropping them here would hide exactly the irregularity these
figures exist to show.

"Fresh" calibration is no older than CALIB_MAX_AGE_H (48 h, the extractor's
lookback). The extractor does not enforce that lookback per row, so older
values do occur; see experiments/common.py.

    python analysis/data_availability_report.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
# Appended, not prepended: release/ has its own common.py, which must not
# shadow experiments/common.py.
sys.path.append(str(REPO / "release"))
import column_names as N  # noqa: E402  (release names label every channel)
import common as C  # noqa: E402

# Same reference palette as the sensor figures. Device identity keeps slots 1/2
# (Q-Exa blue, Marmot orange) in the onset figure; the row-state figure encodes
# state, not device, with slots 1-3 in fixed order plus a neutral for "neither".
SERIES = {C.QEXA: "#2a78d6", C.MARMOT: "#eb6834"}
LABEL = {C.QEXA: "Q-Exa", C.MARMOT: "Marmot"}
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8983"
SURFACE, GRID = "#fcfcfb", "#e5e4df"
STATES = [  # (key, legend label, colour)
    ("both", "sensors + fresh calibration", "#2a78d6"),
    ("sensor_only", "sensors only", "#eb6834"),
    ("calib_only", "fresh calibration only", "#1baf7a"),
    ("neither", "neither", "#d6d5d0"),
]
RAMP_USABLE = ["#fcfcfb", "#9ec5f4", "#2a78d6", "#104281"]
RAMP_UNUSABLE = ["#fcfcfb", "#c9c8c2", "#8a8983", "#52514e"]
LOW_N = 500          # months with fewer jobs are marked: their shares are noisy

CALIB_RX = re.compile(r"^(QB\d+|TC_\d+_\d+)_(.+)$")
SENSOR_AGG_RX = re.compile(r"^(.+)__(mean|value)$")


def months_between(lo: str, hi: str) -> list[str]:
    y, m, out = int(lo[:4]), int(lo[5:7]), []
    while f"{y:04d}-{m:02d}" <= hi:
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def month_of(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True, errors="coerce").dt.strftime("%Y-%m")


def present(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce").notna()


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=7.5)


def month_ticks(ax, months, every=("-01", "-04", "-07", "-10")):
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([m if m.endswith(every) else "" for m in months], fontsize=7.5)


def save(fig, outdir: Path, name: str):
    for ext in ("pdf", "png"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(p, dpi=200, facecolor=SURFACE)
        print(f"wrote {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def row_states(stage10: Path) -> pd.DataFrame:
    """One row per job: device, month, and which of its own data it carries."""
    cols = C.dataset_columns(stage10)
    sensors, calib = C.resolve_device_columns(cols)
    wanted = sorted({C.DEVICE_COL, C.TIME_COL, C.ROW_ID_COL, C.TARGET_COL}
                    | {c for d in C.PAPER_DEVICES for c in sensors.get(d, []) + calib.get(d, [])})
    df = pq.read_table(str(stage10), columns=wanted).to_pandas()
    df = df[df[C.DEVICE_COL].isin(C.PAPER_DEVICES)].reset_index(drop=True)

    out = pd.DataFrame({"device": df[C.DEVICE_COL], "month": month_of(df[C.TIME_COL]),
                        "has_target": df[C.TARGET_COL].notna().to_numpy()})
    out["has_sensor"] = False
    out["sensor_cells"] = 0.0
    out["has_calib"] = False
    out["calib_cells"] = np.nan
    for d in C.PAPER_DEVICES:
        m = (df[C.DEVICE_COL] == d).to_numpy()
        if sensors.get(d):
            p = present(df.loc[m], sensors[d])
            out.loc[m, "has_sensor"] = p.any(axis=1).to_numpy()
            out.loc[m, "sensor_cells"] = p.mean(axis=1).to_numpy()
        if calib.get(d):
            p = present(df.loc[m], calib[d])
            out.loc[m, "has_calib"] = p.any(axis=1).to_numpy()
            out.loc[m, "calib_cells"] = p.mean(axis=1).to_numpy()

    out["calib_age_h"] = C.calibration_age_hours(df[C.ROW_ID_COL]).to_numpy()
    out.loc[~out.has_calib, "calib_age_h"] = np.nan
    out["stale"] = out.has_calib & (out.calib_age_h > C.CALIB_MAX_AGE_H)
    fresh = out.has_calib & ~out.stale
    out["state"] = np.select(
        [out.has_sensor & fresh, out.has_sensor, fresh],
        ["both", "sensor_only", "calib_only"], "neither")
    # Pool AVAIL: Marmot has no calibration, so sensors alone qualify it.
    no_calib_dev = out.device.map(lambda d: not calib.get(d))
    out["in_avail"] = (out.state == "both") | (no_calib_dev & out.has_sensor)
    for pool in C.MONTH_POOLS:
        col = f"in_pool_{pool}"
        out[col] = False
        for d in C.PAPER_DEVICES:
            m = (out.device == d).to_numpy()
            out.loc[m, col] = C.MONTH_POOLS[pool][d](out.loc[m, "month"]).to_numpy()
    return out


def report_table(rows: pd.DataFrame) -> pd.DataFrame:
    g = rows.groupby(["device", "month"])
    t = pd.DataFrame({
        "jobs": g.size(),
        "sensor_rows_pct": g.has_sensor.mean() * 100,
        "sensor_cells_pct": g.sensor_cells.mean() * 100,
        "calib_rows_pct": g.has_calib.mean() * 100,
        "calib_cells_pct": g.calib_cells.mean() * 100,
        "calib_stale_pct": 100 * g.stale.sum() / g.has_calib.sum().replace(0, np.nan),
        "calib_age_median_h": g.calib_age_h.median(),
        "both_fresh_pct": 100 * (rows.state == "both").groupby([rows.device, rows.month]).mean(),
        "pool_A_rows": g.in_pool_A.sum(),
        "pool_AVAIL_rows": g.in_avail.sum(),
    }).reset_index()
    t["device"] = t.device.map(LABEL)
    tot = rows.groupby("device").agg(
        jobs=("month", "size"), sensor_rows_pct=("has_sensor", "mean"),
        sensor_cells_pct=("sensor_cells", "mean"), calib_rows_pct=("has_calib", "mean"),
        calib_cells_pct=("calib_cells", "mean"), calib_age_median_h=("calib_age_h", "median"),
        pool_A_rows=("in_pool_A", "sum"), pool_AVAIL_rows=("in_avail", "sum")).reset_index()
    for c in ("sensor_rows_pct", "sensor_cells_pct", "calib_rows_pct", "calib_cells_pct"):
        tot[c] *= 100
    tot["calib_stale_pct"] = [100 * rows[(rows.device == d)].stale.sum()
                              / max(rows[(rows.device == d)].has_calib.sum(), 1)
                              if rows[(rows.device == d)].has_calib.any() else np.nan
                              for d in tot.device]
    tot["both_fresh_pct"] = [100 * (rows[rows.device == d].state == "both").mean()
                             for d in tot.device]
    tot["device"] = tot.device.map(LABEL)
    tot["month"] = "all"
    t = pd.concat([t, tot[t.columns]], ignore_index=True)
    return t.round(1)


def channel_coverage(stage09: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per-channel monthly coverage from stage 09 (keeps dropped channels)."""
    names = C.dataset_columns(stage09)
    sensors, _ = C.resolve_device_columns(names)
    # One presence column per sensor channel: __mean, or __value for policy channels.
    sens_col = {}
    for d in C.PAPER_DEVICES:
        for c in sensors.get(d, []):
            m = SENSOR_AGG_RX.match(c)
            if m:
                sens_col[(d, m.group(1))] = c
    calib_cols = [c for c in names if CALIB_RX.match(c)]
    df = pq.read_table(str(stage09), columns=[C.DEVICE_COL, C.TIME_COL]
                       + sorted(set(sens_col.values())) + calib_cols).to_pandas()
    df = df[df[C.DEVICE_COL].isin(C.PAPER_DEVICES)].reset_index(drop=True)
    df["month"] = month_of(df[C.TIME_COL])
    jobs = df.groupby([C.DEVICE_COL, "month"]).size()

    recs = []
    for (d, base), col in sens_col.items():
        sub = df[df[C.DEVICE_COL] == d]
        cov = pd.to_numeric(sub[col], errors="coerce").notna().groupby(sub.month).mean()
        recs += [{"device": LABEL[d], "channel": N.channel_name(col), "month": mo,
                  "share": v} for mo, v in cov.items()]
    sens_cov = pd.DataFrame(recs)

    q = df[df[C.DEVICE_COL] == C.QEXA]
    vals = q[calib_cols].apply(pd.to_numeric, errors="coerce")
    # Fidelities arrive from the monitoring system as integers, so they truncate to 0 (or 1).
    has, info = vals.notna(), vals.notna() & ~vals.isin([0, 1])
    recs = []
    for c in calib_cols:
        target, metric = CALIB_RX.match(c).groups()
        # Release vocabulary: q01..q20, coupler pairs as "pair i-j", usable
        # metrics by their release name (t1, t2echo, readout_err_0to1, ...).
        target = (f"q{int(target[2:]):02d}" if target.startswith("QB")
                  else "pair " + "-".join(target.split("_")[1:]))
        metric = N.CALIBRATION_METRICS.get(metric, metric)
        hs = has[c].groupby(q.month).mean()
        ns = info[c].groupby(q.month).mean()
        recs += [{"metric": metric, "target": target, "month": mo,
                  "share_present": hs[mo], "share_not_0_or_1": ns[mo]} for mo in hs.index]
    cal_cov = pd.DataFrame(recs)
    return sens_cov, cal_cov, {"jobs": jobs, "calib_cols": calib_cols}


def metric_status(cal_cov: pd.DataFrame) -> dict[str, str]:
    st = {}
    for m, g in cal_cov.groupby("metric"):
        if g.share_present.max() == 0:
            st[m] = "never published"
        elif g.share_not_0_or_1.max() == 0:
            st[m] = "only 0 or 1, integer-truncated"
        else:
            st[m] = "usable"
    return st


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_availability(rows: pd.DataFrame, outdir: Path):
    months = months_between(rows.month.min(), rows.month.max())
    x = np.arange(len(months))
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.6), sharex=True,
                             gridspec_kw={"hspace": 0.42, "height_ratios": [1, 1, 0.8]})
    fig.patch.set_facecolor(SURFACE)

    for ax, d in zip(axes[:2], C.PAPER_DEVICES):
        sub = rows[rows.device == d]
        share = (sub.groupby(["month", "state"]).size().unstack(fill_value=0)
                 .reindex(index=months, fill_value=0))
        n = share.sum(axis=1)
        frac = share.div(n.replace(0, np.nan), axis=0).fillna(0)
        bottom = np.zeros(len(months))
        for key, _, colour in STATES:
            v = frac[key].to_numpy() if key in frac else np.zeros(len(months))
            # 2px surface gap between stacked fills.
            ax.bar(x, v, bottom=bottom, width=0.82, color=colour,
                   edgecolor=SURFACE, linewidth=0.8, zorder=3)
            bottom += v
        for i, mo in enumerate(months):
            if 0 < n[mo] < LOW_N:
                ax.text(i, 1.03, "†", ha="center", va="bottom", fontsize=7, color=INK_MUTED)
            elif n[mo] == 0:
                ax.text(i, 0.5, "no jobs", rotation=90, ha="center", va="center",
                        fontsize=6, color=INK_MUTED)
        style(ax)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_yticks([0, 0.5, 1])
        avail = int(sub.in_avail.sum())
        ax.set_title(f"{LABEL[d]}: share of jobs by data present   "
                     f"(pool AVAIL {avail:,} of {len(sub):,} rows)",
                     loc="left", fontsize=9.3, color=INK, pad=5)

    ax = axes[2]
    q = rows[(rows.device == C.QEXA) & rows.has_calib]
    stale = q.groupby("month").stale.mean().reindex(months)
    age = q.groupby("month").calib_age_h.median().reindex(months)
    ax.bar(x, stale.fillna(0).to_numpy(), width=0.82, color=SERIES[C.QEXA], zorder=3)
    for i, mo in enumerate(months):
        if stale.get(mo, 0) >= 0.10:
            ax.text(i, stale[mo] + 0.03, f"{age[mo]:.0f} h", ha="center", va="bottom",
                    fontsize=6.5, color=INK_2)
    style(ax)
    ax.set_ylim(0, max(0.5, float(stale.max() or 0) * 1.3))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.set_title(f"Q-Exa: calibration older than {C.CALIB_MAX_AGE_H:.0f} h, share of jobs "
                 f"with calibration (label = median age)", loc="left", fontsize=9.3,
                 color=INK, pad=5)
    month_ticks(ax, months)

    handles = [Patch(facecolor=c, label=l) for _, l, c in STATES]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.915), ncol=4,
               frameon=False, fontsize=7.8, labelcolor=INK_2, handlelength=1.0,
               columnspacing=1.2)
    fig.suptitle("Where the sensor and calibration data exist", x=0.012, ha="left",
                 fontsize=11.5, color=INK, y=0.985)
    fig.text(0.012, 0.935,
             f"Stage 10 training rows. Fresh = calibration sample no older than "
             f"{C.CALIB_MAX_AGE_H:.0f} h; older values are counted as absent.  "
             f"† fewer than {LOW_N} jobs that month.",
             ha="left", fontsize=7.4, color=INK_2)
    fig.subplots_adjust(top=0.845, bottom=0.06, left=0.075, right=0.985)
    save(fig, outdir, "data_availability")


def fig_calibration_onset(rows: pd.DataFrame, cal_cov: pd.DataFrame, status: dict,
                          outdir: Path):
    months = months_between(rows.month.min(), rows.month.max())
    idx = {m: i for i, m in enumerate(months)}
    x = list(range(len(months)))
    fig, (ax_j, ax_c) = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True,
                                     gridspec_kw={"hspace": 0.30})
    fig.patch.set_facecolor(SURFACE)

    width = 0.38
    for k, d in enumerate(C.PAPER_DEVICES):
        j = rows[rows.device == d].groupby("month").size()
        vals = [int(j.get(m, 0)) for m in months]
        off = (k - 0.5) * width
        ax_j.bar([i + off for i in x], vals, width=width * 0.94, color=SERIES[d],
                 zorder=3, label=LABEL[d])
    style(ax_j)
    ax_j.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_j.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0"))
    ax_j.set_ylabel("circuit jobs", fontsize=8.5, color=INK_2)
    ax_j.set_title("Circuit jobs executed", loc="left", fontsize=9.5, color=INK, pad=6)
    ax_j.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK_2,
                handlelength=1.1, borderpad=0.1)

    # Onset: first month a channel returned a value; usable = t1/t2/t2e/error.
    first = (cal_cov[cal_cov.share_present > 0].groupby(["metric", "target"]).month.min())
    usable = first[[status[m] == "usable" for m in first.index.get_level_values(0)]]
    n_all, n_use = len(cal_cov[["metric", "target"]].drop_duplicates()), len(
        cal_cov[cal_cov.metric.map(status) == "usable"][["metric", "target"]].drop_duplicates())
    for series, ls, lab in ((first, (0, (4, 2)), "incl. unusable metrics"),
                            (usable, "-", "usable metrics")):
        cnt = series.value_counts()
        cum = np.cumsum([int(cnt.get(m, 0)) for m in months])
        ax_c.step(x, cum, where="post", color=SERIES[C.QEXA], lw=2.0, ls=ls, zorder=3)
        ax_c.annotate(f"Q-Exa, {lab}  {cum[-1]}", (x[-1], cum[-1]), textcoords="offset points",
                      xytext=(-4, 5), ha="right", fontsize=7.8, color=INK_2, zorder=5)
    style(ax_c)
    ax_c.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    ax_c.set_ylabel("channels online", fontsize=8.5, color=INK_2)
    ax_c.set_title("Calibration channels (metric × qubit or pair) returning data",
                   loc="left", fontsize=9.5, color=INK, pad=6)
    ax_c.set_ylim(0, first.value_counts().sum() * 1.3)

    # The tall 2024-12/2025-01 bars leave no room for an in-plot note, so the
    # first-wave share goes in the subtitle.
    q = rows[rows.device == C.QEXA]
    wave = usable.value_counts().sort_index()
    wave = wave[wave >= 5].index.min() if len(wave) else None
    wave_note = ""
    if wave in idx:
        for ax in (ax_j, ax_c):
            ax.axvline(idx[wave] - 0.5, color=INK_MUTED, lw=1.0, ls=(0, (3, 3)), zorder=2)
        wave_note = (f"Dashed line: first calibration wave ({wave}); "
                     f"{100 * (q.month < wave).mean():.1f}% of Q-Exa jobs ran before it.  ")
    month_ticks(ax_c, months)

    never = sorted(m for m, s in status.items() if s == "never published")
    fig.suptitle("Calibration covers most of the circuit record", x=0.012, ha="left",
                 fontsize=11.5, color=INK, y=0.985)
    fig.text(0.012, 0.935,
             wave_note + f"Marmot publishes no calibration.\nSuperconducting: {n_use} usable of "
             f"{n_all} channels; never published: {', '.join(never) or 'none'}.",
             va="top",
             ha="left", fontsize=7.4, color=INK_2)
    fig.subplots_adjust(top=0.83, bottom=0.095, left=0.088, right=0.985)
    save(fig, outdir, "calibration_onset")


def fig_calibration_timeline(cal_cov: pd.DataFrame, status: dict, jobs: pd.Series,
                             outdir: Path):
    qjobs = jobs.xs(C.QEXA, level=0)
    months = months_between(min(qjobs.index), max(qjobs.index))
    order = sorted(status, key=lambda m: ({"usable": 0}.get(status[m], 1), m))
    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.4, 2.05 * nrow + 1.1), squeeze=False,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.28})
    fig.patch.set_facecolor(SURFACE)

    def tkey(t):
        return tuple(int(v) for v in re.findall(r"\d+", t))

    im = None
    for ax, metric in zip(axes.flat, order):
        g = cal_cov[cal_cov.metric == metric]
        targets = sorted(g.target.unique(), key=tkey)
        piv = (g.pivot(index="target", columns="month", values="share_present")
               .reindex(index=targets, columns=months))
        grid = piv.to_numpy(dtype=float)
        grid[:, [qjobs.get(m, 0) == 0 for m in months]] = np.nan
        ramp = RAMP_USABLE if status[metric] == "usable" else RAMP_UNUSABLE
        cmap = LinearSegmentedColormap.from_list(metric, ramp)
        cmap.set_bad(GRID)
        img = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                        interpolation="nearest")
        if status[metric] == "usable":
            im = img
        ax.set_yticks([0, len(targets) - 1])
        ax.set_yticklabels([targets[0], targets[-1]], fontsize=5.5, color=INK_2)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels([m[2:] if m.endswith(("-01", "-07")) else "" for m in months],
                           fontsize=5.8, color=INK_2)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        colour = INK if status[metric] == "usable" else INK_MUTED
        title = metric
        if status[metric] != "usable":
            title += f"\n{status[metric]}"
        ax.set_title(title, loc="left", fontsize=7.4 if status[metric] == "usable" else 6.6,
                     color=colour, pad=3)
    for ax in list(axes.flat)[len(order):]:
        ax.set_visible(False)

    if im is not None:
        cax = fig.add_axes([0.80, 0.955 - 0.02, 0.17, 0.012])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["0%", "100%"])
        cb.ax.tick_params(labelsize=6.5, colors=INK_2, length=0, pad=2)
        cb.outline.set_visible(False)

    H = fig.get_size_inches()[1]
    fig.suptitle("Calibration availability per metric, qubit and coupler", x=0.012,
                 ha="left", fontsize=11.5, color=INK, y=1 - 0.2 / H)
    fig.text(0.012, 1 - 0.46 / H,
             "Q-Exa. Rows = qubits q01-q20 or coupler pairs; shade = share of that month's "
             "jobs with a value (any age).\nBlue = released as sc_cal_q<NN>_<metric>; grey = "
             "not released (unusable or never published); light grey cell = no jobs.",
             ha="left", fontsize=7.2, color=INK_2, va="top")
    fig.subplots_adjust(top=1 - 1.05 / H, bottom=0.05, left=0.07, right=0.985)
    save(fig, outdir, "calibration_timeline")


# ---------------------------------------------------------------------------
# Analysis sets: definitions and why AVAIL keeps the rows it keeps
# ---------------------------------------------------------------------------

def _by_month(r: pd.DataFrame, mask: pd.Series, top: int = 3) -> str:
    vc = r.loc[mask, "month"].value_counts().sort_values(ascending=False)
    parts = [f"{m} ({n:,})" for m, n in vc.head(top).items()]
    return ", ".join(parts) + (f", +{len(vc) - top} more months" if len(vc) > top else "")


def avail_selection(rows: pd.DataFrame) -> pd.DataFrame:
    """Step-by-step: from every job of a device to the AVAIL rows used in training."""
    recs = []
    for d in C.PAPER_DEVICES:
        r = rows[rows.device == d]
        first = r.loc[r.has_sensor, "month"].min()
        steps = [("All jobs", pd.Series(True, index=r.index), "")]
        started = r.month >= first
        steps.append((f"Ran in or after {first}, the first month with a sensor reading",
                      started, "sensors were not logging yet"))
        k = started & r.has_sensor
        steps.append(("Has at least one sensor reading in its 5-minute window", k,
                      "no reading in the window (sensors off or outage)"))
        if r.has_calib.any():
            k = k & r.has_calib
            steps.append(("Has calibration", k, "no calibration published for the job"))
            k = k & ~r.stale
            steps.append((f"Calibration no older than {C.CALIB_MAX_AGE_H:.0f} h", k,
                          "calibration stale (days to weeks old)"))
        else:
            steps.append(("Calibration: not required (Marmot publishes none)", k, ""))
        if k.sum() != r.in_avail.sum():
            raise AssertionError(f"{d}: selection {k.sum()} != AVAIL {r.in_avail.sum()}")
        steps.append(("= pool AVAIL", k, ""))
        steps.append(("Has a Hellinger target (rows the models train and test on)",
                      k & r.has_target, "job result could not be split per circuit"))
        prev = None
        for i, (label, mask, why) in enumerate(steps):
            n = int(mask.sum())
            removed = 0 if prev is None or label.startswith("=") else int((prev & ~mask).sum())
            recs.append({"device": LABEL[d], "step": i, "criterion": label, "rows": n,
                         "removed": removed,
                         "why_removed": (f"{why}; by month: {_by_month(r, prev & ~mask)}"
                                         if removed else ""),
                         "share_of_device_pct": round(100 * n / len(r), 1)})
            prev = mask
    return pd.DataFrame(recs)


SET_DEFINITIONS = [
    # (CLI name, rule, used by)
    ("(none) — every month",
     "Every job of the device, whatever data it carries.",
     "Circuit-only models: Tier 1, LODO, shot histogram"),
    ("A (default)",
     "Whole months in which sensors were logging. Q-Exa: 2025-04 onward without the "
     "2025-09 sensor outage. Marmot: 2025-03 onward.",
     "Tier 2a/2b and matched Tier 3; Tasks 5, 7, 8. Results in experiments/results/"),
    ("B",
     "Pool A without Q-Exa 2025-04 and 2025-05, when calibration was barely published. "
     "Marmot as in A.",
     "Robustness check (--pool B)"),
    ("AVAIL",
     "Individual jobs that carry the device's own data. Q-Exa: at least one sensor reading "
     f"and calibration no older than {C.CALIB_MAX_AGE_H:.0f} h. Marmot: at least one sensor "
     "reading. Chosen once on the 5-minute dataset, so every window uses the same rows.",
     "Same experiments as A (--pool AVAIL). Results in experiments/results/pool_AVAIL/"),
]


def analysis_sets(rows: pd.DataFrame) -> pd.DataFrame:
    masks = {"(none) — every month": pd.Series(True, index=rows.index),
             "A (default)": rows.in_pool_A, "B": rows.in_pool_B, "AVAIL": rows.in_avail}
    recs = []
    for name, rule, used in SET_DEFINITIONS:
        rec = {"set (--pool)": name, "rule": rule, "used by": used}
        for d in C.PAPER_DEVICES:
            m = masks[name] & (rows.device == d)
            rec[f"{LABEL[d]} rows"] = int(m.sum())
            rec[f"{LABEL[d]} rows with target"] = int((m & rows.has_target).sum())
        recs.append(rec)
    return pd.DataFrame(recs)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if pd.isna(v) else f"{v:,}" if isinstance(v, (int, np.integer))
                                     else str(v) for v in r) + " |")
    return "\n".join(out)


def write_analysis_sets(sets: pd.DataFrame, sel: pd.DataFrame, path: Path):
    body = ["# Analysis sets", "",
            "Which rows each `--pool` trains on (experiments/common.py). Row counts are "
            "stage 10 rows; \"with target\" are the rows that have a Hellinger distance "
            "and so enter training and testing.", "", md_table(sets), "",
            "# Why pool AVAIL keeps the rows it keeps", "",
            "Each step keeps the rows that pass it; `removed` counts the rows the step "
            "drops, and the months where most of them fall.", ""]
    for dev, g in sel.groupby("device", sort=False):
        body += [f"## {dev}", "", md_table(g.drop(columns=["device", "step"])), ""]
    path.write_text("\n".join(body))
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Sensor onset and per-channel timeline from the full data (release names)
# ---------------------------------------------------------------------------

def configured_channels() -> dict[str, list[str]]:
    """Release channel names per device, from the extraction config."""
    import json

    cfg = json.loads(N.SENSOR_CONFIG.read_text())
    out = {}
    for d in C.PAPER_DEVICES:
        paths = [p for ps in cfg[d]["sensor_categories"].values() for p in ps]
        out[d] = [N.STEMS[N.column_key(p)] for p in paths]
    return out


def _jobs_panel(ax, rows, months):
    x = list(range(len(months)))
    width = 0.38
    for k, d in enumerate(C.PAPER_DEVICES):
        j = rows[rows.device == d].groupby("month").size()
        ax.bar([i + (k - 0.5) * width for i in x], [int(j.get(m, 0)) for m in months],
               width=width * 0.94, color=SERIES[d], zorder=3, label=LABEL[d])
    style(ax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0"))
    ax.set_ylabel("circuit jobs", fontsize=8.5, color=INK_2)
    ax.set_title("Circuit jobs executed", loc="left", fontsize=9.5, color=INK, pad=6)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK_2,
              handlelength=1.1, borderpad=0.1)


def fig_sensor_onset(rows, sens_cov, outdir: Path):
    months = months_between(rows.month.min(), rows.month.max())
    idx = {m: i for i, m in enumerate(months)}
    x = list(range(len(months)))
    conf = configured_channels()
    fig, (ax_j, ax_c) = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True,
                                     gridspec_kw={"hspace": 0.30})
    fig.patch.set_facecolor(SURFACE)
    _jobs_panel(ax_j, rows, months)

    first = sens_cov[sens_cov.share > 0].groupby(["device", "channel"]).month.min()
    notes, top = [], 0
    for d in C.PAPER_DEVICES:
        f = first.xs(LABEL[d], level=0) if LABEL[d] in first.index.get_level_values(0) else pd.Series(dtype=str)
        cnt = f.value_counts()
        cum = np.cumsum([int(cnt.get(m, 0)) for m in months])
        top = max(top, cum[-1])
        ax_c.step(x, cum, where="post", color=SERIES[d], lw=2.0, zorder=3)
        ax_c.annotate(f"{LABEL[d]}  {cum[-1]} of {len(conf[d])}", (x[-1], cum[-1]),
                      textcoords="offset points", xytext=(-4, 5), ha="right", fontsize=8,
                      color=INK_2, zorder=5)
        if len(conf[d]) > cum[-1]:
            notes.append(f"{LABEL[d]}: {len(conf[d]) - cum[-1]} configured channel(s) never report")
    style(ax_c)
    ax_c.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    ax_c.set_ylabel("channels online", fontsize=8.5, color=INK_2)
    ax_c.set_title("Sensor channels returning data", loc="left", fontsize=9.5, color=INK, pad=6)
    ax_c.set_ylim(0, top * 1.3)

    q = rows[rows.device == C.QEXA]
    fq = first.xs(LABEL[C.QEXA], level=0).value_counts().sort_index()
    wave = fq[fq >= 5].index.min() if len(fq) else None
    wave_note = ""
    if wave in idx:
        for ax in (ax_j, ax_c):
            ax.axvline(idx[wave] - 0.5, color=INK_MUTED, lw=1.0, ls=(0, (3, 3)), zorder=2)
        wave_note = (f"Dashed line: Q-Exa's first sensor wave ({wave}); "
                     f"{100 * (q.month < wave).mean():.0f}% of Q-Exa jobs ran before it.  ")
    month_ticks(ax_c, months)
    fig.suptitle("The circuit record predates the telemetry record", x=0.012, ha="left",
                 fontsize=11.5, color=INK, y=0.985)
    fig.text(0.012, 0.935, wave_note + "\nFull dataset; onset = first month a channel returned "
             "a reading in a job's window." + ("  " + "; ".join(notes) + "." if notes else ""),
             ha="left", fontsize=7.4, color=INK_2, va="top")
    fig.subplots_adjust(top=0.83, bottom=0.095, left=0.088, right=0.985)
    save(fig, outdir, "sensor_onset")


def fig_sensor_timeline(sens_cov, jobs: pd.Series, trained: set[str], outdir: Path):
    conf = configured_channels()
    all_months = sorted({m for (_, m) in jobs.index})
    months = months_between(all_months[0], all_months[-1])
    heights = [len(conf[d]) for d in C.PAPER_DEVICES]
    fig, axes = plt.subplots(len(heights), 1, figsize=(7.4, 0.105 * sum(heights) + 2.6),
                             gridspec_kw={"height_ratios": heights, "hspace": 0.08})
    fig.patch.set_facecolor(SURFACE)
    for ax, d in zip(axes, C.PAPER_DEVICES):
        cov = sens_cov[sens_cov.device == LABEL[d]].pivot(index="channel", columns="month",
                                                          values="share")
        cov = cov.reindex(index=conf[d], columns=months)
        n_jobs = jobs.xs(d, level=0).reindex(months).fillna(0)

        def onset(ch):
            r = cov.loc[ch]
            on = r[r > 0].index
            return (on[0] if len(on) else "9999", -r.fillna(0).sum())
        order = sorted(conf[d], key=onset)
        grid = cov.loc[order].to_numpy(dtype=float)
        grid = np.where(np.isnan(grid), 0.0, grid)
        grid[:, (n_jobs == 0).to_numpy()] = np.nan
        ramp = RAMP_USABLE if d == C.QEXA else ["#fcfcfb", "#f6b79b", "#eb6834", "#8c3113"]
        cmap = LinearSegmentedColormap.from_list(d, ramp)
        cmap.set_bad(GRID)
        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=5.2)
        for lab, ch in zip(ax.get_yticklabels(), order):
            lab.set_color(INK_2 if ch in trained else INK_MUTED)
            if ch not in trained:
                lab.set_fontstyle("italic")
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels([m if m.endswith(("-01", "-07")) else "" for m in months],
                           fontsize=7, color=INK_2)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        dead = sum(1 for ch in order if not (cov.loc[ch].fillna(0) > 0).any())
        dropped = sum(1 for ch in order if ch not in trained)
        ax.set_title(f"{LABEL[d]}: {len(order)} channels, {dropped} not used in the models"
                     + (f" ({dead} never report)" if dead else ""),
                     loc="left", fontsize=9.3, color=INK, pad=5)
        cax = ax.inset_axes([1.006, 0.0, 0.012, 1.0])
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["0%", "100%"])
        cb.ax.tick_params(labelsize=6.5, colors=INK_2, length=0, pad=2)
        cb.outline.set_visible(False)
    H = fig.get_size_inches()[1]
    fig.suptitle("Telemetry availability per sensor channel", x=0.012, ha="left",
                 fontsize=11.5, color=INK, y=1 - 0.26 / H)
    fig.text(0.012, 1 - 0.56 / H,
             "Full dataset. Shade = share of that month's jobs with a reading in the "
             "window; grey = no jobs that month.\nRows ordered by first data; release "
             "channel names. Grey italic = dropped before training (dead or constant).",
             ha="left", fontsize=7.6, color=INK_2, va="top")
    fig.subplots_adjust(top=1 - 1.15 / H, bottom=0.03, left=0.25, right=0.945)
    save(fig, outdir, "sensor_timeline")


def write_markdown(table: pd.DataFrame, path: Path):
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in table.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else f"{v:,}" if isinstance(v, (int, np.integer))
                                        else str(v) for v in r) + " |")
    path.write_text(
        "# Sensor and calibration availability\n\n"
        "Generated by `analysis/data_availability_report.py` from the stage 10 training rows. "
        "Percentages are of that device-month's jobs, except `calib_stale_pct` (of jobs with "
        f"calibration; stale = older than {C.CALIB_MAX_AGE_H:.0f} h) and the `*_cells_pct` "
        "columns (mean share of the device's sensor or calibration feature cells filled). "
        "`pool_AVAIL_rows` are the rows experiments use with `--pool AVAIL`.\n\n"
        + "\n".join(lines) + "\n")
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage10", default=str(C.DEFAULT_DATASET))
    ap.add_argument("--stage09", default=str(C.CALIB_AGE_SOURCE))
    ap.add_argument("--outdir", default=str(REPO / "analysis" / "figures"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("reading row states from stage 10 ...")
    rows = row_states(Path(args.stage10))
    table = report_table(rows)
    table.to_csv(outdir / "data_availability.csv", index=False)
    print(f"wrote {outdir / 'data_availability.csv'}")
    write_markdown(table, outdir / "data_availability.md")

    print("reading channel coverage from stage 09 ...")
    sens_cov, cal_cov, extra = channel_coverage(Path(args.stage09))
    sens_cov.round(4).to_csv(outdir / "sensor_channel_coverage.csv", index=False)
    cal_cov.round(4).to_csv(outdir / "calibration_channel_coverage.csv", index=False)
    print(f"wrote {outdir / 'sensor_channel_coverage.csv'}, calibration_channel_coverage.csv")
    status = metric_status(cal_cov)
    for m, s in sorted(status.items()):
        print(f"  calibration metric {m:32s} {s}")

    sets = analysis_sets(rows)
    sel = avail_selection(rows)
    sets.to_csv(outdir / "analysis_sets.csv", index=False)
    sel.to_csv(outdir / "avail_row_selection.csv", index=False)
    write_analysis_sets(sets, sel, outdir / "analysis_sets.md")

    s10 = C.dataset_columns(Path(args.stage10))
    trained = {N.channel_name(c) for c in s10 if N.channel_name(c)}
    fig_availability(rows, outdir)
    fig_sensor_onset(rows, sens_cov, outdir)
    fig_sensor_timeline(sens_cov, extra["jobs"], trained, outdir)
    fig_calibration_onset(rows, cal_cov, status, outdir)
    fig_calibration_timeline(cal_cov, status, extra["jobs"], outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
