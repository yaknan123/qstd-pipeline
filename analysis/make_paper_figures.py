#!/usr/bin/env python3
"""Regenerate the figures in the QSTD paper from the published dataset.

The paper's figures were first drawn in the build repository's own analysis
notebook, which is not part of this package: it is written against the internal
column paths and device codes. This script draws the same figures from the
released dataset, using the released column names, so that a reader who
downloads qstd_v1.0 can regenerate every figure in the paper.

It is written against experiments/common.py, which resolves the schema
difference for us: DEVICE_COL, TIME_COL and GROUP_COL each take their published
or internal spelling depending on which dataset is loaded. The same script
therefore runs on the release and on the build repository.

    python3 analysis/make_paper_figures.py                  # every figure
    python3 analysis/make_paper_figures.py --fast           # skip the ones that train a model
    python3 analysis/make_paper_figures.py --only plot3 plot8
    python3 analysis/make_paper_figures.py --list

Figures that train a model (the SHAP panels, the model comparison and the
confusion matrices) take a few minutes each on 1.5M rows; --fast draws only the
ones that come straight from the data.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
import common as C  # noqa: E402

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.05)
    HAS_SNS = True
except ImportError:                                    # heatmaps fall back to imshow
    HAS_SNS = False

SC, ION = C.SC, C.ION
DEV, TGT, TIME = C.DEVICE_COL, C.TARGET_COL, C.TIME_COL

# The systems' public names, and the technology each one is. These are the only
# device names that appear in a figure.
LABEL = {SC: "Q-Exa", ION: "Marmot"}
TECH = {SC: "Superconducting (CZ native gate)", ION: "Ion trap (MS native gate)"}
TECH_SHORT = {SC: "Superconducting", ION: "Ion trap"}
COLOR = {SC: "#0072B2", ION: "#D55E00"}                # Wong 2011, colourblind-safe

H_GOOD_MAX, H_POOR_MIN = C.CLASS_THRESHOLD_GOOD, C.CLASS_THRESHOLD_POOR

# Explicit integer bin edges per device, shared by the gate-count distribution
# and the 2D heatmap so the two figures are read against the same grid. The
# ranges differ by two orders of magnitude between the devices -- the ion trap
# runs MS-decomposed circuits of tens of gates, the superconducting device
# CZ-transpiled ones of tens of thousands -- so a shared axis would compress
# one device into a single bar.
BIN_EDGES = {
    ION: {"2q_gates": [0, 5, 10, 15, 20, 30, 50, 100, 500],
          "depth":    [0, 10, 25, 50, 100, 200, 400, 800, 2000]},
    SC:  {"2q_gates": [0, 10, 30, 80, 150, 300, 700, 2000, 25000],
          "depth":    [0, 20, 50, 120, 250, 500, 1200, 3000, 15000]},
}

# Sensor categories, by released column name. "Temperature" is not one thing:
# the mixing chamber sits at ~10 mK and is the qubit operating point, the 4 K
# and 50 K stages and the still are the cryostat's intermediate stages, and the
# ion lab's probes read room ambient. Lumping them together would average a
# millikelvin signal with a room thermometer, so each is its own category.
SENSOR_CATEGORIES = {
    "mixing_temp": ("sc_cryo_mxc_temp",),
    "stage_temp":  ("sc_cryo_4k_temp", "sc_cryo_50k_temp", "sc_cryo_still_temp"),
    "coolant_temp": ("sc_cooling_return_temp", "sc_cooling_supply_temp"),
    "room_temp":   ("ion_lab_air_temp", "ion_lab_probe"),
    "pressure":    ("sc_cryo_p1", "sc_cryo_p2", "sc_cryo_p3", "sc_cryo_p4",
                    "sc_cryo_p5", "sc_cryo_p6", "ion_lab_air_pressure"),
    "humidity":    ("ion_lab_humidity",),
    "flow":        ("sc_cryo_mix_flow",),
}
# Spelled out for the reader. "Cold-stage" is the 4 K / 50 K / still plates, and
# is deliberately a different row from the mixing chamber.
CATEGORY_LABELS = {
    "mixing_temp": "Mixing-chamber temperature (~10 mK)",
    "stage_temp": "Cold-stage temperature (4 K / 50 K / still)",
    "coolant_temp": "Coolant temperature (supply / return)",
    "room_temp": "Room / lab temperature",
    "pressure": "Pressure",
    "humidity": "Humidity",
    "flow": "Mixture flow",
}
CATEGORY_ORDER = ["mixing_temp", "stage_temp", "coolant_temp", "room_temp",
                  "pressure", "humidity", "flow"]

CATEGORY_COLOUR = {"circuit": "#3d5a80", "sensor": "#ee9b00", "calibration": "#9b2226"}

OUT = REPO / "analysis" / "paper_figures"
REGISTRY: dict[str, dict] = {}


def figure(name: str, *, needs_model: bool = False, description: str = ""):
    """Register a figure so --only and --list can address it by name."""
    def deco(fn):
        REGISTRY[name] = {"fn": fn, "needs_model": needs_model,
                          "description": description or (fn.__doc__ or "").strip()}
        return fn
    return deco


def devices_in(df) -> list[str]:
    """The paper's two devices, in a fixed order, as present in this frame."""
    have = set(df[DEV].astype(str).unique())
    return [d for d in (ION, SC) if d in have]


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {stem}.png / .pdf")


def sensor_columns(df) -> list[str]:
    """Released sensor columns present in this frame."""
    return [c for c in df.columns if c.startswith(("sc_cryo_", "sc_cooling_", "ion_lab_"))]


def calibration_columns(df) -> list[str]:
    return [c for c in df.columns if c.startswith("sc_cal_")]


def classify_sensor(col: str) -> str | None:
    for cat, prefixes in SENSOR_CATEGORIES.items():
        if any(col.startswith(p) for p in prefixes):
            return cat
    return None


def int_bin_label(lo, hi, is_last: bool) -> str:
    """Integer bins are half-open [lo, hi); render them so the edge is unambiguous."""
    return f"≥{int(lo)}" if is_last else f"{int(lo)}–{int(hi) - 1}"


def median_trend(ax, x, y, *, n_groups: int = 40, min_per_group: int = 20):
    """Equal-count groups along x, median of y in each: robust to outliers.

    Returns the elbow (largest positive step in the medians), which is where
    the fidelity starts falling away with circuit size.
    """
    bins = np.unique(np.quantile(np.clip(x, 0, None), np.linspace(0, 1, n_groups)))
    bi = np.digitize(x, bins, right=True)
    mx, my = [], []
    for b in np.unique(bi):
        s = y[bi == b]
        if len(s) >= min_per_group:
            mx.append(np.median(x[bi == b]))
            my.append(np.median(s))
    if not mx:
        return None
    ax.plot(mx, my, "k-", lw=2.5, zorder=10,
            label=f"median trend (n≥{min_per_group} per group)")
    if len(mx) >= 3:
        d = np.diff(my)
        i = int(np.argmax(d))
        return (mx[i] + mx[i + 1]) / 2, d[i]
    return None


# ---------------------------------------------------------------------------
# Figures drawn straight from the dataset
# ---------------------------------------------------------------------------

@figure("plot1", description="H_logical vs H_native, per device, with fidelity regimes")
def plot1_hlog_vs_hnat(df) -> None:
    devs = devices_in(df)
    fig, axes = plt.subplots(1, len(devs), figsize=(4.2 * len(devs) + 0.8, 5.1),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    rows = []
    for ax, d in zip(axes, devs):
        # Both targets must be present: the fit and the diagonal comparison are
        # meaningless on a row where either is null, and polyfit raises on NaN.
        t = df.loc[df[DEV].astype(str) == d,
                   ["hellinger_logical", TGT]].dropna()
        hl = t["hellinger_logical"].values
        hn = t[TGT].values
        col = COLOR[d]

        for lo, hi, c in ((0, H_GOOD_MAX, "#4CAF50"), (H_GOOD_MAX, H_POOR_MIN, "#FFC107"),
                          (H_POOR_MIN, 1.0, "#F44336")):
            ax.axhspan(lo, hi, facecolor=c, alpha=0.08, zorder=0)
        ax.axhline(H_GOOD_MAX, color="#2E7D32", ls=":", lw=0.9, alpha=0.7, zorder=1)
        ax.axhline(H_POOR_MIN, color="#C62828", ls=":", lw=0.9, alpha=0.7, zorder=1)

        ax.scatter(hl, hn, s=4, alpha=0.05 if len(t) > 500_000 else 0.25,
                   color=col, edgecolors="none", rasterized=True, zorder=3)
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="$y = x$", zorder=5)

        coef = np.polyfit(hl, hn, 1)
        r = np.corrcoef(hl, hn)[0, 1]
        above = float((hn > hl).mean())
        xs = np.linspace(0, 1, 100)
        ax.plot(xs, np.polyval(coef, xs), color="#222222", lw=1.6,
                label=f"Fit: slope={coef[0]:.3f}, r={r:.3f}", zorder=6)

        mx, my = float(np.median(hl)), float(np.median(hn))
        ax.scatter(mx, my, s=130, marker="D", edgecolor="white", facecolor=col,
                   linewidth=2, zorder=12, label=f"Median ({mx:.3f}, {my:.3f})")

        ax.text(0.97, 0.03,
                f"N = {len(t):,}\nμ = {hn.mean():.3f}\nσ = {hn.std():.3f}\n"
                f">diag: {above:.1%}",
                transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                family="monospace", zorder=11,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="gray", alpha=0.92))
        ax.set_title(f"{LABEL[d]} ({TECH_SHORT[d]})", fontweight="bold", color=col, pad=6)
        ax.set_xlabel(r"$H_{\mathrm{logical}}$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25, ls=":", zorder=0)
        ax.legend(loc="upper left", framealpha=0.93, fontsize=7.5)
        ax.set_aspect("auto")
        rows.append(dict(device=LABEL[d], n=len(t), slope=coef[0], r=r,
                         median_logical=mx, median_native=my, frac_above_diagonal=above))

    axes[0].set_ylabel(r"$H_{\mathrm{native}}$")
    right = axes[-1].twinx()
    right.set_ylim(1, 0)
    right.set_yticks(np.arange(0, 1.01, 0.2))
    right.set_ylabel(r"$F_H = 1 - H$  (higher is better)", fontsize=10)
    right.set_aspect("auto")
    for y, text, c in ((H_GOOD_MAX / 2, "GOOD", "#1B5E20"),
                       ((H_GOOD_MAX + H_POOR_MIN) / 2, "MEDIUM", "#E65100"),
                       ((H_POOR_MIN + 1) / 2, "POOR", "#B71C1C")):
        axes[-1].text(1.18, y, text, transform=axes[-1].get_yaxis_transform(),
                      fontsize=8, color=c, fontweight="bold", va="center",
                      ha="left", alpha=0.75, rotation=90)
    fig.tight_layout()
    save(fig, "plot1_Hlog_vs_Hnat")
    pd.DataFrame(rows).to_csv(OUT / "plot1_fits.csv", index=False)


@figure("plot2", description="logical vs native 2-qubit gate count distribution, per device")
def plot2_gate_distribution(df) -> None:
    devs = devices_in(df)
    fig, axes = plt.subplots(1, len(devs), figsize=(8.5 * len(devs), 5.2), squeeze=False)
    axes = axes[0]
    for ax, d in zip(axes, devs):
        sub = df.loc[df[DEV].astype(str) == d,
                     ["logical_n_2q_gates", "native_n_2q_gates"]].dropna()
        edges = np.array(BIN_EDGES[d]["2q_gates"], dtype=float)
        lg, _ = np.histogram(np.clip(sub["logical_n_2q_gates"], 0, edges[-1]), bins=edges)
        nt, _ = np.histogram(np.clip(sub["native_n_2q_gates"], 0, edges[-1]), bins=edges)
        x = np.arange(len(edges) - 1)
        ax.bar(x - 0.2, lg, 0.4, color="#6E6E6E", label="logical (as submitted)")
        ax.bar(x + 0.2, nt, 0.4, color=COLOR[d], label="native (after transpilation)")
        ax.set_xticks(x)
        ax.set_xticklabels([int_bin_label(edges[i], edges[i + 1], i == len(edges) - 2)
                            for i in x], rotation=35, ha="right", fontsize=9)
        ax.set_yscale("log")
        ax.set_xlabel("2-qubit gate count  [dimensionless integer]", fontsize=9)
        ax.set_ylabel("Number of circuits", fontsize=9)
        ax.set_title(f"{LABEL[d]}  —  {TECH[d]}", fontweight="bold", color=COLOR[d])
        med_l = sub["logical_n_2q_gates"].median()
        med_n = sub["native_n_2q_gates"].median()
        ax.text(0.97, 0.97,
                f"N = {len(sub):,}\nmedian logical = {med_l:,.0f}\n"
                f"median native  = {med_n:,.0f}\ninflation ×{med_n / max(med_l, 1):.1f}",
                transform=ax.transAxes, fontsize=8.5, ha="right", va="top",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="lightgray", alpha=0.93))
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "plot_2q_gate_distribution_by_device")


@figure("plot3", description="H_native vs native 2-qubit gate count, per device")
def plot3_hnat_vs_2q(df) -> None:
    xc = "native_n_2q_gates"
    devs = devices_in(df)
    p = df[[xc, TGT, DEV]].dropna()
    fig, axes = plt.subplots(1, len(devs), figsize=(7 * len(devs), 6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, d in zip(axes, devs):
        t = p[p[DEV].astype(str) == d]
        x, y = t[xc].values, t[TGT].values
        # Clip to the 99th percentile: beyond it the circuits are shots-starved,
        # and their apparently lower H is a selection artefact, not a physics
        # reversal.
        xmax = np.quantile(x, 0.99)
        ax.scatter(x, y, s=6, alpha=0.12, color=COLOR[d], edgecolors="none",
                   rasterized=True)
        elbow = median_trend(ax, x, y)
        if elbow:
            ax.axvline(elbow[0], color="purple", ls="--", lw=1.2, alpha=0.7,
                       label=f"elbow ≈ {elbow[0]:.0f} 2Q gates")
        ax.axhline(H_GOOD_MAX, color="green", ls=":", lw=1, alpha=0.6,
                   label=f"good/medium (H={H_GOOD_MAX})")
        ax.axhline(H_POOR_MIN, color="red", ls=":", lw=1, alpha=0.6,
                   label=f"medium/poor (H={H_POOR_MIN})")
        ax.text(0.03, 0.97,
                f"N = {len(t):,}\nPearson r = {np.corrcoef(x, y)[0, 1]:.3f}\n"
                f"Spearman ρ = {t[[xc, TGT]].corr(method='spearman').iloc[0, 1]:.3f}\n"
                f"x capped at 99th pct",
                transform=ax.transAxes, fontsize=9, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="lightgray", alpha=0.93))
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1)
        ax.set_title(f"{LABEL[d]}  —  {TECH[d]}", fontweight="bold", color=COLOR[d])
        ax.set_xlabel("Native 2-qubit gate count  [dimensionless integer]", fontsize=9)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    axes[0].set_ylabel("Hellinger Native  [dimensionless, 0–1]", fontsize=9)
    fig.tight_layout()
    save(fig, "plot3_Hnat_vs_2q")


@figure("plot5E", description="median |r| between sensor categories and the targets")
def plot5e_aggregate_heatmap(df) -> None:
    sensors = sensor_columns(df)
    rows = []
    for d in devices_in(df):
        t = df[df[DEV].astype(str) == d]
        cols = [c for c in sensors if t[c].notna().mean() > 0.10]
        for tgt, disp in ((TGT, "H_native"), ("hellinger_logical", "H_logical")):
            for c in cols:
                cat = classify_sensor(c)
                if cat is None:
                    continue
                r = t[[c, tgt]].corr().iat[0, 1]
                if pd.notna(r):
                    rows.append(dict(device=LABEL[d], target=disp, category=cat,
                                     sensor=c, abs_r=abs(r)))
    if not rows:
        print("    no sensor correlations available - skipped")
        return
    agg = (pd.DataFrame(rows).groupby(["device", "target", "category"])["abs_r"]
           .median().reset_index())
    agg.to_csv(OUT / "plot5E_aggregate.csv", index=False)

    targets = ["H_logical", "H_native"]
    # Wide, and laid out by constrained_layout: the category labels are long and
    # the left panel's colourbar otherwise lands on top of the right panel's.
    fig, axes = plt.subplots(1, len(targets), figsize=(8.4 * len(targets), 4.6),
                             squeeze=False, constrained_layout=True)
    axes = axes[0]
    cats = [c for c in CATEGORY_ORDER if c in set(agg["category"])]
    for ax, tgt in zip(axes, targets):
        piv = (agg[agg["target"] == tgt]
               .pivot(index="category", columns="device", values="abs_r")
               .reindex(cats))
        piv.index = [CATEGORY_LABELS[c] for c in piv.index]
        data = piv.values.astype(float)
        im = ax.imshow(np.ma.masked_invalid(data), cmap="magma_r", vmin=0,
                       vmax=float(np.nanmax(agg["abs_r"])), aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns, fontsize=10)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(piv.index, fontsize=9)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                            color="white" if v > 0.5 * np.nanmax(data) else "black")
                else:
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=8,
                            color="#999999")
        ax.set_title(f"median |r| with {tgt}", fontsize=10, fontweight="bold")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="median |Pearson r| across sensors in category")
    fig.suptitle("Sensor categories vs fidelity. Each cell is the median over the "
                 "sensors in that category;\nthe mixing chamber is kept separate "
                 "from the cold stages — they are different physical quantities.",
                 fontsize=9)
    save(fig, "plot5E_aggregate_heatmap")


@figure("plot7a", description="H_native distribution per device (violin), class shares to CSV")
def plot7a_violin(df) -> None:
    devs = devices_in(df)
    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    data = [df.loc[df[DEV].astype(str) == d, TGT].dropna().values for d in devs]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for body, d in zip(parts["bodies"], devs):
        body.set_facecolor(COLOR[d])
        body.set_alpha(0.65)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.0)
    ax.axhline(H_GOOD_MAX, color="green", ls=":", lw=1, alpha=0.7)
    ax.axhline(H_POOR_MIN, color="red", ls=":", lw=1, alpha=0.7)
    ax.set_xticks(range(1, len(devs) + 1))
    ax.set_xticklabels([LABEL[d] for d in devs], fontsize=9)
    ax.set_ylabel(r"$H_{\mathrm{native}}$", fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "plot7a_violin_Hnat_single_col")

    # The class shares used to live in annotation boxes on top of the violins,
    # which crowded a single-column figure. They are a table instead.
    rows = []
    for d in devs:
        s = df.loc[df[DEV].astype(str) == d, TGT].dropna()
        # hellinger_class returns the integer label 0/1/2, so compare on the
        # index into CLASS_NAMES rather than on the name.
        cls = C.hellinger_class(s)
        rows.append(dict(device=LABEL[d], n=len(s), median=float(s.median()),
                         **{f"{name}_pct": round(100 * float((cls == i).mean()), 2)
                            for i, name in enumerate(C.CLASS_NAMES)}))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "plot7a_class_shares.csv", index=False)
    print("    class shares ->  plot7a_class_shares.csv")
    print(out.to_string(index=False).replace("\n", "\n      ").rjust(6))


@figure("plot7b", description="daily mean H_native against mixing-chamber temperature")
def plot7b_daily(df) -> None:
    tmix = "sc_cryo_mxc_temp_mean"
    if tmix not in df.columns:
        print(f"    {tmix} absent - skipped")
        return
    sub = df.loc[df[DEV].astype(str) == SC, [TIME, TGT, tmix]].dropna(subset=[TIME, TGT])
    if sub.empty:
        print("    no Q-Exa rows - skipped")
        return
    sub = sub.assign(date=pd.to_datetime(sub[TIME]).dt.floor("D"))
    daily = (sub.groupby("date")
             .agg(h_mean=(TGT, "mean"),
                  h_p25=(TGT, lambda s: s.quantile(0.25)),
                  h_p75=(TGT, lambda s: s.quantile(0.75)),
                  h_n=(TGT, "count"),
                  tmix=(tmix, "mean"))
             .reset_index().sort_values("date"))
    daily = daily[daily["h_n"] >= 20]
    if daily.empty:
        print("    no day reaches 20 circuits - skipped")
        return
    fig, axl = plt.subplots(figsize=(13, 5))
    axl.fill_between(daily["date"], daily["h_p25"], daily["h_p75"],
                     color=COLOR[SC], alpha=0.22, label="IQR (25–75th pct)")
    axl.plot(daily["date"], daily["h_mean"], color=COLOR[SC], lw=1.8, marker="o",
             markersize=3.5, label=r"daily mean $H_{\mathrm{native}}$")
    axl.set_xlabel("Date", fontsize=11)
    axl.set_ylabel(r"$H_{\mathrm{native}}$  [dimensionless, 0–1]",
                   color=COLOR[SC], fontsize=11)
    axl.tick_params(axis="y", labelcolor=COLOR[SC])
    axl.set_ylim(0, 1)
    axl.grid(alpha=0.3)
    axr = axl.twinx()
    axr.plot(daily["date"], daily["tmix"], color="#B22222", lw=1.3, alpha=0.85,
             label="mixing-chamber temperature")
    axr.set_ylabel("Mixing-chamber temperature  [as released]", color="#B22222",
                   fontsize=11)
    axr.tick_params(axis="y", labelcolor="#B22222")
    axr.grid(False)
    r = daily[["h_mean", "tmix"]].corr().iat[0, 1]
    axl.set_title(f"{LABEL[SC]} — daily fidelity and mixing-chamber temperature  "
                  f"(n={len(daily)} days, r={r:.3f})", fontweight="bold")
    h1, l1 = axl.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    axl.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    save(fig, "plot7b_daily_Hnat_Tmixing")
    daily.to_csv(OUT / "plot7b_daily.csv", index=False)


@figure("plot8", description="mean H_native over 2-qubit gate count x depth, per device")
def plot8_heatmap(df) -> None:
    xc, yc = "native_n_2q_gates", "native_depth"
    devs = devices_in(df)
    p = df[[xc, yc, TGT, DEV]].dropna()
    fig, axes = plt.subplots(1, len(devs), figsize=(8.5 * len(devs), 7), squeeze=False)
    axes = axes[0]
    for ax, d in zip(axes, devs):
        sub = p[p[DEV].astype(str) == d].copy()
        xe = np.array(BIN_EDGES[d]["2q_gates"], dtype=float)
        ye = np.array(BIN_EDGES[d]["depth"], dtype=float)
        sub["tb"] = np.clip(np.digitize(sub[xc], xe[1:-1], right=False), 0, len(xe) - 2)
        sub["db"] = np.clip(np.digitize(sub[yc], ye[1:-1], right=False), 0, len(ye) - 2)
        idx = pd.Index(range(len(ye) - 1), name="db")
        col = pd.Index(range(len(xe) - 1), name="tb")
        counts = (sub.groupby(["db", "tb"]).size().unstack(fill_value=0)
                  .reindex(index=idx, columns=col, fill_value=0))
        piv = (sub.groupby(["db", "tb"])[TGT].mean().unstack()
               .reindex(index=idx, columns=col))
        # A cell with under 20 circuits is noise, not a measurement.
        piv = piv.where(counts >= 20)
        im = ax.imshow(np.ma.masked_invalid(piv.values.astype(float)), cmap="viridis",
                       vmin=0, vmax=1, origin="lower", aspect="auto")
        xl = [int_bin_label(xe[i], xe[i + 1], i == len(xe) - 2) for i in range(len(xe) - 1)]
        yl = [int_bin_label(ye[i], ye[i + 1], i == len(ye) - 2) for i in range(len(ye) - 1)]
        ax.set_xticks(range(len(xl)))
        ax.set_xticklabels(xl, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(yl)))
        ax.set_yticklabels(yl, fontsize=9)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                n = counts.iat[i, j]
                v = piv.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}\nn={n:,}", ha="center", va="center",
                            fontsize=6.5, color="white" if v > 0.5 else "black")
        ax.set_xlabel("Native 2-qubit gate count", fontsize=9)
        ax.set_ylabel("Native depth", fontsize=9)
        ax.set_title(f"{LABEL[d]}  —  {TECH[d]}", fontweight="bold", color=COLOR[d])
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="Mean $H_{native}$  [dimensionless, 0–1]")
    fig.suptitle("Blank cells hold fewer than 20 circuits.", fontsize=9, y=1.01)
    fig.tight_layout()
    save(fig, "plot8_heatmap_Hnat")


@figure("plot9", description="top Spearman correlations with H_native, per device")
def plot9_correlations(df, top: int = 10) -> None:
    base = [c for c in C.CIRCUIT_FEATURES if c in df.columns]
    per_dev = {}
    for d in devices_in(df):
        t = df[df[DEV].astype(str) == d]
        feats = list(base)
        # A sensor or calibration channel only enters a device's panel if it is
        # actually filled for that device; otherwise the correlation is computed
        # on a handful of rows.
        feats += [c for c in sensor_columns(df) + calibration_columns(df)
                  if t[c].notna().mean() > 0.50]
        feats = [f for f in dict.fromkeys(feats) if f in t.columns]
        if not feats:
            continue
        cdf = t[feats + [TGT]].apply(pd.to_numeric, errors="coerce")
        s = cdf.corr(method="spearman").loc[feats, TGT].dropna()
        if not s.empty:
            per_dev[d] = s
    if not per_dev:
        print("    no correlations available - skipped")
        return
    fig, axes = plt.subplots(1, len(per_dev), figsize=(6.4 * len(per_dev), 0.34 * top + 1.8),
                             squeeze=False)
    axes = axes[0]
    for ax, (d, s) in zip(axes, per_dev.items()):
        top_s = s.reindex(s.abs().sort_values(ascending=False).index)[:top].iloc[::-1]
        ax.barh(range(len(top_s)), top_s.values,
                color=["#9b2226" if v < 0 else "#3d5a80" for v in top_s.values])
        ax.set_yticks(range(len(top_s)))
        ax.set_yticklabels(top_s.index, fontsize=8)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(f"{LABEL[d]}\nSpearman correlation with {TGT}", fontsize=9)
        ax.set_xlabel("correlation")
        ax.spines[["top", "right"]].set_visible(False)
        s.sort_values().to_csv(OUT / f"plot9_correlations_{LABEL[d].lower()}.csv")
    fig.tight_layout()
    save(fig, "plot9_corr_sidebyside")


# ---------------------------------------------------------------------------
# Figures that need a trained model
# ---------------------------------------------------------------------------

def _tier1_model(df, *, sample_shap: int = 50_000):
    """Fit the paper's Tier 1 model and explain it on a holdout subsample.

    The same contract as experiments/train_tiers.py: 21 circuit features plus
    device identity, 80/20 GroupShuffleSplit grouped by the job id, seed 42. The
    SHAP values are computed on a subsample of the holdout because
    TreeExplainer on 300k rows x 22 features buys no extra precision in a
    ranking and costs minutes.
    """
    feats = C.tier1_feature_columns()
    d, _ = C.add_device_encoding(df)
    d = C.prepare_frame(d, feats)
    tr, te = C.group_holdout_split(d)
    train, test = d.iloc[tr], d.iloc[te]
    model = C.make_lgbm_regressor()
    model.fit(train[feats], train[C.TARGET_COL].astype("float32"))
    r2, mae, rmse = C.regression_scores(
        test[C.TARGET_COL].astype("float32"), model.predict(test[feats]))
    print(f"    Tier 1 fitted: R2(holdout)={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} "
          f"(train={len(train):,} holdout={len(test):,})")
    n = min(sample_shap, len(test))
    sub = test.sample(n, random_state=C.SEED) if n < len(test) else test
    sv = C.shap_tree_explainer(model).shap_values(sub[feats])
    return model, feats, sub, sv, dict(r2=r2, mae=mae, rmse=rmse,
                                       n_train=len(train), n_holdout=len(test))


def _shap_bar(ax, names, values, colours=None, *, top=20):
    order = np.argsort(values)[-top:]
    ax.barh(range(len(order)), np.asarray(values)[order],
            color=[colours[i] for i in order] if colours else "#3d5a80")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(np.asarray(names)[order], fontsize=8)
    ax.set_xlabel("mean |SHAP|  (Hellinger units)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.01)


@figure("tier1_shap", needs_model=True,
        description="Tier 1 SHAP bar, beeswarm, compact top-10 and gain-vs-SHAP scatter")
def tier1_shap_figures(df) -> None:
    model, feats, sub, sv, score = _tier1_model(df)
    mean_abs = np.abs(sv).mean(axis=0)
    tbl = (pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs,
                         "gain": model.booster_.feature_importance("gain")})
           .sort_values("mean_abs_shap", ascending=False))
    tbl.to_csv(OUT / "tier1_global_shap_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 0.34 * len(feats) + 1.3))
    _shap_bar(ax, feats, mean_abs, top=len(feats))
    ax.set_title(f"Tier 1 — features by mean |SHAP|  "
                 f"(R² holdout = {score['r2']:.4f})", fontsize=10)
    fig.tight_layout()
    save(fig, "tier1_global_shap_bar")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    _shap_bar(ax, feats, mean_abs, top=10)
    ax.set_title("Tier 1 — top 10 features", fontsize=10)
    fig.tight_layout()
    save(fig, "tier1_shap_compact_top10")

    # Gain is what the trees split on; mean |SHAP| is how much each feature moves
    # the prediction. They disagree, which is the point of showing both.
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    g = tbl["gain"] / tbl["gain"].sum()
    s = tbl["mean_abs_shap"] / tbl["mean_abs_shap"].sum()
    ax.scatter(g, s, s=42, color="#3d5a80", alpha=0.85, edgecolor="white")
    lim = max(g.max(), s.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="equal share")
    for _, r in tbl.head(8).iterrows():
        ax.annotate(r["feature"],
                    (r["gain"] / tbl["gain"].sum(),
                     r["mean_abs_shap"] / tbl["mean_abs_shap"].sum()),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    rho = tbl[["gain", "mean_abs_shap"]].corr(method="spearman").iat[0, 1]
    ax.set_xlabel("share of total split gain")
    ax.set_ylabel("share of total mean |SHAP|")
    ax.set_title(f"Tier 1 — gain vs SHAP ranking (Spearman ρ = {rho:.3f})",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "tier1_gain_vs_shap_scatter")

    # Beeswarm, if shap is importable as a plotting library; the bar above is the
    # one the paper leans on, so a missing beeswarm is not fatal.
    try:
        import shap as shaplib
        fig = plt.figure(figsize=(7.6, 0.3 * len(feats) + 1.6))
        shaplib.summary_plot(sv, sub[feats], show=False, max_display=len(feats))
        save(plt.gcf(), "tier1_global_shap_beeswarm")
    except Exception as e:                                    # pragma: no cover
        print(f"    beeswarm skipped ({type(e).__name__}: {e})")

    # Per-class heatmap: mean |SHAP| within each fidelity class, which says
    # whether a feature matters everywhere or only for the circuits that failed.
    cls = C.hellinger_class(sub[C.TARGET_COL])
    rows = {name: np.abs(sv[cls == i]).mean(axis=0)
            for i, name in enumerate(C.CLASS_NAMES) if (cls == i).any()}
    if rows:
        heat = pd.DataFrame(rows, index=feats).T
        keep = heat.max(axis=0).sort_values(ascending=False).index[:15]
        heat = heat[keep]
        fig, ax = plt.subplots(figsize=(0.55 * len(keep) + 2.6, 2.6))
        im = ax.imshow(heat.values, cmap="magma_r", aspect="auto")
        ax.set_xticks(range(len(keep)))
        ax.set_xticklabels(keep, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(heat)))
        ax.set_yticklabels(heat.index, fontsize=9)
        ax.set_title("Tier 1 — mean |SHAP| within each fidelity class", fontsize=10)
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="mean |SHAP|")
        fig.tight_layout()
        save(fig, "tier1_shap_per_class_heatmap")
        heat.to_csv(OUT / "tier1_shap_per_class.csv")

    fig, ax = plt.subplots(figsize=(7.4, 0.34 * len(feats) + 1.3))
    order = np.argsort(tbl["gain"].values)
    ax.barh(range(len(order)), tbl["gain"].values[order], color="#6c757d")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(tbl["feature"].values[order], fontsize=8)
    ax.set_xlabel("LightGBM split gain")
    ax.set_title("Tier 1 — features by split gain", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "tier1_global_importance")


@figure("tier2_shap", needs_model=True,
        description="per-device Tier 2 SHAP bars (sensor, and calibration on its own)")
def tier2_shap_figures(df) -> None:
    # The same order experiments/train_tiers.py uses, and the order matters:
    # restrict to the device, then to the analysis pool (the months where the
    # sensors are actually populated -- training on every month would fit a model
    # on mostly-empty telemetry), then drop rows without a target, then split,
    # and only then apply the coverage filter -- to the TRAINING frame, so the
    # holdout cannot influence which features exist.
    sensors, calib = C.resolve_device_columns(list(df.columns))
    for d in devices_in(df):
        t = df[df[DEV].astype(str) == d].reset_index(drop=True)
        t = C.apply_pool(t, d, C.DEFAULT_POOL)
        dev_sensor = [c for c in sensors.get(d, []) if c in t.columns]
        dev_calib = [c for c in calib.get(d, []) if c in t.columns]
        feats = [c for c in C.CIRCUIT_FEATURES if c in t.columns] + dev_sensor + dev_calib
        sub = C.prepare_frame(t, feats)
        if len(sub) < 1000:
            print(f"    {LABEL[d]}: only {len(sub):,} rows in pool "
                  f"{C.DEFAULT_POOL} - skipped")
            continue
        tr, te = C.group_holdout_split(sub)
        train, test = sub.iloc[tr].reset_index(drop=True), sub.iloc[te].reset_index(drop=True)
        feats, _ = C.coverage_filter(train, feats)
        model = C.make_lgbm_regressor()
        model.fit(train[feats], train[C.TARGET_COL].astype("float32"))
        r2, _, _ = C.regression_scores(test[C.TARGET_COL].astype("float32"),
                                       model.predict(test[feats]))
        n = min(30_000, len(test))
        s = test.sample(n, random_state=C.SEED) if n < len(test) else test
        sv = C.shap_tree_explainer(model).shap_values(s[feats])
        mean_abs = np.abs(sv).mean(axis=0)

        def kind(c):
            return ("calibration" if c in dev_calib
                    else "sensor" if c in dev_sensor else "circuit")

        kinds = [kind(c) for c in feats]
        tier = "2b" if dev_calib else "2a"
        kept_s = sum(1 for c in feats if c in set(dev_sensor))
        kept_c = sum(1 for c in feats if c in set(dev_calib))
        pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs,
                      "feature_type": kinds}).sort_values(
            "mean_abs_shap", ascending=False).to_csv(
            OUT / f"tier2_{d}_shap_importance.csv", index=False)
        print(f"    {LABEL[d]} Tier {tier}: pool {C.DEFAULT_POOL}, "
              f"train={len(train):,} holdout={len(test):,}, {len(feats)} features "
              f"(circuit={len(feats) - kept_s - kept_c}, sensor={kept_s}, "
              f"calib={kept_c}), R2={r2:.4f}")

        colours = [CATEGORY_COLOUR[k] for k in kinds]
        longest = max(len(c) for c in feats)
        fig, ax = plt.subplots(figsize=(6.2 + 0.085 * longest, 0.34 * 20 + 1.35))
        _shap_bar(ax, feats, mean_abs, colours, top=20)
        ax.set_title(f"{LABEL[d]}, Tier {tier} — top features by mean |SHAP| "
                     f"(R² = {r2:.4f})", fontsize=10)
        seen = [k for k in CATEGORY_COLOUR if k in set(kinds)]
        if len(seen) > 1:
            ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLOUR[k])
                               for k in seen], labels=seen, fontsize=8,
                      loc="lower right", frameon=False)
        fig.tight_layout()
        save(fig, f"tier2_{d}_shap_bar")

        # Calibration on its own: 100 calibration features against 20 circuit
        # ones would swamp the combined bar.
        if dev_calib:
            idx = [i for i, k in enumerate(kinds) if k == "calibration"]
            names = [feats[i] for i in idx]
            fig, ax = plt.subplots(
                figsize=(6.2 + 0.085 * max(len(n) for n in names), 0.34 * 20 + 1.35))
            _shap_bar(ax, names, mean_abs[idx], top=20)
            ax.set_title(f"{LABEL[d]}, Tier {tier} — top calibration features "
                         f"by mean |SHAP|", fontsize=10)
            fig.tight_layout()
            save(fig, f"tier2_{d}_calibration_shap_bar")


@figure("models", needs_model=True,
        description="model comparison and the per-class confusion matrices")
def model_comparison(df) -> None:
    from sklearn.metrics import confusion_matrix

    feats = C.tier1_feature_columns()
    d, _ = C.add_device_encoding(df)
    d = C.prepare_frame(d, feats)
    tr, te = C.group_holdout_split(d)
    train, test = d.iloc[tr], d.iloc[te]
    ytr = train[C.TARGET_COL].astype("float32")
    yte = test[C.TARGET_COL].astype("float32")

    models = {"LightGBM": C.make_lgbm_regressor(),
              "Random forest": C.make_rf_regressor()}
    rows, preds = [], {}
    for name, m in models.items():
        m.fit(train[feats], ytr)
        p = m.predict(test[feats])
        r2, mae, rmse = C.regression_scores(yte, p)
        rows.append(dict(model=name, r2=r2, mae=mae, rmse=rmse))
        preds[name] = p
        print(f"    {name}: R2={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "multi_model_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, metric, better in zip(axes, ("r2", "mae", "rmse"),
                                  ("higher is better", "lower is better",
                                   "lower is better")):
        ax.bar(res["model"], res[metric], color=["#3d5a80", "#ee9b00"], width=0.55)
        for i, v in enumerate(res[metric]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{metric.upper()}  ({better})", fontsize=10)
        ax.margins(y=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle("Tier 1 on the holdout, same split and features", fontweight="bold")
    fig.tight_layout()
    save(fig, "multi_model_comparison")

    # The regression is turned into the three fidelity classes so the errors can
    # be read as confusions rather than as a single number.
    truth = C.hellinger_class(yte)
    fig, axes = plt.subplots(1, len(preds), figsize=(4.6 * len(preds), 4.2),
                             squeeze=False)
    for ax, (name, p) in zip(axes[0], preds.items()):
        cm = confusion_matrix(truth, C.hellinger_class(pd.Series(p)),
                              labels=[0, 1, 2])
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels(C.CLASS_NAMES)
        ax.set_yticks(range(3))
        ax.set_yticklabels(C.CLASS_NAMES)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{norm[i, j]:.2f}\n{cm[i, j]:,}", ha="center",
                        va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
        acc = (cm.diagonal().sum() / cm.sum())
        ax.set_title(f"{name}  (accuracy {acc:.4f})", fontsize=10)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="row-normalised share")
    fig.suptitle("Fidelity classes derived from the regression output",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "supplementary_confusion_matrices")


@figure("forest", description="the nine Delta-R2 configurations (needs the results JSONs)")
def delta_r2_forest(df) -> None:
    """Read the experiment results and draw the Delta-R2 forest."""
    script = REPO / "experiments" / "plot_delta_r2_forest.py"
    if not script.exists():
        print("    experiments/plot_delta_r2_forest.py absent - skipped")
        return
    import subprocess
    r = subprocess.run([sys.executable, str(script),
                        "--out", str(OUT / "delta_r2_forest")],
                       capture_output=True, text=True)
    print("   ", (r.stdout or r.stderr).strip().replace("\n", "\n    ")[:600])


# ---------------------------------------------------------------------------

def main() -> int:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="draw only these figures (see --list)")
    ap.add_argument("--fast", action="store_true",
                    help="skip the figures that train a model")
    ap.add_argument("--list", action="store_true", help="list the figure names and exit")
    ap.add_argument("--outdir", type=Path, default=OUT)
    ap.add_argument("--dataset", type=Path, default=None,
                    help="dataset to draw from (default: the resolved QSTD parquet)")
    ap.add_argument("--limit", type=int, default=None,
                    help="read only the first N rows; for a smoke test, never for the paper")
    a = ap.parse_args()

    if a.list:
        print("\n  figure names:\n")
        for k, v in REGISTRY.items():
            flag = "  [trains a model]" if v["needs_model"] else ""
            print(f"    {k:14s} {v['description']}{flag}")
        print()
        return 0

    OUT = a.outdir
    names = list(REGISTRY) if not a.only else a.only
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        ap.error(f"unknown figure(s) {unknown}; --list shows the names")
    if a.fast:
        names = [n for n in names if not REGISTRY[n]["needs_model"]]

    dataset = a.dataset or C.DEFAULT_DATASET
    print(f"\n  dataset: {dataset}")
    print(f"  schema:  {'published' if C.PUBLIC_SCHEMA else 'internal'} "
          f"(device column {DEV!r}, time column {TIME!r})")
    C.set_all_seeds()

    # One read, every column these figures touch.
    cols = C.dataset_columns(dataset)
    want = [c for c in cols if c.startswith(("sc_", "ion_"))]
    want += [c for c in (DEV, TIME, TGT, "hellinger_logical", C.GROUP_COL,
                         "shots", "status", "batch_size") if c in cols]
    want += [c for c in C.CIRCUIT_FEATURES if c in cols]
    df = C.load_dataset(sorted(set(want)), dataset=dataset, limit=a.limit)
    C.warn_if_alignment_stale(list(df.columns))
    df = df[df[DEV].astype(str).isin(C.PAPER_DEVICES)]
    print(f"  rows: {len(df):,}   devices: "
          f"{', '.join(f'{LABEL[d]} {int((df[DEV] == d).sum()):,}' for d in devices_in(df))}")
    OUT.mkdir(parents=True, exist_ok=True)

    failed = []
    for n in names:
        print(f"\n  [{n}] {REGISTRY[n]['description']}")
        try:
            REGISTRY[n]["fn"](df)
        except Exception as e:
            failed.append((n, f"{type(e).__name__}: {e}"))
            print(f"    FAILED  {type(e).__name__}: {e}")
    print(f"\n  {len(names) - len(failed)}/{len(names)} figure groups drawn into {OUT}")
    for n, e in failed:
        print(f"    FAILED {n}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
