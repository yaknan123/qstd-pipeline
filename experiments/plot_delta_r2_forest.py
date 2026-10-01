#!/usr/bin/env python3
"""Forest plot of the nine ΔR² configurations (Table IX).

A nine-row table of numbers near zero does not tell a reader what the result is.
The result is: every effect is tiny, some exclude zero, and which ones do tracks
how restrictive the analysis set is rather than how large the effect is. A forest
plot says that at a glance -- points with ±1 sd whiskers against a line at zero,
on an axis scaled so the reader can see how little of R² this is.

    python3 experiments/plot_delta_r2_forest.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
POOL_ORDER = ["A", "B", "AVAIL"]
POOL_NOTE = {"A": "every month with sensors",
             "B": "A, minus the two months calibration was barely published",
             "AVAIL": "only rows that carry the device's own data"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path,
                    default=REPO / "experiments/results_rebuild_20260928/task9_split_summary.csv")
    ap.add_argument("--out", type=Path,
                    default=REPO / "experiments/results_rebuild_20260928/publication/analysis/delta_r2_forest")
    a = ap.parse_args()

    t = pd.read_csv(a.summary)
    t = t[t.tier != "Tier 3"].copy()
    t["pool"] = pd.Categorical(t["pool"], POOL_ORDER, ordered=True)
    t = t.sort_values(["pool", "device", "tier"], ascending=[False, True, True])

    labels, means, sds, res = [], [], [], []
    for r in t.itertuples():
        feat = "sensor" if r.tier == "Tier 2a" else "sensor + calibration"
        labels.append(f"{r.pool}   {r.device}, +{feat}")
        means.append(r.delta_mean); sds.append(r.delta_sd)
        res.append(r.resolved == "yes")

    fig, ax = plt.subplots(figsize=(9.5, 0.46 * len(labels) + 2.0))
    y = range(len(labels))
    for i, (m, s, ok) in enumerate(zip(means, sds, res)):
        c = "#0b525b" if ok else "#9aa3a8"
        ax.errorbar(m, i, xerr=s, fmt="o", color=c, ecolor=c,
                    capsize=3.5, markersize=6, lw=1.6)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("ΔR² against the circuit-only baseline  (mean ± 1 sd over 5 group splits)")
    ax.set_title("Telemetry contribution: every effect is under +0.002 on an R² of about 0.90",
                 fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.06)

    lo, hi = ax.get_xlim()
    ax.set_xlim(min(lo, -0.0015), max(hi, 0.0042))
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#0b525b"),
                       plt.Line2D([], [], marker="o", ls="", color="#9aa3a8")],
              labels=["mean ± 1 sd excludes zero", "does not"],
              fontsize=8.5, loc="lower right", frameon=False)

    # Scale bar: how much of R² this whole axis is.
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax.annotate(f"the whole axis spans {span:.3f} of R²; the baseline is ≈ 0.90",
                xy=(0.5, -0.16), xycoords="axes fraction", ha="center",
                fontsize=8.5, color="#444")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(a.out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {a.out}.png / .pdf  ({len(labels)} configurations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
