#!/usr/bin/env bash
# =============================================================================
# reproduce_all.sh — every result in the paper, from the published dataset
# =============================================================================
#
#   uv sync
#   # put qstd_v1.0.parquet in data/   (see data/README.md)
#   bash scripts/reproduce_all.sh
#
# About two hours on 8 cores, CPU only. Results go to experiments/results/
# (and experiments/results/pool_<name>/ for the non-default analysis sets).
# The last step compares what was computed against the published numbers.
#
# Options:
#   SKIP_SLOW=1   omit the neural baseline and leave-one-device-out, the two
#                 longest steps
# =============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
DATA="data/qstd_v1.0.parquet"

if [[ ! -f "$DATA" ]]; then
    echo "!! $DATA not found. See data/README.md for where to get it."
    exit 1
fi

RUN=(uv run python)
export MQSS_RESULTS_DIR="${MQSS_RESULTS_DIR:-$PWD/experiments/results}"
mkdir -p "$MQSS_RESULTS_DIR"

step() { printf '\n\033[1m=== %s\033[0m\n' "$*"; }

step "Table III — tiers per device (analysis set A)"
"${RUN[@]}" experiments/train_tiers.py | tee "$MQSS_RESULTS_DIR/train_tiers.log"

step "Table V — feature categories"
"${RUN[@]}" experiments/tier2a_sensor_only.py | tee "$MQSS_RESULTS_DIR/tier2a.log"

step "Table IV — shot distribution and the single-bucket robustness check"
"${RUN[@]}" experiments/shot_histogram.py | tee "$MQSS_RESULTS_DIR/shot_histogram.log"

if [[ "${SKIP_SLOW:-0}" != "1" ]]; then
    step "Table VI — leave-one-device-out"
    "${RUN[@]}" experiments/lodo.py | tee "$MQSS_RESULTS_DIR/lodo.log"
fi

step "Table VIII — class shift"
"${RUN[@]}" experiments/class_shift.py | tee "$MQSS_RESULTS_DIR/class_shift.log"

step "Analysis sets B and AVAIL"
for POOL in B AVAIL; do
    "${RUN[@]}" experiments/train_tiers.py --pool "$POOL" --tier 2 \
        | tee "$MQSS_RESULTS_DIR/train_tiers_$POOL.log"
    "${RUN[@]}" experiments/tier2a_sensor_only.py --pool "$POOL" \
        | tee "$MQSS_RESULTS_DIR/tier2a_$POOL.log"
    "${RUN[@]}" experiments/class_shift.py --pool "$POOL" \
        | tee "$MQSS_RESULTS_DIR/class_shift_$POOL.log"
done
"${RUN[@]}" experiments/compare_pools.py --pools A B AVAIL \
    | tee "$MQSS_RESULTS_DIR/compare_pools.log"

step "Table IX — DeltaR2 as mean +/- sd over five group splits"
"${RUN[@]}" experiments/split_repeats.py --pools A B AVAIL \
    | tee "$MQSS_RESULTS_DIR/split_repeats.log"

step "Table X — forward-in-time split"
"${RUN[@]}" experiments/forward_split.py --pools A B AVAIL \
    | tee "$MQSS_RESULTS_DIR/forward_split.log"

# Not one of the paper's tables. The alignment window was checked at 2, 5, 10 and
# 20 minutes and makes a negligible difference (at most +0.0012 R² against a
# baseline of 0.91), so only the 5-minute build is published and the finding is a
# methods statement rather than a table. The script runs only if you have built
# the other windows yourself.
step "Alignment window check (not a paper table)"
WINDOWS=(--window "5=$DATA")
for W in 02 10 20; do
    F="data/windows/window_$W.parquet"
    [[ -f "$F" ]] && WINDOWS+=(--window "$((10#$W))=$F")
done
if [[ ${#WINDOWS[@]} -eq 2 ]]; then
    echo "   only the 5-minute build is published, so there is nothing to compare"
    echo "   against — skipping. This is expected; see data/README.md."
else
    "${RUN[@]}" experiments/window_ablation.py "${WINDOWS[@]}" \
        | tee "$MQSS_RESULTS_DIR/window_ablation.log"
fi

step "Checking against the published numbers"
"${RUN[@]}" scripts/check_results.py
