#!/usr/bin/env bash
# ==============================================================================
#  Fadfid: run every evaluation that needs no new human labels, one after another
# ==============================================================================
#  Run in Ubuntu from the Fadfid folder, with the web server STOPPED:
#    source ~/fadfid-venv/bin/activate
#    bash scripts/final_results.sh
#
#  1. Five conditions on the contrast set and personas       results/summary.csv
#  2. Keyword vs rules vs Fanar-only crisis detection         results/detectors/
#  3. Harm-to-others detection                                results/harm/
#  4. Empathy and cultural scores with a second model         results/summary_judged.csv
#
#  Later steps add the trained classifier and the held-out set automatically
#  if those files exist.
# ==============================================================================
set -euo pipefail
export FADFID_BACKEND=fanar FADFID_4BIT=0
cd "$(dirname "$0")/.."
mkdir -p results
LOG=results/final_run.log
: > "$LOG"
step() { echo; echo "=== $1  ($(date +%H:%M))" | tee -a "$LOG"; }

EXTRA=()
[ -f data/heldout_set.csv ] && EXTRA=(--extra data/heldout_set.csv)

step "1/4 Five conditions"
python eval/run_eval.py --backend fanar "${EXTRA[@]}" 2>&1 | tee -a "$LOG"

step "2/4 Crisis detectors"
SETS=(data/contrast_set.csv)
[ -f data/heldout_set.csv ] && SETS+=(data/heldout_set.csv)
[ -f data/classifier_test.csv ] && SETS+=(data/classifier_test.csv)
python eval/compare_detectors.py --sets "${SETS[@]}" --keep-contrast 2>&1 | tee -a "$LOG"

step "3/4 Harm to others"
HSETS=(data/harm_set.csv)
[ -f data/harm_heldout.csv ] && HSETS+=(data/harm_heldout.csv)
python eval/harm_eval.py --sets "${HSETS[@]}" 2>&1 | tee -a "$LOG"

step "4/4 Empathy and cultural judge (first run downloads about 15 GB)"
python eval/judge.py 2>&1 | tee -a "$LOG"

step "Done. Send results/final_run.log and the results folder to Claude."
