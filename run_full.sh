#!/usr/bin/env bash
#
# run_full.sh — one-shot driver: bench → report → commit → push.
#
# Idempotent. Safe to run after each new model lands in
# /tmp/qwen35-models/. Picks up whatever models the bench script
# can resolve at the moment of invocation.
#
# Usage:
#   release/scripts/... wait, wrong repo. Just:
#   ./run_full.sh                       # local-only (no push)
#   PUSH=1 ./run_full.sh                # commit + push to origin
#
# Output:
#   results_v3.json — raw responses (one row per model × file)
#   REPORT2.md      — markdown report with mermaid charts
#   run.log         — captured bench stdout for the run

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PUSH="${PUSH:-0}"

echo "[1/3] Running bench (this is the slow step — 5+ min per model)…"
python3 bench_v3.py 2>&1 | tee run.log

# bench_v3.py writes REPORT2.md itself, but we keep the run.log so
# someone reading the repo can sanity-check the timing numbers came
# from a real run, not a stale write.

if [[ "$PUSH" == "1" ]]; then
    echo "[2/3] Committing…"
    git add results_v3.json REPORT2.md run.log
    # Auto-generate a commit message from the model list. Pull model
    # names out of REPORT2.md's aggregate-metrics table — first column
    # is the model name in backticks.
    models=$(awk -F'|' '/^\| `/ {gsub(/` ?/, "", $2); print $2}' REPORT2.md | tr '\n' ' ')
    git commit -m "Bench v3 run: ${models}"
    echo "[3/3] Pushing…"
    git push
else
    echo "[2/3] PUSH=0 — skipping commit. Inspect REPORT2.md, then:"
    echo "       PUSH=1 ./run_full.sh   # to publish"
fi

echo "DONE"
