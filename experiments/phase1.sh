#!/usr/bin/env bash
# Reproduce every Phase 1 measurement. Each step is resumable: finished episodes are skipped,
# and rerunning a fault step on a finished directory only recomputes its comparison.csv.
# Requires .env with OPENAI_API_KEY/OPENAI_BASE_URL (TritonAI) and ALFWorld data.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
TEAM=configs/llm_team.yaml          # LLM observation, memory and planner
VERIFY=configs/llm_team_verify.yaml # the same team plus an LLM verifier on every plan
# Evaluation workload: valid_unseen (never used for development), 3 games per task type.
GAMES="1:0 1:1 1:2 2:0 2:1 2:2 3:0 3:1 3:2 4:0 4:1 4:2 5:0 5:1 5:2 6:0 6:1 6:2"
SEEDS="42 43 44"

# 1. Clean baselines, three seeds each.
$PY -m experiments.run_clean --out results/team-clean --agents $TEAM --split valid_unseen \
    --games $GAMES --seeds $SEEDS --workers 3
$PY -m experiments.summarize results/team-clean --csv results/team-clean/summary.csv
$PY -m experiments.run_clean --out results/team-verify-clean --agents $VERIFY --split valid_unseen \
    --games $GAMES --seeds $SEEDS --workers 3
$PY -m experiments.summarize results/team-verify-clean --csv results/team-verify-clean/summary.csv

# 2. Executed full restart: rerun every seed-42 game with live model calls (a fresh cache
#    namespace), as a restart after a detected fault would. Also the live-latency check.
$PY -m experiments.run_clean --out results/team-restart --agents $TEAM --split valid_unseen \
    --games $GAMES --seeds 42 --workers 3 --cache-namespace restart-1
$PY -m experiments.summarize results/team-restart --csv results/team-restart/summary.csv

# 3. Fault injection into every successful seed-42 clean run: 4 stages x operators x
#    early/mid/late. The schedule is written to results/team-faults/schedule.jsonl first.
$PY -m experiments.run_faults --clean results/team-clean --out results/team-faults --agents $TEAM \
    --positions early mid late --workers 3

# 4. Repeats on seeds 43 and 44 (mid position) for run-to-run variability.
for SEED in 43 44; do
  $PY -m experiments.run_faults --clean results/team-clean --out results/team-faults-s$SEED \
      --agents $TEAM --positions mid --clean-seed $SEED --workers 3
done

# 5. Always verify under the identical (seed-42, mid-position) fault schedule.
$PY -m experiments.run_faults --clean results/team-verify-clean --out results/team-verify-faults \
    --agents $VERIFY --schedule results/team-faults/schedule.jsonl --positions mid --workers 3

# 6. Tables.
$PY -m analysis.report --clean results/team-clean/summary.csv \
    --faults results/team-faults/comparison.csv \
    --repeats results/team-faults-s43/comparison.csv results/team-faults-s44/comparison.csv \
    --restart results/team-restart/summary.csv \
    --verify-clean results/team-verify-clean/summary.csv \
    --verify-faults results/team-verify-faults/comparison.csv --out results/phase1-tables.md
