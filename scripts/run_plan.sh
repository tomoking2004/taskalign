#!/usr/bin/env bash
# The measurements context.md §8 plans next, as one job that records itself.
#
#   bash scripts/run_plan.sh                 every stage, in the plan's order
#   bash scripts/run_plan.sh check           only make sure the machine is ready
#   bash scripts/run_plan.sh map strength    only the stages named
#
#   check      the package, the dataset and the reported weights are where the code looks
#   map        where the answer depends, by layer and position; chooses the site to align  (§8.5, step 1)
#   strength   three runs at other alignment weights, and their dependence                  (§8.4)
#   position   six runs aligned at the chosen site, and the outcome they fall in            (§8.5, step 2)
#
# Each stage ends by rebuilding the progress report from what analyses/ now holds, and
# committing and pushing its results — so a stage that finished stays finished whatever
# happens to the next, and pulling on another machine brings the results and the deck
# together. The stages take roughly an hour, nine hours and eighteen hours on the machine
# that trained the reported runs; only the first is an estimate.
#
# It is safe to start again. A run that finished is not retrained; the trials and the
# measurements run again from the weights, and write what they wrote before; and a stage
# whose results and deck come out unchanged records nothing.
#
# Run it detached (tmux, or /ops:exec's detached form): the last stage outlives any
# terminal. The log is logs/run-plan-<start>.log.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1  # so the log reads in the order things happened
PY=${PYTHON:-python}
DECK="presentations/20261127_進捗報告.pptx"
MAP="analyses/probe-dependence-map.json"

STAGES=(${@:-check map strength position})
for name in "${STAGES[@]}"; do
    case $name in
        check|map|strength|position) ;;
        *) echo "unknown stage: $name (check, map, strength, position)"; exit 2 ;;
    esac
done

# A pull that stopped on a conflict leaves the branch mid-rebase, and a commit made on
# top of that would be made in the wrong place. Someone has to finish or abort it first.
if [[ -d .git/rebase-merge || -d .git/rebase-apply ]]; then
    echo "a rebase is in progress here; finish it (or \`git rebase --abort\`) and start again"
    exit 1
fi

mkdir -p logs
LOG="logs/run-plan-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

say() { printf '\n━━ %s  %s\n' "$(date '+%m-%d %H:%M:%S')" "$*"; }

# Rebuild the deck, then commit and push the paths given. A deck that fails to build
# must not cost the results, so it is reported and left as it was.
record() {
    local message=$1; shift
    "$PY" scripts/update_progress.py && git add -- "$DECK" \
        || echo "⚠️  the deck was not rebuilt; recording the results without it"
    git add -- "$@"
    if git diff --cached --quiet; then
        say "nothing new to record"
        return
    fi
    git commit -m "$message"
    # The other machine shares this branch: take its commits first, keeping whatever
    # this job has not staged out of the way.
    git pull --rebase --autostash
    git push
    say "recorded and pushed: ${message%%$'\n'*}"
}

# Stage the finished snapshots of one variant. A run aligned at several positions saves
# a stack of representations past what the remote accepts for one file (100MB); it stays
# in runs/ and in the working tree, and everything else about the run is recorded.
add_snapshots() {
    local dir
    # shellcheck disable=SC2231
    for dir in snapshots/*_${1}_*; do
        [[ -f $dir/metrics/eval.json && $dir != *_debug ]] || continue
        local stack=$dir/metrics/representations.pt
        if [[ -f $stack && $(stat -c %s "$stack") -gt 95000000 ]]; then
            git add -- "$dir" ":(exclude)$stack"
        else
            git add -- "$dir"
        fi
    done
}

# Prove the wiring in minutes, clear what that left behind, then train for real.
train() {
    local sweep=$1 pattern=$2 runs
    runs=$("$PY" -c "from taskalign.config import cfg; print(cfg.runs_root)")
    say "$sweep: trial"
    "$PY" scripts/sweep.py "$sweep" --debug
    rm -rf -- "$runs"/*_${pattern}_*_debug snapshots/*_${pattern}_*_debug
    say "$sweep: training"
    "$PY" scripts/sweep.py "$sweep"
}

stage_check() {
    say "check"
    "$PY" - <<'EOF'
from taskalign.analysis import require_checkpoints
from taskalign.config import cfg

for path in (cfg.gqa.train_questions, cfg.gqa.val_questions, cfg.gqa.testdev_questions,
             cfg.gqa.images, cfg.snapshots_root):
    if not path.exists():
        raise SystemExit(f"not found: {path}")
require_checkpoints()
print(f"device {cfg.device} · dataset {cfg.gqa.root} · runs {cfg.runs_root}")
print("the reported nine checkpoints are in place")
EOF
    git fetch --quiet
    echo "the remote is reachable"
}

stage_map() {
    say "map: trial over 8 prompts"
    "$PY" scripts/probe_dependence_map.py reported --prompts 8
    say "map: the nine reported runs"
    "$PY" scripts/probe_dependence_map.py reported
    say "map: the two runs aligned at layers 8 and 24"
    "$PY" scripts/probe_dependence_map.py layer \
        || echo "⚠️  the layer runs were not measured (see above); the plan goes on without them"
    record "chore: add the map of where the answer depends

context.md §8.5's first step: the relative sensitivity of §A.4 at every
block's output and every prompt position, for the nine reported runs,
and the site the second step aligns, chosen from the baseline's map by
the rule fixed beforehand." "$MAP"
}

stage_strength() {
    train strength 'lambda*'
    say "strength: dependence at each weight"
    "$PY" scripts/probe_dependence_map.py strength
    "$PY" scripts/summarize_sweep.py
    add_snapshots 'lambda*'
    record "chore: add the runs at other alignment weights

context.md §8.4: the signature-labelled arm at weights 0.03, 0.3 and
1.0, seed 42, with what each representation carries, what the answers
score and how hard they lean on the aligned site." "$MAP" analyses/summarize-sweep.json
}

stage_position() {
    train position relocated
    say "position: dependence at the new site"
    "$PY" scripts/probe_dependence_map.py position
    "$PY" scripts/summarize_relocation.py
    add_snapshots relocated
    record "chore: add the runs aligned at the site the map chose

context.md §8.5's second step: both aligned arms at the three reported
seeds, aligned where the baseline's answer depends most, with the
outcome of the step's table they fall in." "$MAP" analyses/summarize-relocation.json
}

for name in "${STAGES[@]}"; do
    "stage_$name"
done
say "done"
