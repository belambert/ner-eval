#!/usr/bin/env bash
# Train and evaluate every method on Few-NERD's coarse and fine labels.
#
# Usage: scripts/run_all.sh [evaluate options...]
#
# Each run writes $RESULTS/<method>-<level>.jsonl, saves its predicted tags to
# $RESULTS/preds/ for `ner-eval rescore`, and logs training progress to
# $RESULTS/logs/<method>-<level>.log. Finished runs are skipped, so rerunning resumes
# an interrupted sweep; set FORCE=1 to redo them. Set SAVE=1 to also save each model
# to $RESULTS/models/. Extra arguments go to every `ner-eval evaluate` call. For
# example, a quick smoke test:
#
#   RESULTS=results/smoke METHODS="bert modernbert" scripts/run_all.sh \
#       -o max_train=2000 -o epochs=1 --max-eval 1000
set -uo pipefail

RESULTS=${RESULTS:-results}
METHODS=${METHODS:-"memorize bert roberta modernbert ettin"}
LEVELS=${LEVELS:-"coarse fine"}
save=()
[[ -n ${SAVE:-} ]] && save=(--save "$RESULTS/models")

mkdir -p "$RESULTS/logs"
failed=()
for m in $METHODS; do
  for lvl in $LEVELS; do
    out="$RESULTS/$m-$lvl.jsonl"
    log="$RESULTS/logs/$m-$lvl.log"
    if [[ -s $out && -z ${FORCE:-} ]]; then
      echo "skip $m/$lvl: $out exists"
      continue
    fi

    rm -f "$out"
    echo "$(date '+%F %T') start $m/$lvl"
    # keep going if one run fails so a long sweep isn't lost
    if ! uv run ner-eval evaluate "$m" "$lvl" --out "$out" --preds "$RESULTS/preds" "${save[@]}" "$@" 2>"$log"; then
      echo "FAILED $m/$lvl; see $log"
      failed+=("$m/$lvl")
    fi
  done
done

if ((${#failed[@]})); then
  echo "failed runs: ${failed[*]}"
  exit 1
fi
