#!/usr/bin/env bash
# Train and evaluate every method on every dataset.
#
# Usage: scripts/run_all.sh [train options...]
#
# Each run saves its model to $RESULTS/models/, writes $RESULTS/<method>-<dataset>.jsonl,
# saves its predicted tags to $RESULTS/preds/ for `ner-eval rescore`, and logs training
# progress to $RESULTS/logs/<method>-<dataset>.log. Finished runs are skipped, so
# rerunning resumes an interrupted sweep; set FORCE=1 to redo them. Extra arguments go
# to every `ner-eval train` call. For example, a quick smoke test:
#
#   RESULTS=results/smoke METHODS="bert modernbert" scripts/run_all.sh \
#       -o max_train=2000 -o epochs=1 --max-eval 1000
set -uo pipefail

RESULTS=${RESULTS:-results}
METHODS=${METHODS:-"bert roberta modernbert"}
DATASETS=${DATASETS:-"cleanconll"}

mkdir -p "$RESULTS/logs"
failed=()
for m in $METHODS; do
  for ds in $DATASETS; do
    out="$RESULTS/$m-$ds.jsonl"
    log="$RESULTS/logs/$m-$ds.log"
    if [[ -s $out && -z ${FORCE:-} ]]; then
      echo "skip $m/$ds: $out exists"
      continue
    fi

    rm -f "$out"
    echo "$(date '+%F %T') start $m/$ds"
    # keep going if one run fails so a long sweep isn't lost
    if ! uv run ner-eval train "$m" "$ds" --models "$RESULTS/models" --out "$out" --preds "$RESULTS/preds" "$@" 2>"$log"; then
      echo "FAILED $m/$ds; see $log"
      failed+=("$m/$ds")
    fi
  done
done

if ((${#failed[@]})); then
  echo "failed runs: ${failed[*]}"
  exit 1
fi
