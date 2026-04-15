#!/usr/bin/env bash
# Score all three datasets with a classifier-head reward model,
# then compute violation-detection accuracy.
#
# Required env:
#   MODEL   HuggingFace model id or local path
#           (e.g. Skywork/Skywork-Reward-V2-Llama-3.1-8B)
#
# Optional env:
#   DEVICE       default: cuda:0
#   OUT_ROOT     default: results/rm_classifier
#   DATASETS     space-separated dataset dirs
#
# Example:
#   MODEL=Skywork/Skywork-Reward-V2-Llama-3.1-8B bash scripts/run_rm_classifier.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

: "${MODEL:?MODEL is required}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DEVICE="${DEVICE:-cuda:0}"
OUT_ROOT="${OUT_ROOT:-results/rm_classifier}"
SLUG="${MODEL//\//_}"

if [ "${DATASETS+x}" ]; then
  # shellcheck disable=SC2206
  DATASETS_ARR=($DATASETS)
else
  DATASETS_ARR=(data/airlines data/healthcare data/insurance)
fi

for DATA_DIR in "${DATASETS_ARR[@]}"; do
  TAG="$(basename "$DATA_DIR")"
  SCORES_DIR="$OUT_ROOT/$TAG/$SLUG"
  echo "========================================"
  echo "Dataset : $TAG"
  echo "Model   : $MODEL"
  echo "Output  : $SCORES_DIR"
  echo "========================================"

  python3 -m convjudge.inference.rm_classifier \
    --model-name "$MODEL" \
    --data-dir   "$DATA_DIR" \
    --output-dir "$SCORES_DIR" \
    --device     "$DEVICE"

  python3 -m convjudge.evaluation.eval_rm \
    --scores-dir "$SCORES_DIR" \
    --output-dir "$SCORES_DIR"
done

echo ""
echo "Done. Output: $OUT_ROOT"
