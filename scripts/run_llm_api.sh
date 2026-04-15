#!/usr/bin/env bash
# Run the LLM-as-judge on all three datasets via a remote API provider.
#
# Required env:
#   PROVIDER   one of: claude | deepseek | gemini | kimi | qwen | openai_compatible
#   MODEL      provider-specific model id (e.g. deepseek-reasoner, claude-opus-4-6)
#
# Optional env:
#   DATASETS      space-separated dataset dirs   (default: all three under data/)
#   OUT_ROOT      output root                    (default: results/llm_api)
#   N             attempts per conversation      (default: 1)
#   NUM_WORKERS   concurrent requests            (default: 32)
#
# Example:
#   PROVIDER=deepseek MODEL=deepseek-reasoner bash scripts/run_llm_api.sh
#   PROVIDER=openai_compatible MODEL=qwen3-plus \
#     OPENAI_COMPAT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
#     OPENAI_COMPAT_API_KEY=sk-xxx \
#     bash scripts/run_llm_api.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

: "${PROVIDER:?PROVIDER is required (claude|deepseek|gemini|kimi|qwen|openai_compatible)}"
: "${MODEL:?MODEL is required}"

# Load .env if present so the user doesn't have to export keys in every shell.
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUT_ROOT="${OUT_ROOT:-results/llm_api}"
N="${N:-1}"
NUM_WORKERS="${NUM_WORKERS:-32}"

if [ "${DATASETS+x}" ]; then
  # shellcheck disable=SC2206
  DATASETS_ARR=($DATASETS)
else
  DATASETS_ARR=(data/airlines data/healthcare data/insurance)
fi

for DATA_DIR in "${DATASETS_ARR[@]}"; do
  TAG="$(basename "$DATA_DIR")"
  RUN_OUT="$OUT_ROOT/$TAG"
  echo "========================================"
  echo "Dataset : $TAG"
  echo "Provider: $PROVIDER"
  echo "Model   : $MODEL"
  echo "Output  : $RUN_OUT"
  echo "========================================"

  python3 -m convjudge.inference.llm_api \
    --provider "$PROVIDER" \
    --model    "$MODEL" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RUN_OUT" \
    --n "$N" \
    --num-workers "$NUM_WORKERS"

  python3 -m convjudge.evaluation.eval_llm \
    --usage-root "$RUN_OUT" \
    --data-root  "$DATA_DIR" \
    --output-dir "$RUN_OUT/usage_accuracy"
done

echo ""
echo "Done. Output: $OUT_ROOT"
