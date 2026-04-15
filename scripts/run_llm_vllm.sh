#!/usr/bin/env bash
# Run the LLM-as-judge on all three datasets via local vLLM inference.
#
# Required env:
#   MODEL     HuggingFace model id or local path (e.g. Qwen/Qwen3-8B)
#
# Optional env:
#   CUDA_VISIBLE_DEVICES   default: 0
#   TP                     tensor parallel size (default: 1)
#   BATCH_SIZE             (default: 8)
#   MAX_MODEL_LEN          (default: 38912)
#   MAX_TOKENS             (default: 38912)
#   GPU_MEMORY_UTILIZATION (default: 0.9)
#   DTYPE                  (default: bfloat16)
#   TEMPERATURE TOP_P TOP_K MIN_P    sampling (any subset)
#   N                      attempts per conversation (default: 1)
#   DATASETS               space-separated dataset dirs
#   OUT_ROOT               output root (default: results/llm_vllm)
#
# Example:
#   MODEL=Qwen/Qwen3-8B TEMPERATURE=0.6 TOP_P=0.95 N=4 bash scripts/run_llm_vllm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

: "${MODEL:?MODEL is required (e.g. Qwen/Qwen3-8B)}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TP="${TP:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-38912}"
MAX_TOKENS="${MAX_TOKENS:-38912}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DTYPE="${DTYPE:-bfloat16}"
N="${N:-1}"
OUT_ROOT="${OUT_ROOT:-results/llm_vllm}"

SAMPLING_ARGS=()
[ -n "${TEMPERATURE:-}" ] && SAMPLING_ARGS+=(--temperature "$TEMPERATURE")
[ -n "${TOP_P:-}" ]       && SAMPLING_ARGS+=(--top-p "$TOP_P")
[ -n "${TOP_K:-}" ]       && SAMPLING_ARGS+=(--top-k "$TOP_K")
[ -n "${MIN_P:-}" ]       && SAMPLING_ARGS+=(--min-p "$MIN_P")

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
  echo "Model   : $MODEL   (tp=$TP, batch=$BATCH_SIZE)"
  echo "GPU     : ${CUDA_VISIBLE_DEVICES}"
  echo "Output  : $RUN_OUT"
  echo "========================================"

  python3 -m convjudge.inference.llm_vllm \
    --model "$MODEL" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RUN_OUT" \
    --n "$N" \
    --tensor-parallel-size "$TP" \
    --batch-size "$BATCH_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --dtype "$DTYPE" \
    "${SAMPLING_ARGS[@]}" \
    --trust-remote-code

  python3 -m convjudge.evaluation.eval_llm \
    --usage-root "$RUN_OUT" \
    --data-root  "$DATA_DIR" \
    --output-dir "$RUN_OUT/usage_accuracy"
done

echo ""
echo "Done. Output: $OUT_ROOT"
