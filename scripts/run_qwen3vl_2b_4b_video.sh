#!/usr/bin/env bash

# Run Qwen3-VL 2B/4B (and quantized variants) on TrafficQA, MotionBench, and MVBench at 1fps.
# Launches two models at a time:
#   - First round:  Qwen3-VL-2B-Instruct      on GPUs 0,1
#                   Qwen3-VL-4B-Instruct      on GPUs 2,3
#   - Second round: Qwen3-VL-2B-Instruct-AWQ-W8A8 on GPUs 0,1
#                   Qwen3-VL-4B-Instruct-AWQ-W8A8 on GPUs 2,3
#
# Usage:
#   cd /home/ubuntu/VLMEvalKit
#   bash scripts/run_qwen3vl_2b_4b_video.sh
#
# Requirements:
#   - 4 GPUs visible to CUDA
#   - Qwen3-VL models registered in `vlmeval/config.py`

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATASETS=(
  "TrafficQA_test_1fps"
  "MotionBench_1fps"
  "MVBench_MP4_1fps"
)

# Model names must match keys in `supported_VLM` / `qwen3vl_series` in `vlmeval/config.py`
PAIR1_MODEL_A="Qwen3-VL-2B-Instruct"
PAIR1_MODEL_B="Qwen3-VL-4B-Instruct"

PAIR2_MODEL_A="Qwen3-VL-2B-Instruct-AWQ-W8A8"
PAIR2_MODEL_B="Qwen3-VL-4B-Instruct-AWQ-W8A8"

WORK_DIR_BASE="${ROOT_DIR}/outputs/qwen3vl_2b_4b_video"
mkdir -p "${WORK_DIR_BASE}"

run_pair_for_dataset() {
  local dataset="$1"
  local model_a="$2"
  local model_b="$3"
  local tag="$4"

  echo "============================================================"
  echo "Dataset: ${dataset} | Pair tag: ${tag}"
  echo "  Model A (GPU 0,1): ${model_a}"
  echo "  Model B (GPU 2,3): ${model_b}"
  echo "============================================================"

  local work_a="${WORK_DIR_BASE}/${dataset}/${tag}/${model_a}"
  local work_b="${WORK_DIR_BASE}/${dataset}/${tag}/${model_b}"
  mkdir -p "${work_a}" "${work_b}"

  CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --data "${dataset}" \
    --model "${model_a}" \
    --work-dir "${work_a}" \
    --verbose \
    > "${work_a}/log.txt" 2>&1 &
  pid_a=$!

  CUDA_VISIBLE_DEVICES=2,3 python run.py \
    --data "${dataset}" \
    --model "${model_b}" \
    --work-dir "${work_b}" \
    --verbose \
    > "${work_b}/log.txt" 2>&1 &
  pid_b=$!

  echo "Launched PIDs: A=${pid_a}, B=${pid_b} for dataset ${dataset}, tag ${tag}"
  wait "${pid_a}"
  wait "${pid_b}"
}

for ds in "${DATASETS[@]}"; do
  # Round 1: base 2B + base 4B
  run_pair_for_dataset "${ds}" "${PAIR1_MODEL_A}" "${PAIR1_MODEL_B}" "base"

  # Round 2: quantized 2B + quantized 4B
  run_pair_for_dataset "${ds}" "${PAIR2_MODEL_A}" "${PAIR2_MODEL_B}" "quant_awq_w8a8"
done

echo "All Qwen3-VL 2B/4B video runs completed."

