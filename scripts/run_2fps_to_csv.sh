#!/usr/bin/env bash

# Run evaluations on TrafficQA and MotionBench at 2fps and output results to CSV files separately.
#
# Usage:
#   cd /home/ubuntu/VLMEvalKit
#   bash scripts/run_2fps_to_csv.sh [MODEL_NAME] [GPU_IDS]
#
# Arguments:
#   MODEL_NAME: Model to evaluate (default: Qwen3-VL-4B-Instruct)
#   GPU_IDS: GPU IDs to use (default: 0,1)
#
# Examples:
#   bash scripts/run_2fps_to_csv.sh Qwen3-VL-4B-Instruct 0,1
#   bash scripts/run_2fps_to_csv.sh Qwen3-VL-2B-Instruct-AWQ-W8A8 2,3

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Configuration
MODEL_NAME="${1:-Qwen3-VL-4B-Instruct}"
GPU_IDS="${2:-0,1}"
WORK_DIR="${ROOT_DIR}/outputs/2fps_eval"

# Create output directories
mkdir -p "${WORK_DIR}"

echo "============================================================"
echo "Running 2fps Evaluations"
echo "  Model: ${MODEL_NAME}"
echo "  GPUs: ${GPU_IDS}"
echo "  Work Dir: ${WORK_DIR}"
echo "============================================================"

# Function to run evaluation and generate CSV
run_and_export() {
    local dataset="$1"
    local output_csv="${WORK_DIR}/${MODEL_NAME}_${dataset}_2fps.csv"

    echo ""
    echo "------------------------------------------------------------"
    echo "Dataset: ${dataset}"
    echo "Output CSV: ${output_csv}"
    echo "------------------------------------------------------------"

    # Run evaluation
    CUDA_VISIBLE_DEVICES=${GPU_IDS} python run.py \
        --data "${dataset}" \
        --model "${MODEL_NAME}" \
        --work-dir "${WORK_DIR}" \
        --verbose

    # Find the rating JSON file
    local rating_file=$(find "${WORK_DIR}" -name "*${MODEL_NAME}*${dataset}*rating.json" -type f 2>/dev/null | head -1)

    if [[ -z "${rating_file}" ]]; then
        echo "WARNING: Rating file not found for ${dataset}, skipping CSV export"
        return 1
    fi

    echo "Found rating file: ${rating_file}"

    # Convert JSON to CSV using Python
    python - <<EOF
import json
import csv
import sys

rating_file = "${rating_file}"
output_csv = "${output_csv}"

try:
    with open(rating_file, 'r') as f:
        data = json.load(f)

    # Determine if this is TrafficQA or MotionBench format
    if 'basic_understanding' in data:
        # TrafficQA format
        rows = []
        for category, metrics in data.items():
            if isinstance(metrics, dict) and 'acc' in metrics:
                rows.append({
                    'category': category,
                    'accuracy': round(metrics['acc'] * 100, 2),
                    'correct': metrics.get('correct', 0),
                    'total': metrics.get('total', 0),
                    'valid': metrics.get('valid', 0)
                })

        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['category', 'accuracy', 'correct', 'total', 'valid'])
            writer.writeheader()
            writer.writerows(rows)

        print(f"Exported TrafficQA results to: {output_csv}")

    else:
        # MotionBench format (question_type based)
        rows = []
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict):
                if 'question_type' in item:
                    rows.append({
                        'question_type': item.get('question_type', ''),
                        'accuracy': item.get('acc', 0),
                        'success': item.get('success', 0),
                        'overall': item.get('overall', 0)
                    })

        if not rows:
            # Handle dict format where keys are question types
            for key, value in data.items():
                if isinstance(value, dict):
                    rows.append({
                        'question_type': key,
                        'accuracy': value.get('acc', 0),
                        'success': value.get('success', 0),
                        'overall': value.get('overall', 0)
                    })

        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['question_type', 'accuracy', 'success', 'overall'])
            writer.writeheader()
            writer.writerows(rows)

        print(f"Exported MotionBench results to: {output_csv}")

except Exception as e:
    print(f"ERROR exporting to CSV: {e}")
    sys.exit(1)
EOF

    echo "CSV saved: ${output_csv}"
}

# Run evaluations for both datasets
run_and_export "TrafficQA_test_2fps"
run_and_export "MotionBench_2fps"

echo ""
echo "============================================================"
echo "All evaluations completed!"
echo "Results saved to: ${WORK_DIR}"
echo "============================================================"
ls -la "${WORK_DIR}"/*.csv 2>/dev/null || echo "No CSV files found"
