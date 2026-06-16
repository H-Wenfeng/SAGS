#!/bin/bash
# Train and evaluate on SCARED dataset (binocular)

set -e

DATASETS=("dataset_1" "dataset_2" "dataset_3" "dataset_6" "dataset_7")
SHORTNAMES=("d1k1" "d2k1" "d3k1" "d6k1" "d7k1")
DATA_ROOT="data/scared"

echo "========== SCARED Binocular =========="

for i in "${!DATASETS[@]}"; do
    ds="${DATASETS[$i]}"
    sn="${SHORTNAMES[$i]}"
    
    echo "--- Training: ${sn} ---"
    python train.py \
        -s "${DATA_ROOT}/${ds}/keyframe_1" \
        --port 6017 \
        --expname "scared/${sn}" \
        --configs "arguments/scared/${sn}.py"

    echo "--- Rendering: ${sn} ---"
    python render.py \
        --model_path "output/scared/${sn}" \
        --skip_train --skip_video \
        --configs "arguments/scared/${sn}.py"

    echo "--- Metrics: ${sn} ---"
    python metrics.py --model_paths "output/scared/${sn}"
done

echo "========== All SCARED experiments completed! =========="
