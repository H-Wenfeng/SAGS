#!/bin/bash
# Evaluate all trained models

set -e

echo "========== Evaluating all models =========="

# EndoNeRF Binocular
for scene in pulling cutting; do
    if [ -d "output/endonerf/${scene}" ]; then
        echo "Evaluating: endonerf/${scene}"
        python metrics.py --model_paths "output/endonerf/${scene}"
    fi
done

# EndoNeRF Monocular
for scene in pulling_mono cutting_mono; do
    if [ -d "output/endonerf/${scene}" ]; then
        echo "Evaluating: endonerf/${scene}"
        python metrics.py --model_paths "output/endonerf/${scene}"
    fi
done

# SCARED
for sn in d1k1 d2k1 d3k1 d6k1 d7k1; do
    if [ -d "output/scared/${sn}" ]; then
        echo "Evaluating: scared/${sn}"
        python metrics.py --model_paths "output/scared/${sn}"
    fi
done

echo "========== Evaluation complete! =========="
