#!/bin/bash
# Train and evaluate on EndoNeRF dataset (binocular + monocular)

set -e

# ==================== Binocular ====================
echo "========== EndoNeRF Binocular =========="

echo "[1/6] Training: Pulling (binocular)"
python train.py -s data/pulling --port 6017 --expname endonerf/pulling --configs arguments/endonerf/pulling.py

echo "[2/6] Training: Cutting (binocular)"
python train.py -s data/cutting --port 6017 --expname endonerf/cutting --configs arguments/endonerf/cutting.py

echo "[3/6] Rendering: Pulling (binocular)"
python render.py --model_path output/endonerf/pulling --skip_train --skip_video --configs arguments/endonerf/pulling.py

echo "[4/6] Rendering: Cutting (binocular)"
python render.py --model_path output/endonerf/cutting --skip_train --skip_video --configs arguments/endonerf/cutting.py

echo "[5/6] Metrics: Pulling (binocular)"
python metrics.py --model_paths output/endonerf/pulling

echo "[6/6] Metrics: Cutting (binocular)"
python metrics.py --model_paths output/endonerf/cutting

# ==================== Monocular ====================
echo "========== EndoNeRF Monocular =========="

echo "[1/6] Training: Pulling (monocular)"
python train.py -s data/pulling --port 6017 --expname endonerf/pulling_mono --configs arguments/endonerf/pulling_mono.py

echo "[2/6] Training: Cutting (monocular)"
python train.py -s data/cutting --port 6017 --expname endonerf/cutting_mono --configs arguments/endonerf/cutting_mono.py

echo "[3/6] Rendering: Pulling (monocular)"
python render.py --model_path output/endonerf/pulling_mono --skip_train --skip_video --configs arguments/endonerf/pulling_mono.py

echo "[4/6] Rendering: Cutting (monocular)"
python render.py --model_path output/endonerf/cutting_mono --skip_train --skip_video --configs arguments/endonerf/cutting_mono.py

echo "[5/6] Metrics: Pulling (monocular)"
python metrics.py --model_paths output/endonerf/pulling_mono

echo "[6/6] Metrics: Cutting (monocular)"
python metrics.py --model_paths output/endonerf/cutting_mono

echo "========== All EndoNeRF experiments completed! =========="
