# SAGS: Self-Adaptive Alias-Free Gaussian Splatting for Dynamic Surgical Endoscopic Reconstruction

This is the official implementation of the paper:

**SAGS: Self-Adaptive Alias-Free Gaussian Splatting for Dynamic Surgical Endoscopic Reconstruction**

Wenfeng Huang, Xiangyun Liao, Yinling Qian, Hao Liu, Yongming Yang, Wenjing Jia, Qiong Wang

[[Paper]](https://arxiv.org/abs/2510.27318)

## Overview

SAGS is a self-adaptive alias-free Gaussian splatting framework for dynamic endoscopic scene reconstruction. It introduces:

1. **Self-Adaptive Weighted Deformation Decoder (SAD)**: A multi-head attention-based mechanism with dynamically learned weights (γ₁, γ₂) and Affine transformations that adaptively balances global geometric consistency (via self-attention) and local deformation refinement (via MLP).

2. **Alias-Free Processing**: 3D smoothing filters and 2D Mip filters that suppress high-frequency artifacts in both volumetric and image-space representations, following Mip-Splatting principles.



<p align="center">
  <img src="assets/pipeline.png" width="90%">
</p>

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.8+
- PyTorch 1.13+

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SAGS.git
cd SAGS

# Create conda environment
conda create -n sags python=3.8
conda activate sags

# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
pip install -r requirements.txt

# Install submodules
pip install submodules/depth-diff-gaussian-rasterization
pip install submodules/simple-knn
```

## Data Preparation

### EndoNeRF Dataset

Download the [EndoNeRF dataset](https://github.com/med-air/EndoNeRF) and organize it as:

```
data/
├── pulling/
│   ├── images/
│   ├── depth/          # binocular depth
│   ├── monodepth/      # monocular depth (optional)
│   ├── masks/
│   └── poses_bounds.npy
└── cutting/
    ├── images/
    ├── depth/
    ├── monodepth/
    ├── masks/
    └── poses_bounds.npy
```

### SCARED Dataset

Download the [SCARED dataset](https://endovissub2019-scared.grand-challenge.org/) and organize it as:

```
data/scared/
├── dataset_1/keyframe_1/
│   └── data/
│       ├── left_finalpass/
│       ├── disparity/
│       ├── frame_data/
│       └── reprojection_data/
├── dataset_2/keyframe_1/
...
```

For monocular depth, generate depth maps using [Depth Anything](https://github.com/LiheYoung/Depth-Anything) and place them under `data/left_monodam/`.

## Training

### EndoNeRF (Binocular)

```bash
# Pulling scene
python train.py -s data/pulling --port 6017 --expname endonerf/pulling --configs arguments/endonerf/pulling.py

# Cutting scene
python train.py -s data/cutting --port 6017 --expname endonerf/cutting --configs arguments/endonerf/cutting.py
```

### EndoNeRF (Monocular)

```bash
python train.py -s data/pulling --port 6017 --expname endonerf/pulling_mono --configs arguments/endonerf/pulling_mono.py
python train.py -s data/cutting --port 6017 --expname endonerf/cutting_mono --configs arguments/endonerf/cutting_mono.py
```

### SCARED (Binocular)

```bash
python train.py -s data/scared/dataset_1/keyframe_1 --port 6017 --expname scared/d1k1 --configs arguments/scared/d1k1.py
python train.py -s data/scared/dataset_2/keyframe_1 --port 6017 --expname scared/d2k1 --configs arguments/scared/d2k1.py
python train.py -s data/scared/dataset_3/keyframe_1 --port 6017 --expname scared/d3k1 --configs arguments/scared/d3k1.py
python train.py -s data/scared/dataset_6/keyframe_1 --port 6017 --expname scared/d6k1 --configs arguments/scared/d6k1.py
python train.py -s data/scared/dataset_7/keyframe_1 --port 6017 --expname scared/d7k1 --configs arguments/scared/d7k1.py
```

## Rendering

```bash
# Render test views
python render.py --model_path output/endonerf/pulling --skip_train --skip_video --configs arguments/endonerf/pulling.py

# Render with 3D mesh reconstruction
python render.py --model_path output/endonerf/pulling --skip_train --skip_video --reconstruct --configs arguments/endonerf/pulling.py
```

## Evaluation

```bash
python metrics.py --model_paths output/endonerf/pulling
```

## Results

### EndoNeRF Dataset

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| EndoNeRF | 36.062 | 0.933 | 0.089 |
| EndoSurf | 36.529 | 0.954 | 0.074 |
| EndoGaussian (Binocular) | 38.088 | 0.962 | 0.048 |
| **SAGS (Binocular)** | **39.164** | **0.970** | **0.025** |
| EndoGaussian (Monocular) | 37.464 | 0.960 | 0.052 |
| **SAGS (Monocular)** | **37.711** | **0.962** | **0.043** |

### SCARED Dataset (Binocular, Average)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| EndoNeRF | 26.31 | 0.815 | 0.303 |
| EndoSurf | 26.88 | 0.840 | 0.283 |
| EndoGaussian | 29.40 | 0.857 | 0.220 |
| **SAGS (Ours)** | **30.56** | **0.866** | **0.161** |

## Citation

```bibtex
@article{huang2025sags,
  title={SAGS: Self-Adaptive Alias-Free Gaussian Splatting for Dynamic Surgical Endoscopic Reconstruction},
  author={Huang, Wenfeng and Liao, Xiangyun and Qian, Yinling and Liu, Hao and Yang, Yongming and Jia, Wenjing and Wang, Qiong},
  journal={arXiv preprint arXiv:2510.27318},
  year={2025}
}
```

## Acknowledgments

This work builds upon the following projects:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [EndoGaussian](https://github.com/yifliu3/EndoGaussian)
- [Mip-Splatting](https://github.com/autonomousvision/mip-splatting)
- [4D Gaussian Splatting](https://github.com/hustvl/4DGaussians)
- [HexPlane](https://github.com/Caoang327/HexPlane)

## License

This project is licensed under the terms of the LICENSE.md file. See [LICENSE.md](LICENSE.md) for details.
