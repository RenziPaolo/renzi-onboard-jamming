# RFInject

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%2BCuPy%20GPU-orange.svg)](https://pytorch.org/)
[![spikingjelly](https://img.shields.io/badge/spikingjelly-0.0.6-red.svg)](https://github.com/fangweizhen/spikingjelly)

A PyTorch-based deep-learning framework for **detecting, localizing, and segmenting Radio
Frequency Interference (RFI)** in **Synthetic Aperture Radar (SAR)** imagery. RFInject provides
three model families — **classification**, **segmentation / bounding-box localization**, and
**anomaly detection** — that run as standard ANNs or as **spiking neural networks (SNNs)** via
[spikingjelly](https://github.com/fangweizhen/spikingjelly).

> A research thesis project by **Paolo Renzi**. Managed with full operational freedom.
> Only constraints: maintainable code and transparent progress reporting.

---

## Overview

RFI — unwanted electromagnetic signals from communication/navigation/electronic systems — corrupts
SAR images and degrades downstream tasks (ship detection, flood mapping, interferometry). RFInject
takes a deep-learning approach: given a SAR "echo" and an "RFI" channel, the system can **detect**
whether RFI is present, **localize** it with a bounding box, **segment** it pixel-wise, or
**reconstruct** the clean image. It also benchmarks model **robustness** to synthetic noise and RFI,
comparing ANNs against SNNs under energy-based and firing-rate-based decision rules.

---

## Project Structure

```
RFInject/
├── dataset/
│   ├── dataset.py          # SARZarrDataset — lazy Zarr loading, resize & normalize
│   ├── fast_dataset.py     # fastSARZarrDataset — pre-loaded tensor cache
│   └── patch_dataset.py    # SARZarrPatchDataset — fixed-size patch splitting
├── models/
│   ├── classification/     # SimpleLinear, CNN (simp), SNN (LIF), CSNN, LCNN-GWHA
│   ├── segmentation/       # Spiking ResFormer (DSSA+GWFFN), Unet, SSD/FPN/Seg heads
│   └── anomalyDetection/   # Spiking CNN-VAE autoencoders, Inception-ResNet hybrids
├── training/               # training_simple.py, train_snn.py, train_segmentation.py, train_anomaly.py
├── testing/                # test_thesis.py (robustness), test_seg.py (bbox), test_anomaly.py
├── plots/                  # plot_dist_simple.py, plot_dist_thesis.py, plot_thesis.py, plot_anomaly.py
├── podman/                 # Dockerfile, requirements.txt, podmancommand
├── presentation/           # Thesis report & slides
├── test.ipynb              # Interactive notebook
├── test.py                 # Smoke-test script
├── test_thesis.sh          # Batch robustness-eval runner
├── plots.sh                # Batch plotting runner
├── RFInject.yml            # Hydra-style hyperparameters
└── .gitignore
```

The code is organized into three tiers:

- **`dataset/`** — Loads SAR Zarr data (echo + RFI channels, bbox annotations) with lazy loading,
  fast pre-loaded tensors, and patch-based splitting. Supports complex→real/imag channel stacking
  and per-channel min-max normalization.
- **`models/`** — Model definitions across all three tasks.
- **`training/`** — Training scripts with checkpointing, TensorBoard, and `args.txt` reproducibility.
- **`testing/`** — Evaluation scripts for robustness, bbox localization, and anomaly detection.
- **`plots/`** — Distribution plots (Gaussian PDFs of energy/spike rates) and bbox visualization.

---

## Installation

### Option A — Podman / Docker (recommended, GPU)

```bash
# Build
podman build -f podman/Dockerfile -t rfinject:cuda .

# Run (mount code + data, attach GPU)
podman run -it --ipc host --network host \
  -v /home/renzi/:/work/project/ \
  -v /mnt/ssd2/renzi:/mnt/ssd2/renzi \
  --device nvidia.com/gpu=all \
  docker.io/renzipaolo/cachyosdiagh.dockerfile:latest
```

The Dockerfile installs CachyOS, CUDA, cuDNN, Miniconda, and all Python deps.
Working directory inside the container is `/app`.

### Option B — Local Python (CUDA GPU required, Python 3.10+)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install cupy-cuda12x
pip install spikingjelly norse zarr timm piq scikit-learn seaborn matplotlib scipy \
            einops apex tqdm prettytable tensorboard torchviz ujson
# — or —
pip install -r podman/requirements.txt
```

---

## Quick Start

```bash
# 1. Smoke test (dataset + forward pass on GPU)
python test.py

# 2. Train an ANN classifier
python training/training_simple.py -model conv -dataset mnist -epoch 50 -batch 64 -lr 1e-3

# 3. Train a spiking segmentation model (multi-GPU)
torchrun --nproc_per_node=2 training/train_segmentation.py \
    -model s -head ssd -T 4 -epoch 300 -batch 2 -lr 5e-4 \
    -loss MSE_CIoU -fast_dataset -H 224 -W 224

# 4. Evaluate robustness under noise (ANN)
python testing/test_thesis.py \
    --model_path mnist/ann/standardFalse/modelmnist.pt \
    --output_dir mnist/ann/standardFalse/ \
    --model linear --type standard --batch 1 --dataset mnist --tau -4.5

# 5. Test bbox localization
python testing/test_seg.py --model_dir ./logs/ -data_dir /mnt/ssd2/renzi/RFInject \
    --output_dir test_results --fast_dataset --batch 1 --visualize

# 6. Plot energy/spike distributions
python plots/plot_dist_simple.py -json_path mnist/ann/standardFalse/results_mnist_multiple.json

# 7. Run the full batch evaluation matrix
bash test_thesis.sh
```

---

## Models

### Classification (`models/classification/`)

| Model | File | Description |
|-------|------|-------------|
| `SimpleLinearModel` | `linear.py` | Single linear layer (MNIST baseline). |
| `SAR_RF_Interference_Classifier_simp` | `conv.py` | 2-layer CNN, adaptive pool, dropout. 2-channel input. |
| `SAR_RF_Interference_Classifier_SNN` | `snn.py` | LIF-based SNN via spikingjelly, configurable τ. |
| `CSNN` | `CSNN.py` | Convolutional SNN (Flatten→Conv→LIF→Linear). |
| `LCNN_GWHA` | `LCNNGWHA.py` | LCNN + Group-Wise Hybrid Attention (channel + spatial). |

### Segmentation / Localization — Spiking ResFormer (`models/segmentation/`)

Flagship model: a spiking vision-transformer / ConvNeXt hybrid with custom spike-aware blocks
in `spikingresformer.py`:

| Component | Description |
|-----------|-------------|
| `IF` / `LIF` / `PLIF` | Integrate-and-fire, leaky IF (τ=2, ATan surrogate), parametric LIF neurons. |
| `SpikingConv2d` | Conv2d → BatchNorm → Spiking activation. |
| `GWFFN` | Gated Wide Feed-Forward Network with depthwise conv + residuals. |
| `DSSA` | Differential Spiking Self-Attention — firing-rate-normalized multi-head attention. |
| `SpikingResformer` | Full backbone. |
| `SpikingResformerSeg` | + `SpikingSegHead` for semantic segmentation. |
| `SpikingResformerSSD` | SSD-style bbox regression (`SpikingDetectionHeadSSD`). |
| `SpikingResformerFPN` | Feature Pyramid Network head. |
| `Unet` | Spiking U-Net with conv-transpose decoder. |

Model sizes: **`s`**, **`xs`**, **`xxs`**, **`m`**. Channel depths range from `[32, 64]`
up to `[64, 384, 768]`. The config `RFInject.yml` references `MSFormer_10_128_SS` (10-layer
SSD head, 128 channels).

### Anomaly Detection (`models/anomalyDetection/`)

| Model | File | Description |
|-------|------|-------------|
| `CSNNanomaly_image` / `_line` | `CSNNanomaly.py` | Spiking CNN autoencoder (2D or 1D). |
| `CSNNanomaly_*_VAE` / `_LIF` | `CSNNanomaly.py` | VAE and LIF variants, combined `_VAE_LIF`. |
| `CSNNanomaly_norse` | `CSNNanomaly_norse.py` | Norse backend variant. |
| `AnomalyImageDetectionModel` | `FDCCpimage.py` | Inception-v3 + conv head (2D). |
| `AnomalyDetectionModel1D` | `FDCCp.py` | 1D Inception-ResNet-v2 + conv head. |

Uses SSIM + reconstruction MSE (`piq.ssim`) to detect RFI as reconstruction anomalies.

### SNN execution modes

All spiking models support: **multi-step** (`-step_mode m`, `[T,B,C,H,W]`, vectorized) or
**single-step** (`-step_mode s`, Python loop over T). Spike decoding: `temporal` (time-weighted
mean), `rate` (mean over T), or `sum` (accumulate over T).

---

## Training

All scripts save checkpoints, TensorBoard logs, and `args.txt` for reproducibility.

### Classification (ANN)

```bash
python training/training_simple.py -model <linear|conv> -dataset <mnist|fmnist|CIFAR10|EuroSAT> \
    -epoch 100 -batch 64 -lr 1e-3 -data_dir ./data
```

### Spiking Classification (SNN)

```bash
python training/train_snn.py -T 100 -batch 64 -epochs 100 -lr 1e-3 \
    -tau 2.0 -model <standard|conv> -dataset mnist -out-dir ./logs
```

### Segmentation / BBox Localization (DDP-ready)

```bash
torchrun --nproc_per_node=2 training/train_segmentation.py \
    -model s -head ssd -T 4 -epoch 300 -batch 2 -lr 5e-4 \
    -loss MSE_CIoU -scale 2000 -decoding temporal \
    -data_dir /mnt/ssd2/renzi/RFInject -H 224 -W 224 -fast_dataset
```

Key flags: `-model` (s/xs/xxs/m/unet), `-head` (standard/ssd/fpn), `-loss`
(MSE/MAE/CIoU/DIoU/MSE_CIoU), `-decoding` (temporal/rate/sum), `-step_mode` (m/s),
`-scale` (bbox output scaling), `-only_mag`/`-only_ang` (channel selection).

### Anomaly Detection

```bash
python training/train_anomaly.py -model image -vae -T 100 -epoch 100 -batch 16 \
    -lr 1e-3 -loss MSE -data_dir ./data -out-dir ./anomaly_logs
```

---

## Evaluation

| Script | Task | Key metrics |
|--------|------|-------------|
| `testing/test_thesis.py` | Robustness under `none` / `gaussian_noise` / `RFI` | Energy, firing rate, acc/precision/recall/F1 at threshold τ |
| `testing/test_seg.py` | Bbox localization | Loss, IoU, CIoU, DIoU |
| `testing/test_anomaly.py` | Anomaly detection | Reconstruction error, SSIM, MSE |

Batch evaluation: `bash test_thesis.sh` runs all ANN + SNN models across all conditions.

---

## Configuration

`RFInject.yml` — Hydra-style hyperparameters for the segmentation pipeline:

```yaml
dataset: SARZarr
model: MSFormer_10_128_SS
distributed: True
num_classes: 1
img_size: 224
batch_size: 2
lr: 5e-4
sched: cosine
epochs: 300
opt: adam
amp: True
```

> **Note:** `MSFormer_10_128_SS` = 10-layer Spiking ResFormer, 128 channels, SSD head.
> For classification/SNN scripts, hyperparameters are CLI flags instead.

---

## Container Deployment

`podman/` contains the Dockerfile, `requirements.txt`, and `podmancommand` (ready-to-run
podman commands). The image is based on CachyOS with CUDA 12.x and CuPy pre-installed.

```bash
podman build -f podman/Dockerfile -t rfinject:cuda .
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `torch` / `torchvision` | Deep learning framework |
| `cupy-cuda13x` | GPU backend for spikingjelly |
| `spikingjelly` | SNN layers & neurons (LIF, IF, PLIF, PoissonEncoder) |
| `norse` | Alternative SNN library |
| `zarr` | SAR data storage |
| `timm` | Vision transformer utilities |
| `piq` | SSIM / image quality metrics |
| `scikit-learn` | Classification metrics |
| `seaborn`/`matplotlib`/`scipy` | Plotting & distributions |
| `einops` | Tensor reshaping |
| `apex` | Mixed-precision training |
| `prettytable` | Parameter counting |
| `tensorboard`/`torchviz` | Training visualization |
| `tqdm`/`ujson` | Progress bars & JSON I/O |

---

## Citation

```bibtex
@software{renzi2024rfinject,
  author = {Renzi, Paolo},
  title  = {Combining Helmholtz Free Energy and Spike Neural Networks Dynamics for Efficient Raw SAR RFI Detection and Segmentation},
  year   = {2026},
  url    = {https://github.com/RenziPaolo/renzi-onboard-jamming}
}
```

Related libraries: [spikingjelly](https://github.com/fangweizhen/spikingjelly),
[cupy/cupy](https://github.com/cupy/cupy), [norse](https://github.com/norse/norse).

---

