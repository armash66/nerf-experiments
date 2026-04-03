# Neural Radiance Fields: Modular Implementation and Ablations

A from-scratch, research-oriented PyTorch implementation of [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) (Mildenhall et al., ECCV 2020).

This repository is built to understand the method deeply rather than just replicate it. Every component is written from scratch, mathematically aligned with the original paper, and structured to support systematic ablation studies and methodological extensions.

## Core Features

- **Full NeRF Architecture** — 8-layer MLP with skip connections, view-dependent color conditioning, and explicit geometry/appearance separation.
- **Hierarchical Volume Sampling** — Implements both stratified coarse sampling and inverse-transform fine importance sampling.
- **Differentiable Volume Rendering** — Geometrically correct alpha compositing using accumulated transmittance.
- **TinyNeRF Validation Strategy** — Includes a lightweight, continuous MLP variant (`--use_tiny`) for rapid pipeline validation before scaling compute.
- **Structured Experiment Tracking** — Per-experiment output isolation with CSV logging, configuration persistence, checkpointing, and automated renders.
- **360-Degree Orbital Rendering** — Dedicated module for generating multi-view frame sequences, MP4 videos, and GIFs from trained checkpoints.
- **Cross-Experiment Evaluation** — Comparison utility for analyzing PSNR/loss curves across independent experimental runs.
- **Checkpoint Resume** — Resume interrupted training from any saved checkpoint without data loss.

## Project Architecture

The codebase is purposefully modular to isolate the rendering equation from the neural representation, enabling rapid prototyping of new sampling or encoding strategies. Successive experiments and ablation studies are isolated in the `experiments/` directory to preserve historical benchmarks.

```text
nerf-experiments/
├── nerf/
│   ├── encoding.py      # Positional encoding (high-frequency Fourier features)
│   ├── model.py         # Implicit neural representations (Continuous MLPs)
│   ├── ray.py           # Pinhole camera model and ray generation (intrinsics/extrinsics)
│   └── renderer.py      # Stratified/hierarchical sampling + volume rendering equation
├── experiments/         # Detailed experimental logs, metrics, and ablation studies
│   └── exp1_tiny_nerf_baseline.md
├── train.py             # Main optimization loop with CSV logging and checkpoints
├── render.py            # Novel view synthesis inference (single image + orbit video)
├── render_360.py        # Dedicated 360-degree multi-view rendering pipeline
├── evaluate.py          # Cross-experiment comparison and evaluation utility
├── config.py            # Centralized hyperparameter configuration (argparse)
├── data/
│   └── tiny_nerf_data.npz   # Bundled synthetic dataset
└── requirements.txt
```

### Per-Experiment Output Structure

Each training run produces an isolated output directory under `outputs/<experiment_name>/`:

```text
outputs/
└── experiment_1/
    ├── frames/              # 360-degree render sequence (per-frame PNGs)
    │   ├── frame_000.png
    │   ├── frame_001.png
    │   └── ...
    ├── renders/             # Intermediate validation renders during training
    │   ├── render_00500.png
    │   ├── render_01000.png
    │   └── render_final.png
    ├── logs/
    │   └── train_log.csv    # Quantitative metrics (iteration, loss, psnr, lr, time)
    ├── checkpoints/
    │   └── ckpt_03000.pt    # Serialized model weights + optimizer state
    ├── orbit.mp4            # Assembled 360 video
    ├── orbit.gif            # Assembled 360 GIF
    ├── training_curves.png  # Loss and PSNR plots
    └── config.txt           # Frozen hyperparameter snapshot
```

## Quick Start & Reproducibility

### 1. Environment Setup

Ensure you have a modern GPU environment configured with PyTorch, then install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Network Optimization

The repository includes a bundled dataset (`data/tiny_nerf_data.npz`) for immediate training.

```bash
# Fast validation run with TinyNeRF (~5 min on GPU)
python train.py --use_tiny --n_iters 3000 --experiment experiment_1

# Full hierarchical NeRF model (matches paper architecture)
python train.py --n_iters 10000 --experiment experiment_2

# Resume optimization from a specific convergence point
python train.py --resume outputs/experiment_1/checkpoints/ckpt_03000.pt --experiment experiment_1
```

Training automatically generates:
- `outputs/<experiment>/renders/` — Intermediate test view renders at regular intervals.
- `outputs/<experiment>/checkpoints/` — Serialized network weights and optimizer states.
- `outputs/<experiment>/logs/train_log.csv` — Quantitative metrics in machine-readable format.
- `outputs/<experiment>/config.txt` — Frozen hyperparameter configuration.
- `outputs/<experiment>/training_curves.png` — Visualized loss and PSNR trajectories.

### 3. 360-Degree Rendering

Generate multi-view frame sequences and assembled videos from trained checkpoints:

```bash
# Standard 60-frame orbital render
python render_360.py --checkpoint outputs/experiment_1/checkpoints/ckpt_03000.pt \
                     --experiment experiment_1

# High-density 120-frame render with custom orbit parameters
python render_360.py --checkpoint outputs/experiment_1/checkpoints/ckpt_03000.pt \
                     --experiment experiment_1 --n_frames 120 --radius 4.0

# Frames only (skip video/GIF assembly)
python render_360.py --checkpoint outputs/experiment_1/checkpoints/ckpt_03000.pt \
                     --experiment experiment_1 --no_video --no_gif
```

### 4. Single-View Rendering

Generate individual novel views or orbit videos using the general render module:

```bash
# Render a single high-resolution image from a targeted viewpoint
python render.py --checkpoint outputs/experiment_1/checkpoints/ckpt_03000.pt --use_tiny

# Render a cinematic orbit video
python render.py --checkpoint outputs/experiment_1/checkpoints/ckpt_03000.pt --video --use_tiny
```

### 5. Experiment Evaluation

Compare training metrics across experiments:

```bash
# Single experiment analysis
python evaluate.py --experiments experiment_1

# Cross-experiment comparison
python evaluate.py --experiments experiment_1 experiment_2 experiment_3
```

## Training Log Format

The CSV log file (`train_log.csv`) uses the following schema:

```text
iteration,loss,psnr,lr,time
100,0.050000,12.90,5.00e-04,2.3
500,0.010000,20.10,4.95e-04,10.5
```

| Field | Type | Description |
|---|---|---|
| `iteration` | int | Current optimization step |
| `loss` | float | Combined MSE loss (coarse + fine) |
| `psnr` | float | Peak Signal-to-Noise Ratio in dB |
| `lr` | float | Current learning rate |
| `time` | float | Elapsed wall-clock time in seconds |

## Configuration & Hyperparameters

The pipeline is highly controllable. All critical hyperparameters can be overridden from the command line interface.

| Flag | Default | Description |
|---|---|---|
| `--experiment` | `experiment_1` | Experiment name (isolates output directory) |
| `--use_tiny` | `false` | Fallback to TinyNeRF architecture (faster, excludes view dependence) |
| `--n_iters` | `10000` | Total optimization iterations |
| `--n_train` | `100` | Number of training images allocated per batch |
| `--batch_size` | `1024` | Number of ray samples per optimization step |
| `--lr` | `5e-4` | Initial Adam learning rate |
| `--n_coarse` | `64` | Initial stratified volume samples per ray |
| `--n_fine` | `128` | Secondary hierarchical samples per ray (0 to disable) |
| `--log_every` | `100` | Frequency of terminal metric output and CSV logging |
| `--render_every` | `500` | Frequency of intermediate validation renders |
| `--resume` | `None` | Path to checkpoint for resuming training |

For a comprehensive list of algorithmic toggles, run `python config.py`.

## Method Overview & Computational Pipeline

```text
Input Images + Camera Poses
        │
        ▼
   Ray Generation ─────────── ray.py
   r(t) = o + t*d
        │
        ▼
   Stratified Sampling ────── renderer.py
        │
        ▼
   Positional Encoding ────── encoding.py
   γ(x) = [x, sin(2⁰x), cos(2⁰x), ..., sin(2^(L-1)x), cos(2^(L-1)x)]
        │
        ▼
   NeRF MLP ───────────────── model.py
   (encoded xyz, encoded dir) → (RGB, σ)
        │
        ▼
   Volume Rendering ──────── renderer.py
   C(r) = Σ Tᵢ · αᵢ · cᵢ
        │
        ▼
   MSE Loss → Backprop
```

During the full forward pass, the coarse network establishes a rough probability density function of surface locations. This PDF guides the fine network's sampling strategy, effectively concentrating computational capacity on areas where solid geometry actually exists.

## References

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) — Mildenhall et al., ECCV 2020
- [Tiny NeRF notebook](https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb) — minimal reference by the original authors
- [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) — faithful PyTorch port by Yen-Chen Lin

## License

MIT
