# Neural Radiance Fields: Modular Implementation and Ablations

A from-scratch, research-oriented PyTorch implementation of [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) (Mildenhall et al., ECCV 2020).

This repository is built to understand the method deeply rather than just replicate it. Every component is written from scratch, mathematically aligned with the original paper, and structured to support systematic ablation studies and methodological extensions.

## Core Features

- **Full NeRF Architecture** — 8-layer MLP with skip connections, view-dependent color conditioning, and explicit geometry/appearance separation.
- **Hierarchical Volume Sampling** — Implements both stratified coarse sampling and inverse-transform fine importance sampling.
- **Differentiable Volume Rendering** — Geometrically correct alpha compositing using accumulated transmittance.
- **TinyNeRF Validation Strategy** — Includes a lightweight, continuous MLP variant (`--use_tiny`) for rapid pipeline validation before scaling compute.
- **Experiment Tracking** — Built-in checkpointing, automated validation renders, and continuous metric tracking (MSE, PSNR).
- **Novel View Synthesis** — Tools to render individual novel viewpoints or dynamically generate 360° orbital camera trajectories.

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
├── train.py             # Main optimization loop with logging and checkpoints
├── render.py            # Novel view synthesis inference scripts
├── config.py            # Centralized hyperparameter configuration (argparse)
├── data/
│   └── tiny_nerf_data.npz   # Bundled synthetic dataset
└── requirements.txt
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
python train.py --use_tiny --n_iters 3000

# Full hierarchical NeRF model (matches paper architecture)
python train.py --n_iters 10000

# Resume optimization from a specific convergence point
python train.py --resume outputs/checkpoints/ckpt_05000.pt
```

Optimization artifacts are saved dynamically to `outputs/`:
- `outputs/renders/` — Intermediate test view synthesis to monitor qualitative progress.
- `outputs/checkpoints/` — Network weights and optimizer states.
- `outputs/training_curves.png` — Visualized loss and PSNR trajectories.

### 3. Novel View Synthesis

Generate new views using fully converged networks:

```bash
# Render a single high-resolution image from a targeted viewpoint
python render.py --checkpoint outputs/checkpoints/ckpt_10000.pt

# Render a cinematic 360° orbital trajectory
python render.py --checkpoint outputs/checkpoints/ckpt_10000.pt --video --n_frames 120
```

## Configuration & Hyperparameters

The pipeline is highly controllable. All critical hyperparameters can be overridden from the command line interface. 

| Flag | Default | Description |
|---|---|---|
| `--use_tiny` | `false` | Fallback to TinyNeRF architecture (faster, excludes view dependence) |
| `--n_iters` | `10000` | Total optimization iterations |
| `--n_train` | `100` | Number of training images allocated per batch |
| `--batch_size` | `1024` | Number of ray samples per optimization step |
| `--lr` | `5e-4` | Initial Adam learning rate |
| `--n_coarse` | `64` | Initial stratified volume samples per ray |
| `--n_fine` | `128` | Secondary hierarchical samples per ray (0 to disable) |
| `--log_every` | `100` | Frequency of terminal metric output |
| `--render_every` | `500` | Frequency of intermediate validation renders |

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
