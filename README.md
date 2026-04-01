# nerf-experiments

A from-scratch PyTorch implementation of [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) (Mildenhall et al., ECCV 2020).

Built to understand the method deeply — every component is written from scratch with detailed comments tracing back to the paper.

## Features

- **Full NeRF architecture** — 8-layer MLP with skip connections, view-dependent color, geometry/appearance separation
- **Coarse-to-fine rendering** — stratified coarse sampling + hierarchical importance sampling
- **Differentiable volume rendering** — correct alpha compositing with transmittance
- **TinyNeRF mode** — lightweight model for fast pipeline validation (`--use_tiny`)
- **Checkpoint & resume** — save/load training state at any point
- **Configurable everything** — all hyperparameters exposed via CLI flags
- **Novel view synthesis** — render images and 360° orbit videos from trained models

## Project Structure

```
nerf-experiments/
├── nerf/
│   ├── encoding.py      # Positional encoding (Fourier features)
│   ├── model.py         # NeRF MLP — full model + TinyNeRF variant
│   ├── ray.py           # Ray generation from camera intrinsics + pose
│   └── renderer.py      # Stratified/hierarchical sampling + volume rendering
├── train.py             # Training loop with logging, metrics, checkpoints
├── render.py            # Novel view synthesis (single image + orbit video)
├── config.py            # All hyperparameters (argparse CLI)
├── tiny_nerf_data.npz   # Bundled dataset (106 images of a Lego scene)
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

The dataset (`tiny_nerf_data.npz`) is already included in the repo.

```bash
# Fast validation run with TinyNeRF (~5 min on GPU, ~15 min on CPU)
python train.py --use_tiny --n_iters 3000

# Full model (matches paper architecture)
python train.py --n_iters 10000

# Resume from checkpoint
python train.py --resume outputs/checkpoints/ckpt_05000.pt
```

Training outputs are saved to `outputs/`:
- `outputs/renders/` — test view renders at regular intervals
- `outputs/checkpoints/` — model checkpoints
- `outputs/training_curves.png` — loss and PSNR plots

### 3. Render novel views

```bash
# Single image from a new viewpoint
python render.py --checkpoint outputs/checkpoints/ckpt_10000.pt

# 360° orbit video
python render.py --checkpoint outputs/checkpoints/ckpt_10000.pt --video --n_frames 120
```

## Configuration

All settings can be overridden from the command line. Key flags:

| Flag | Default | Description |
|---|---|---|
| `--use_tiny` | `false` | Use TinyNeRF (faster, no view dependence) |
| `--n_iters` | `10000` | Training iterations |
| `--n_train` | `100` | Number of training images |
| `--batch_size` | `1024` | Rays per batch |
| `--lr` | `5e-4` | Learning rate |
| `--n_coarse` | `64` | Coarse samples per ray |
| `--n_fine` | `128` | Fine samples per ray (0 to disable) |
| `--log_every` | `100` | Print metrics every N steps |
| `--render_every` | `500` | Save test render every N steps |

Run `python config.py` to see all available options.

## Method Overview

```
Input Images + Camera Poses
        │
        ▼
   Ray Generation ─────────── ray.py
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

The coarse network guides importance sampling for the fine network, concentrating compute where geometry actually exists.

## References

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) — Mildenhall et al., ECCV 2020
- [Tiny NeRF notebook](https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb) — minimal reference by the original authors
- [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) — faithful PyTorch port by Yen-Chen Lin

## License

MIT
