# nerf-experiments

> Rebuilding how machines see space — from flat images to living 3D worlds.

A from-scratch PyTorch implementation of Neural Radiance Fields (NeRF), built to understand the fundamental limits of 2D-to-3D reconstruction and push toward multimodal scene understanding.

---

## What this is

Most 3D reconstruction pipelines are black boxes. This project is the opposite — every component is written from scratch, understood before it's used, and designed to be modified.

The baseline follows the original NeRF paper (Mildenhall et al., 2020). The experiments go beyond it.

---

## Research direction

Standard NeRF takes RGB images and learns to represent a scene volumetrically. That works well when the scene is fully visible. It breaks down when it isn't.

This project investigates:

- What NeRF actually *can't* reconstruct — occluded geometry, unseen surfaces
- Where the line is between reconstruction and hallucination
- Whether fusing signals beyond RGB (depth, thermal) can recover information that light alone cannot capture

The goal is not to beat benchmarks. It's to understand the boundaries of the method and build something honest about what it knows and doesn't.

---

## Structure

```
nerf-experiments/
├── nerf/
│   ├── rays.py          # Ray generation from camera intrinsics + pose
│   ├── encoding.py      # Positional encoding (Fourier features)
│   ├── model.py         # NeRF MLP — (x,y,z,θ,φ) → (RGB, density)
│   └── renderer.py      # Volume rendering along rays
├── train.py             # Training loop
├── render.py            # Novel view synthesis from trained model
├── config.py            # Hyperparameters
├── requirements.txt
└── .gitignore
```

---

## Pipeline

```
Images + Poses
      ↓
  Ray Generation          rays.py
      ↓
  Point Sampling          renderer.py
      ↓
  Positional Encoding     encoding.py
      ↓
  MLP Query               model.py
  (x,y,z,θ,φ) → (RGB, σ)
      ↓
  Volume Rendering        renderer.py
      ↓
  Rendered Image → Loss → Backprop
```

---

## Implementation status

| Component | Status |
|---|---|
| Ray generation | ✅ Done |
| Positional encoding | 🔄 In progress |
| NeRF MLP | ⬜ Pending |
| Volume renderer | ⬜ Pending |
| Training loop | ⬜ Pending |
| Novel view synthesis | ⬜ Pending |

---

## Setup

```bash
git clone https://github.com/your-username/nerf-experiments
cd nerf-experiments
pip install -r requirements.txt
```

Training on the Blender synthetic dataset:

```bash
python train.py --config config.py --scene lego
```

---

## References

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) — Mildenhall et al., 2020
- [Tiny NeRF](https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb) — minimal reference implementation

---

*Built to understand, not just to replicate.*
