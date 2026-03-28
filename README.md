# nerf-experiments

A minimal implementation of Neural Radiance Fields (NeRF) for learning 3D scene representations from multi-view images and rendering novel viewpoints.

## Overview

This project focuses on understanding and implementing the core components of NeRF from scratch. The goal is to study how a neural network can represent a continuous 3D scene and render images from arbitrary viewpoints.

## Objectives

- Implement the NeRF pipeline step by step
- Understand ray-based scene representation and volume rendering
- Train a model to reconstruct scenes from images
- Analyze behavior under limited input views

## Structure

```bash
nerf-experiments/
├── data/
├── models/
├── utils/
├── train.py
├── render.py
├── config.py
```

## Implementation Plan

The project is organized into modular components, each responsible for a specific part of the NeRF pipeline. The implementation will proceed step by step, ensuring clarity and control over each stage.

### 1. Ray Generation (`utils/rays.py`)

- Convert image pixels into rays in 3D space.
- For each pixel:
  - Compute ray origin (camera position)
  - Compute ray direction (through pixel into scene)
- Use camera intrinsics (focal length) and extrinsics (pose matrix).

**Output:**
- `rays_o`: ray origins
- `rays_d`: ray directions

---

### 2. Sampling Along Rays (`utils/sampling.py`)

- Sample a fixed number of points along each ray between near and far bounds.
- These points represent candidate locations in 3D space.

**Purpose:**
- Approximate the continuous scene using discrete samples.

**Output:**
- 3D sample points per ray

---

### 3. Neural Network (`models/nerf_mlp.py`)

- A multi-layer perceptron (MLP) that models the scene.
- Input:
  - 3D coordinates (x, y, z)
- Output:
  - Color (R, G, B)
  - Density (σ)

**Role:**
- Learn how the scene behaves at any 3D location.

---

### 4. Volume Rendering (`utils/rendering.py`)

- Combine sampled points along each ray to produce a final pixel color.
- Use predicted density values to compute contribution weights.
- Closer and denser points contribute more.

**Key idea:**
- Simulate how light accumulates along a ray.

**Output:**
- Rendered pixel color

---

### 5. Training Pipeline (`train.py`)

- Load images and camera poses.
- Generate rays for each image.
- Sample points along rays.
- Pass samples through the network.
- Render predicted image.
- Compute loss against ground truth.
- Backpropagate and update model.

**Loss Function:**
- Mean Squared Error (MSE)

---

### 6. View Synthesis (`render.py`)

- Use trained model to render images from new camera poses.
- Generate rays from new viewpoints.
- Pass through full pipeline (sampling + network + rendering).

**Result:**
- Novel view generation

---

### 7. Configuration (`config.py`)

- Store hyperparameters such as:
  - Number of samples per ray
  - Learning rate
  - Training steps
  - Near/far bounds

---

## Development Approach

The implementation will follow a staged process:

1. Implement each component independently
2. Validate outputs at every step
3. Integrate components gradually
4. Train on a small dataset
5. Extend with experiments (e.g., fewer input views)

The focus is on understanding and control, not just end results.

## Status

Initial implementation in progress.

## Direction

The project will extend toward experimentation with sparse-view reconstruction and performance improvements over the baseline model.

## References

Based on the original NeRF paper and related work available on arXiv.