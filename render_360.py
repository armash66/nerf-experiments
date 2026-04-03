"""
render_360.py — Dedicated 360-Degree Multi-View Rendering Module
Generates a complete orbital rendering sequence from a trained NeRF checkpoint.

Outputs:
    - Per-frame PNGs to outputs/<experiment>/frames/frame_000.png, frame_001.png, ...
    - MP4 video to outputs/<experiment>/orbit.mp4
    - GIF to outputs/<experiment>/orbit.gif

Usage:
    python render_360.py --checkpoint outputs/experiment_1/checkpoints/ckpt_03000.pt \\
                         --experiment experiment_1

    python render_360.py --checkpoint outputs/experiment_1/checkpoints/ckpt_03000.pt \\
                         --experiment experiment_1 --n_frames 120 --radius 4.0

    python render_360.py --checkpoint outputs/experiment_1/checkpoints/ckpt_03000.pt \\
                         --experiment experiment_1 --no_video --no_gif
"""

import os
import sys
import argparse
import numpy as np
import torch
import imageio
from tqdm import trange

from render import load_model, render_image, generate_orbit_poses
from config import get_config


# ---------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------

def parse_render360_args():
    """Parse arguments specific to the 360 rendering module."""
    parser = argparse.ArgumentParser(
        description="360-degree orbital rendering from a trained NeRF checkpoint."
    )

    # Required
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to trained checkpoint .pt file")
    parser.add_argument("--experiment",  type=str, default="experiment_1",
                        help="Experiment name (determines output directory)")

    # Orbit parameters
    parser.add_argument("--n_frames",    type=int,   default=60,
                        help="Number of frames in the 360 orbit (60-120 recommended)")
    parser.add_argument("--elevation",   type=float, default=-30.0,
                        help="Camera elevation angle in degrees (negative = above)")
    parser.add_argument("--radius",      type=float, default=4.0,
                        help="Orbit radius from scene center")

    # Render dimensions
    parser.add_argument("--H",           type=int,   default=100,
                        help="Render height in pixels")
    parser.add_argument("--W",           type=int,   default=100,
                        help="Render width in pixels")
    parser.add_argument("--focal",       type=float, default=138.0,
                        help="Focal length for rendering")

    # Output controls
    parser.add_argument("--no_video",    action="store_true",
                        help="Skip MP4 video generation")
    parser.add_argument("--no_gif",      action="store_true",
                        help="Skip GIF generation")
    parser.add_argument("--fps",         type=int,   default=30,
                        help="Frames per second for MP4 output")
    parser.add_argument("--gif_fps",     type=int,   default=15,
                        help="Frames per second for GIF output")

    render_args, remaining = parser.parse_known_args()
    return render_args, remaining


# ---------------------------------------------
# 360 RENDERING PIPELINE
# ---------------------------------------------

def render_360(render_args, cfg):
    """
    Full 360-degree rendering pipeline.

    1. Load trained model from checkpoint.
    2. Generate orbital camera poses.
    3. Render each frame and save as PNG.
    4. Assemble frames into MP4 and/or GIF.
    """

    # ── Output directories ────────────────────────────────────────
    out_dir    = os.path.join("outputs", render_args.experiment)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────
    model_coarse, model_fine, pos_enc_xyz, pos_enc_dir = load_model(
        render_args.checkpoint, cfg
    )

    H     = render_args.H
    W     = render_args.W
    focal = render_args.focal

    # ── Generate poses ────────────────────────────────────────────
    poses = generate_orbit_poses(
        n_frames  = render_args.n_frames,
        elevation = render_args.elevation,
        radius    = render_args.radius,
    )

    print(f"\n-- 360 Rendering Configuration ----------")
    print(f"  Checkpoint:   {render_args.checkpoint}")
    print(f"  Experiment:   {render_args.experiment}")
    print(f"  Frames:       {render_args.n_frames}")
    print(f"  Resolution:   {H}x{W}")
    print(f"  Elevation:    {render_args.elevation} deg")
    print(f"  Radius:       {render_args.radius}")
    print(f"  Output:       {frames_dir}/")
    print(f"-----------------------------------------\n")

    # ── Render loop ───────────────────────────────────────────────
    frames = []
    print(f"Rendering {len(poses)} frames...")

    for i in trange(len(poses), desc="Frames"):
        rgb, _ = render_image(
            poses[i],
            model_coarse, model_fine,
            pos_enc_xyz, pos_enc_dir,
            H, W, focal, cfg,
        )
        frames.append(rgb)

        # Save individual frame
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
        imageio.imwrite(frame_path, rgb)

    print(f"Saved {len(frames)} frames to {frames_dir}/")

    # ── Video generation (MP4) ────────────────────────────────────
    if not render_args.no_video:
        video_path = os.path.join(out_dir, "orbit.mp4")
        imageio.mimwrite(video_path, frames, fps=render_args.fps, quality=8)
        print(f"MP4 saved: {video_path}")

    # ── GIF generation ────────────────────────────────────────────
    if not render_args.no_gif:
        gif_path = os.path.join(out_dir, "orbit.gif")
        # Subsample frames for GIF to reduce file size
        gif_step = max(1, render_args.fps // render_args.gif_fps)
        gif_frames = frames[::gif_step]
        imageio.mimwrite(gif_path, gif_frames, duration=1000 // render_args.gif_fps)
        print(f"GIF saved: {gif_path}")

    print(f"\n360 rendering complete. All outputs in: {out_dir}/")


# ---------------------------------------------
# ENTRY POINT
# ---------------------------------------------

if __name__ == "__main__":
    render_args, remaining = parse_render360_args()

    # Load training config (pass remaining args to config parser)
    sys.argv = [sys.argv[0]] + remaining
    cfg = get_config()

    render_360(render_args, cfg)
