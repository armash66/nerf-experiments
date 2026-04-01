"""
render.py — Novel View Synthesis
Loads a trained NeRF checkpoint and renders images from new camera poses.

Two modes:
    1. Single image: render from one specific pose
    2. Orbit video: render a 360 degree orbit around the scene

Usage:
    # Render test image
    python render.py --checkpoint outputs/checkpoints/ckpt_10000.pt

    # Render 360 orbit video
    python render.py --checkpoint outputs/checkpoints/ckpt_10000.pt --video

    # Render with custom elevation and radius
    python render.py --checkpoint outputs/checkpoints/ckpt_10000.pt --video \
                     --n_frames 120 --elevation 30.0 --radius 4.0
"""

import os
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tqdm import trange

from config        import get_config
from nerf.ray      import get_rays
from nerf.encoding import PositionalEncoding
from nerf.model    import NeRFMLP, TinyNeRFMLP
from nerf.renderer import render_rays


# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# CAMERA POSE GENERATION
# ─────────────────────────────────────────────

def pose_spherical(theta: float, phi: float, radius: float) -> torch.Tensor:
    """
    Generate a camera-to-world matrix for a camera sitting on a sphere,
    looking toward the origin.

    Args:
        theta:  azimuth angle in degrees (0-360, horizontal rotation)
        phi:    elevation angle in degrees (negative = above scene)
        radius: distance from origin
    Returns:
        c2w: (4, 4) camera-to-world matrix
    """
    def rot_x(angle):
        rad = np.deg2rad(angle)
        return np.array([
            [1, 0,           0,          0],
            [0, np.cos(rad), -np.sin(rad), 0],
            [0, np.sin(rad),  np.cos(rad), 0],
            [0, 0,           0,          1],
        ], dtype=np.float32)

    def rot_z(angle):
        rad = np.deg2rad(angle)
        return np.array([
            [np.cos(rad), -np.sin(rad), 0, 0],
            [np.sin(rad),  np.cos(rad), 0, 0],
            [0,           0,           1, 0],
            [0,           0,           0, 1],
        ], dtype=np.float32)

    # Translate camera back by radius
    translate = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, radius],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    c2w = rot_z(theta) @ rot_x(phi) @ translate
    # Flip axes to match NeRF convention
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return torch.tensor(c2w, dtype=torch.float32)


def generate_orbit_poses(
    n_frames:  int   = 40,
    elevation: float = -30.0,
    radius:    float = 4.0,
) -> list[torch.Tensor]:
    """
    Generate a list of poses for a 360 orbit around the scene.

    Args:
        n_frames:  number of frames in the orbit
        elevation: camera elevation in degrees (negative = above scene)
        radius:    orbit radius
    Returns:
        list of (4, 4) c2w matrices
    """
    thetas = np.linspace(0, 360, n_frames, endpoint=False)
    return [pose_spherical(theta, elevation, radius) for theta in thetas]


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str, cfg):
    """
    Load trained model weights from checkpoint.

    Returns:
        model_coarse, model_fine, pos_enc_xyz, pos_enc_dir
    """
    pos_enc_xyz = PositionalEncoding(
        num_freqs=cfg.num_freqs_xyz, input_dim=3
    ).to(device)
    pos_enc_dir = PositionalEncoding(
        num_freqs=cfg.num_freqs_dir, input_dim=3
    ).to(device)

    if cfg.use_tiny:
        model_coarse = TinyNeRFMLP(
            xyz_dim    = pos_enc_xyz.output_dim,
            hidden_dim = cfg.hidden_dim,
        ).to(device)
        model_fine = None
    else:
        model_coarse = NeRFMLP(
            xyz_dim    = pos_enc_xyz.output_dim,
            dir_dim    = pos_enc_dir.output_dim,
            hidden_dim = cfg.hidden_dim,
            num_layers = cfg.num_layers,
            skip_layer = cfg.skip_layer,
        ).to(device)
        model_fine = NeRFMLP(
            xyz_dim    = pos_enc_xyz.output_dim,
            dir_dim    = pos_enc_dir.output_dim,
            hidden_dim = cfg.hidden_dim,
            num_layers = cfg.num_layers,
            skip_layer = cfg.skip_layer,
        ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_coarse.load_state_dict(ckpt["model_coarse"])
    if model_fine and "model_fine" in ckpt:
        model_fine.load_state_dict(ckpt["model_fine"])

    model_coarse.eval()
    if model_fine:
        model_fine.eval()

    print(f"Loaded checkpoint: {checkpoint_path}  (step {ckpt['step']})")
    return model_coarse, model_fine, pos_enc_xyz, pos_enc_dir


# ─────────────────────────────────────────────
# SINGLE IMAGE RENDER
# ─────────────────────────────────────────────

@torch.no_grad()
def render_image(
    pose:         torch.Tensor,
    model_coarse: torch.nn.Module,
    model_fine,
    pos_enc_xyz:  torch.nn.Module,
    pos_enc_dir:  torch.nn.Module,
    H: int, W: int, focal: float,
    cfg,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Render a single image from a given camera pose.

    Returns:
        rgb:   (H, W, 3) uint8
        depth: (H, W)    float32 normalized to [0, 1]
    """
    rays_o, rays_d = get_rays(H, W, focal, pose.to(device))
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    color_chunks = []
    depth_chunks = []

    for i in range(0, rays_o.shape[0], cfg.batch_size):
        out = render_rays(
            rays_o[i:i+cfg.batch_size],
            rays_d[i:i+cfg.batch_size],
            model_coarse     = model_coarse,
            pos_enc_xyz      = pos_enc_xyz,
            pos_enc_dir      = pos_enc_dir,
            near             = cfg.near,
            far              = cfg.far,
            n_coarse         = cfg.n_coarse,
            randomize        = False,
            model_fine       = model_fine,
            n_fine           = cfg.n_fine if not cfg.use_tiny else 0,
            white_background = cfg.white_bg,
        )
        key = "fine" if "fine" in out else "coarse"
        color_chunks.append(out[key]["color_map"].cpu())
        depth_chunks.append(out[key]["depth_map"].cpu())

    rgb   = torch.cat(color_chunks).reshape(H, W, 3).numpy()
    depth = torch.cat(depth_chunks).reshape(H, W).numpy()

    # Normalize depth to [0, 1] for visualization
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    return rgb, depth


# ─────────────────────────────────────────────
# ORBIT VIDEO
# ─────────────────────────────────────────────

def render_video(
    poses:        list[torch.Tensor],
    model_coarse: torch.nn.Module,
    model_fine,
    pos_enc_xyz:  torch.nn.Module,
    pos_enc_dir:  torch.nn.Module,
    H: int, W: int, focal: float,
    cfg,
    save_path:    str,
):
    """
    Render an orbit video and save as .mp4.

    Args:
        poses:     list of (4,4) camera poses
        save_path: output path e.g. "outputs/orbit.mp4"
    """
    frames = []
    print(f"\nRendering {len(poses)} frames...")

    for i, pose in enumerate(poses):
        rgb, _ = render_image(
            pose, model_coarse, model_fine,
            pos_enc_xyz, pos_enc_dir,
            H, W, focal, cfg,
        )
        frames.append(rgb)
        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{len(poses)}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimwrite(save_path, frames, fps=30, quality=8)
    print(f"Video saved: {save_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def add_render_args(cfg):
    """Add render-specific args on top of training config."""
    import argparse
    parser = argparse.ArgumentParser(parents=[argparse.ArgumentParser()], add_help=False)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint .pt file")
    parser.add_argument("--video",      action="store_true",
                        help="Render 360 orbit video instead of single image")
    parser.add_argument("--n_frames",   type=int,   default=40,
                        help="Number of frames in orbit video")
    parser.add_argument("--elevation",  type=float, default=-30.0,
                        help="Camera elevation for orbit (degrees)")
    parser.add_argument("--radius",     type=float, default=4.0,
                        help="Orbit radius")
    parser.add_argument("--H",          type=int,   default=100,
                        help="Render height")
    parser.add_argument("--W",          type=int,   default=100,
                        help="Render width")
    parser.add_argument("--focal",      type=float, default=138.0,
                        help="Focal length for render poses")
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    import sys
    import argparse

    # Parse render-specific args
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video",      action="store_true")
    parser.add_argument("--n_frames",   type=int,   default=40)
    parser.add_argument("--elevation",  type=float, default=-30.0)
    parser.add_argument("--radius",     type=float, default=4.0)
    parser.add_argument("--H",          type=int,   default=100)
    parser.add_argument("--W",          type=int,   default=100)
    parser.add_argument("--focal",      type=float, default=138.0)
    render_args, remaining = parser.parse_known_args()

    # Load training config (reads remaining args)
    sys.argv = [sys.argv[0]] + remaining
    cfg = get_config()

    # Load model
    model_coarse, model_fine, pos_enc_xyz, pos_enc_dir = load_model(
        render_args.checkpoint, cfg
    )

    H     = render_args.H
    W     = render_args.W
    focal = render_args.focal

    out_dir = cfg.out_dir

    if render_args.video:
        # 360 orbit video
        poses = generate_orbit_poses(
            n_frames  = render_args.n_frames,
            elevation = render_args.elevation,
            radius    = render_args.radius,
        )
        render_video(
            poses, model_coarse, model_fine,
            pos_enc_xyz, pos_enc_dir,
            H, W, focal, cfg,
            save_path=os.path.join(out_dir, "orbit.mp4"),
        )

    else:
        # Single test image
        pose = pose_spherical(theta=0.0, phi=-30.0, radius=render_args.radius)
        rgb, depth = render_image(
            pose, model_coarse, model_fine,
            pos_enc_xyz, pos_enc_dir,
            H, W, focal, cfg,
        )

        # Save and display
        os.makedirs(out_dir, exist_ok=True)
        imageio.imwrite(os.path.join(out_dir, "render.png"), rgb)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(rgb);              ax1.set_title("RGB");  ax1.axis("off")
        ax2.imshow(depth, cmap="plasma"); ax2.set_title("Depth"); ax2.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "render_with_depth.png"))
        plt.show()
        print(f"Saved to {out_dir}/")