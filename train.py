"""
train.py — NeRF Training Loop
Ties together all modules:
    config.py    → hyperparameters
    rays.py      → ray generation
    encoding.py  → positional encoding
    model.py     → NeRF MLP
    renderer.py  → volume rendering + sampling

Usage:
    python train.py                          # default config
    python train.py --use_tiny               # faster, smaller model
    python train.py --n_iters 20000 --lr 1e-3
    python train.py --resume outputs/checkpoints/ckpt_05000.pt
"""

import os
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

from config   import get_config
from nerf.ray      import get_rays
from nerf.encoding import PositionalEncoding
from nerf.model    import NeRFMLP, TinyNeRFMLP
from nerf.renderer import render_rays


# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data(path: str):
    """
    Load TinyNeRF .npz dataset.
    Download: https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz

    Returns:
        images: (N, H, W, 3) float32 in [0, 1]
        poses:  (N, 4, 4)    camera-to-world matrices
        focal:  float        focal length in pixels
        H, W:   int          image dimensions
    """
    data   = np.load(path)
    images = torch.tensor(data["images"], dtype=torch.float32)
    poses  = torch.tensor(data["poses"],  dtype=torch.float32)
    focal  = float(data["focal"])
    H, W   = images.shape[1], images.shape[2]
    print(f"Loaded {len(images)} images — H={H} W={W} focal={focal:.2f}")
    return images, poses, focal, H, W


# ─────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────

def build_model(cfg, pos_enc_xyz, pos_enc_dir):
    """
    Build coarse (and optionally fine) NeRF models based on config.

    Full NeRF uses two separate networks:
        - coarse: sampled uniformly, guides fine sampling
        - fine:   sampled importance-weighted, final output

    TinyNeRF uses one network, no fine pass.
    """
    if cfg.use_tiny:
        model_coarse = TinyNeRFMLP(
            xyz_dim    = pos_enc_xyz.output_dim,
            hidden_dim = cfg.hidden_dim,
        ).to(device)
        model_fine = None
        print("Using TinyNeRFMLP (no view dependence, no fine pass)")

    else:
        model_coarse = NeRFMLP(
            xyz_dim    = pos_enc_xyz.output_dim,
            dir_dim    = pos_enc_dir.output_dim,
            hidden_dim = cfg.hidden_dim,
            num_layers = cfg.num_layers,
            skip_layer = cfg.skip_layer,
        ).to(device)

        # Fine model has same architecture as coarse
        model_fine = NeRFMLP(
            xyz_dim    = pos_enc_xyz.output_dim,
            dir_dim    = pos_enc_dir.output_dim,
            hidden_dim = cfg.hidden_dim,
            num_layers = cfg.num_layers,
            skip_layer = cfg.skip_layer,
        ).to(device)
        print("Using full NeRFMLP (view dependent, coarse + fine)")

    total_params = sum(p.numel() for p in model_coarse.parameters())
    print(f"Coarse model parameters: {total_params:,}")

    return model_coarse, model_fine


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────

def save_checkpoint(path, step, model_coarse, model_fine, optimizer):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "step":          step,
        "model_coarse":  model_coarse.state_dict(),
        "optimizer":     optimizer.state_dict(),
    }
    if model_fine is not None:
        ckpt["model_fine"] = model_fine.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, model_coarse, model_fine, optimizer):
    ckpt  = torch.load(path, map_location=device)
    model_coarse.load_state_dict(ckpt["model_coarse"])
    if model_fine is not None and "model_fine" in ckpt:
        model_fine.load_state_dict(ckpt["model_fine"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start = ckpt["step"]
    print(f"Resumed from step {start}")
    return start


# ─────────────────────────────────────────────
# TEST RENDER
# ─────────────────────────────────────────────

@torch.no_grad()
def render_test_view(
    model_coarse, model_fine,
    pos_enc_xyz, pos_enc_dir,
    pose, H, W, focal, cfg, step, out_dir,
):
    """Render a full image from a test pose and save it."""
    model_coarse.eval()
    if model_fine:
        model_fine.eval()

    rays_o, rays_d = get_rays(H, W, focal, pose.to(device))
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    chunks = []
    for i in range(0, rays_o.shape[0], cfg.batch_size):
        out = render_rays(
            rays_o[i:i+cfg.batch_size],
            rays_d[i:i+cfg.batch_size],
            model_coarse   = model_coarse,
            pos_enc_xyz    = pos_enc_xyz,
            pos_enc_dir    = pos_enc_dir,
            near           = cfg.near,
            far            = cfg.far,
            n_coarse       = cfg.n_coarse,
            randomize      = False,
            model_fine     = model_fine,
            n_fine         = cfg.n_fine if not cfg.use_tiny else 0,
            white_background = cfg.white_bg,
        )
        # Use fine output if available, else coarse
        key = "fine" if "fine" in out else "coarse"
        chunks.append(out[key]["color_map"].cpu())

    rendered = torch.cat(chunks).reshape(H, W, 3).numpy()
    rendered = (rendered * 255).clip(0, 255).astype(np.uint8)

    render_dir = os.path.join(out_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)
    path = os.path.join(render_dir, f"step_{step:05d}.png")
    imageio.imwrite(path, rendered)

    model_coarse.train()
    if model_fine:
        model_fine.train()

    return rendered


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train(cfg):
    # ── Directories ───────────────────────────────────────────────
    ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────
    images, poses, focal, H, W = load_data(cfg.data_path)
    train_images = images[:cfg.n_train].to(device)
    train_poses  = poses[:cfg.n_train].to(device)
    test_pose    = poses[cfg.test_idx]

    # ── Encodings ─────────────────────────────────────────────────
    pos_enc_xyz = PositionalEncoding(
        num_freqs=cfg.num_freqs_xyz, input_dim=3
    ).to(device)
    pos_enc_dir = PositionalEncoding(
        num_freqs=cfg.num_freqs_dir, input_dim=3
    ).to(device)

    # ── Models ────────────────────────────────────────────────────
    model_coarse, model_fine = build_model(cfg, pos_enc_xyz, pos_enc_dir)

    params = list(model_coarse.parameters())
    if model_fine:
        params += list(model_fine.parameters())

    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=cfg.lr_decay
    )

    # ── Resume ────────────────────────────────────────────────────
    start = 0
    if cfg.resume:
        start = load_checkpoint(
            cfg.resume, model_coarse, model_fine, optimizer
        )

    # ── Metrics ───────────────────────────────────────────────────
    loss_history = []
    psnr_history = []

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"\nTraining on {device} for {cfg.n_iters} iterations...\n")

    # ── Loop ──────────────────────────────────────────────────────
    for i in trange(start, cfg.n_iters, desc="Training"):

        # Random training image
        idx   = torch.randint(0, cfg.n_train, (1,)).item()
        image = train_images[idx]   # (H, W, 3)
        pose  = train_poses[idx]    # (4, 4)

        # Generate rays
        rays_o, rays_d = get_rays(H, W, focal, pose)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        target = image.reshape(-1, 3)

        # Random ray batch
        perm   = torch.randperm(rays_o.shape[0], device=device)[:cfg.batch_size]
        rays_o = rays_o[perm]
        rays_d = rays_d[perm]
        target = target[perm]

        # Forward pass
        out = render_rays(
            rays_o, rays_d,
            model_coarse     = model_coarse,
            pos_enc_xyz      = pos_enc_xyz,
            pos_enc_dir      = pos_enc_dir,
            near             = cfg.near,
            far              = cfg.far,
            n_coarse         = cfg.n_coarse,
            randomize        = True,
            model_fine       = model_fine,
            n_fine           = cfg.n_fine if not cfg.use_tiny else 0,
            white_background = cfg.white_bg,
        )

        # Loss: MSE on coarse + fine outputs
        mse_coarse = torch.mean((out["coarse"]["color_map"] - target) ** 2)
        if "fine" in out:
            mse_fine = torch.mean((out["fine"]["color_map"] - target) ** 2)
            loss = mse_coarse + mse_fine
        else:
            mse_fine = mse_coarse
            loss = mse_coarse

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ── Metrics ───────────────────────────────────────────────
        psnr = -10.0 * torch.log10(mse_fine.detach()).item()
        loss_history.append(loss.item())
        psnr_history.append(psnr)

        # ── Logging ───────────────────────────────────────────────
        if (i + 1) % cfg.log_every == 0:
            tqdm.write(
                f"[{i+1:>6}/{cfg.n_iters}] "
                f"loss={loss.item():.4f}  "
                f"psnr={psnr:.2f}dB  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # ── Test render ───────────────────────────────────────────
        if (i + 1) % cfg.render_every == 0:
            render_test_view(
                model_coarse, model_fine,
                pos_enc_xyz, pos_enc_dir,
                test_pose, H, W, focal, cfg,
                step=i+1, out_dir=cfg.out_dir,
            )

        # ── Checkpoint ────────────────────────────────────────────
        if (i + 1) % cfg.save_every == 0:
            save_checkpoint(
                path         = os.path.join(ckpt_dir, f"ckpt_{i+1:05d}.pt"),
                step         = i + 1,
                model_coarse = model_coarse,
                model_fine   = model_fine,
                optimizer    = optimizer,
            )

    # ── Training curves ───────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(loss_history); ax1.set_title("MSE Loss"); ax1.set_xlabel("Iteration")
    ax2.plot(psnr_history); ax2.set_title("PSNR (dB)"); ax2.set_xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "training_curves.png"))
    print(f"\nDone. Outputs saved to: {cfg.out_dir}/")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cfg = get_config()
    print("\n── Config ───────────────────────────────")
    for k, v in vars(cfg).items():
        print(f"  {k:<22} {v}")
    print("─────────────────────────────────────────\n")
    train(cfg)