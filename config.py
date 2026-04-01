"""
config.py — Hyperparameters & Configuration
All tunable settings in one place.

How to use:
    from config import get_config
    cfg = get_config()
    print(cfg.lr)

To override from command line:
    python train.py --lr 1e-3 --n_iters 10000
"""

import argparse


def get_config() -> argparse.Namespace:
    """
    Returns config as a Namespace object so you can
    access values like cfg.lr instead of cfg["lr"].
    All values can be overridden from the command line.
    """
    parser = argparse.ArgumentParser(description="NeRF Training Config")

    # ── Data ──────────────────────────────────────────────────────
    parser.add_argument("--data_path",  type=str,   default="tiny_nerf_data.npz",
                        help="Path to dataset (.npz for TinyNeRF)")
    parser.add_argument("--n_train",    type=int,   default=100,
                        help="Number of training images to use")
    parser.add_argument("--test_idx",   type=int,   default=101,
                        help="Index of test image for evaluation")

    # ── Positional Encoding ────────────────────────────────────────
    parser.add_argument("--num_freqs_xyz", type=int, default=10,
                        help="Frequency bands for xyz encoding (L in paper)")
    parser.add_argument("--num_freqs_dir", type=int, default=4,
                        help="Frequency bands for direction encoding")

    # ── Model ─────────────────────────────────────────────────────
    parser.add_argument("--hidden_dim", type=int,   default=256,
                        help="Hidden layer width (256 in paper)")
    parser.add_argument("--num_layers", type=int,   default=8,
                        help="Number of layers in geometry network (8 in paper)")
    parser.add_argument("--skip_layer", type=int,   default=5,
                        help="Layer index to inject skip connection (5 in paper)")
    parser.add_argument("--use_tiny",   action="store_true",
                        help="Use TinyNeRFMLP instead of full NeRFMLP")

    # ── Rendering ─────────────────────────────────────────────────
    parser.add_argument("--near",       type=float, default=2.0,
                        help="Near bound for ray sampling")
    parser.add_argument("--far",        type=float, default=6.0,
                        help="Far bound for ray sampling")
    parser.add_argument("--n_coarse",   type=int,   default=64,
                        help="Coarse samples per ray")
    parser.add_argument("--n_fine",     type=int,   default=128,
                        help="Fine samples per ray (0 to disable fine pass)")
    parser.add_argument("--white_bg",   action="store_true",
                        help="Composite renders against white background")

    # ── Training ──────────────────────────────────────────────────
    parser.add_argument("--n_iters",    type=int,   default=10000,
                        help="Total training iterations")
    parser.add_argument("--batch_size", type=int,   default=1024,
                        help="Number of rays per batch")
    parser.add_argument("--lr",         type=float, default=5e-4,
                        help="Initial learning rate")
    parser.add_argument("--lr_decay",   type=float, default=0.9998,
                        help="Exponential LR decay per iteration")

    # ── Logging & Saving ──────────────────────────────────────────
    parser.add_argument("--log_every",    type=int, default=100,
                        help="Print loss every N iterations")
    parser.add_argument("--render_every", type=int, default=500,
                        help="Save test render every N iterations")
    parser.add_argument("--save_every",   type=int, default=1000,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--out_dir",      type=str, default="outputs",
                        help="Directory for renders and checkpoints")
    parser.add_argument("--resume",       type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


if __name__ == "__main__":
    # Print all config values when run directly
    cfg = get_config()
    print("\n── NeRF Config ──────────────────────────")
    for k, v in vars(cfg).items():
        print(f"  {k:<20} {v}")
    print("─────────────────────────────────────────\n")