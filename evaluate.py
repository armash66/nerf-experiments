"""
evaluate.py — Cross-Experiment Evaluation and Comparison
Loads CSV training logs from one or more experiments and generates
comparative plots and summary statistics.

Usage:
    # Evaluate a single experiment
    python evaluate.py --experiments experiment_1

    # Compare multiple experiments
    python evaluate.py --experiments experiment_1 experiment_2 experiment_3

    # Custom output path
    python evaluate.py --experiments experiment_1 experiment_2 --output outputs/comparison.png
"""

import os
import csv
import argparse
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# LOG PARSING
# ─────────────────────────────────────────────

def load_training_log(experiment_name: str) -> dict:
    """
    Load a train_log.csv file from an experiment directory.

    Returns:
        dict with keys: iteration, loss, psnr, lr, time (each a list)
    """
    log_path = os.path.join("outputs", experiment_name, "logs", "train_log.csv")
    if not os.path.isfile(log_path):
        print(f"WARNING: Log file not found: {log_path}")
        return None

    data = {"iteration": [], "loss": [], "psnr": [], "lr": [], "time": []}

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["iteration"].append(int(row["iteration"]))
            data["loss"].append(float(row["loss"]))
            data["psnr"].append(float(row["psnr"]))
            data["lr"].append(float(row["lr"]))
            data["time"].append(float(row["time"]))

    return data


# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────

def print_summary(experiments: dict):
    """Print a formatted summary table of final metrics across experiments."""
    print("\n── Experiment Summary ──────────────────────────────────────────")
    print(f"  {'Experiment':<20} {'Final PSNR':>12} {'Final Loss':>12} {'Time (s)':>10} {'Iters':>8}")
    print(f"  {'─' * 20} {'─' * 12} {'─' * 12} {'─' * 10} {'─' * 8}")

    for name, data in experiments.items():
        if data is None:
            print(f"  {name:<20} {'N/A':>12} {'N/A':>12} {'N/A':>10} {'N/A':>8}")
            continue
        final_psnr = data["psnr"][-1] if data["psnr"] else 0.0
        final_loss = data["loss"][-1] if data["loss"] else 0.0
        final_time = data["time"][-1] if data["time"] else 0.0
        final_iter = data["iteration"][-1] if data["iteration"] else 0
        print(
            f"  {name:<20} "
            f"{final_psnr:>10.2f}dB "
            f"{final_loss:>12.6f} "
            f"{final_time:>10.1f} "
            f"{final_iter:>8d}"
        )

    print(f"  {'─' * 62}")


# ─────────────────────────────────────────────
# COMPARATIVE PLOT
# ─────────────────────────────────────────────

def plot_comparison(experiments: dict, output_path: str):
    """Generate side-by-side loss and PSNR comparison plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, data in experiments.items():
        if data is None:
            continue
        ax1.plot(data["iteration"], data["loss"],  label=name, linewidth=1.2)
        ax2.plot(data["iteration"], data["psnr"],  label=name, linewidth=1.2)

    ax1.set_title("MSE Loss Comparison")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("PSNR Comparison (dB)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("PSNR (dB)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nComparison plot saved: {output_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeRF Experiment Evaluation")
    parser.add_argument("--experiments", type=str, nargs="+", required=True,
                        help="List of experiment names to evaluate")
    parser.add_argument("--output",     type=str, default="outputs/comparison.png",
                        help="Output path for comparison plot")
    args = parser.parse_args()

    # Load all logs
    experiments = {}
    for exp_name in args.experiments:
        data = load_training_log(exp_name)
        experiments[exp_name] = data

    # Print summary
    print_summary(experiments)

    # Generate plot (only if at least one experiment has data)
    valid = {k: v for k, v in experiments.items() if v is not None}
    if valid:
        plot_comparison(valid, args.output)
    else:
        print("No valid experiment logs found. Skipping plot generation.")
