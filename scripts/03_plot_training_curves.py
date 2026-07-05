"""
Plot baseline training curves.

Input:
    results/logs/unet_training_log.csv
    results/logs/attention_unet_training_log.csv
    results/logs/deeplabv3plus_training_log.csv
Optional:
    results/logs/ours_training_log.csv, if available later.

Output:
    visualization/outputs/training_loss_curve.{png,pdf,svg}
    visualization/outputs/validation_dice_curve.{png,pdf,svg}
    visualization/outputs/validation_iou_curve.{png,pdf,svg}
"""
from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from viz_utils import ensure_dir, save_figure_all_formats


def read_log(path):
    """Read training log with or without header."""
    path = Path(path)
    df = pd.read_csv(path)

    # If pandas treated first numeric row as header, reload without header.
    first_col = str(df.columns[0]).lower()
    if first_col not in {"epoch", "epochs"}:
        df = pd.read_csv(path, header=None)
        # Your baseline log rows look like:
        # epoch, train_loss, val_dice, val_iou, val_accuracy, val_sensitivity, val_specificity, val_hd95
        names = ["epoch", "train_loss", "val_dice", "val_iou", "val_accuracy", "val_sensitivity", "val_specificity", "val_hd95"]
        df.columns = names[: len(df.columns)]
    else:
        # Normalize possible column names.
        df = df.rename(columns={
            "dice": "val_dice",
            "iou": "val_iou",
            "loss": "train_loss",
            "hd95": "val_hd95",
        })
    return df


def plot_metric(log_items, metric, ylabel, out_base):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, df in log_items:
        if metric not in df.columns:
            print(f"[Skip] {label}: metric '{metric}' not found in columns {list(df.columns)}")
            continue
        ax.plot(df["epoch"], df[metric], linewidth=2, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    ax.set_title(ylabel)
    fig.tight_layout()
    save_figure_all_formats(fig, out_base)
    plt.close(fig)
    print(f"Saved: {out_base}.png/.pdf/.svg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="results/logs")
    parser.add_argument("--out_dir", default="visualization/outputs")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    candidates = [
        ("U-Net", Path(args.log_dir) / "unet_training_log.csv"),
        ("Attention U-Net", Path(args.log_dir) / "attention_unet_training_log.csv"),
        ("DeepLabV3+", Path(args.log_dir) / "deeplabv3plus_training_log.csv"),
        ("Ours", Path(args.log_dir) / "ours_training_log.csv"),
        ("Ours", Path(args.log_dir) / "our_training_log.csv"),
    ]

    logs = []
    used_labels = set()
    for label, path in candidates:
        if path.exists() and label not in used_labels:
            logs.append((label, read_log(path)))
            used_labels.add(label)
            print(f"Loaded {label}: {path}")

    if not logs:
        raise FileNotFoundError(f"No training logs found in {args.log_dir}")

    out_dir = Path(args.out_dir)
    plot_metric(logs, "train_loss", "Training Loss", out_dir / "training_loss_curve")
    plot_metric(logs, "val_dice", "Validation Dice", out_dir / "validation_dice_curve")
    plot_metric(logs, "val_iou", "Validation IoU", out_dir / "validation_iou_curve")
    if any("val_hd95" in df.columns for _, df in logs):
        plot_metric(logs, "val_hd95", "Validation HD95", out_dir / "validation_hd95_curve")


if __name__ == "__main__":
    main()
