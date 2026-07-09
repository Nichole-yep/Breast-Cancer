
# AUTO PATH FIX FOR FINAL GITHUB STRUCTURE
from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
for _p in [_PROJECT_ROOT, _PROJECT_ROOT / "src"]:
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# END AUTO PATH FIX
# visualization/07_plot_ours_training_curves.py
# -*- coding: utf-8 -*-

"""
Plot training curves for DBDS-Net / Our model from our_model_training_log.csv.

Expected CSV columns:
    epoch
    train_loss
    val_dice
    val_iou
    val_hd95_mean

Example:
    python visualization/07_plot_ours_training_curves.py --csv outputs/results/logs/our_model_training_log.csv

Outputs:
    outputs/visualization/outputs/ours_training_loss_curve.png/.pdf/.svg
    outputs/visualization/outputs/ours_validation_dice_curve.png/.pdf/.svg
    outputs/visualization/outputs/ours_validation_iou_curve.png/.pdf/.svg
    outputs/visualization/outputs/ours_validation_hd95_curve.png/.pdf/.svg
    outputs/visualization/outputs/ours_training_curves_panel.png/.pdf/.svg
"""

from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_figure_all_formats(fig, out_base):
    """
    Save figure as PNG, PDF and SVG.
    """
    out_base = Path(out_base)
    fig.savefig(str(out_base) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", bbox_inches="tight")
    print(f"Saved: {out_base}.png/.pdf/.svg")


def find_column(df, candidates):
    """
    Find a column from several possible names.
    This makes the script robust to small column-name differences.
    """
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Cannot find required column. Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def plot_single_curve(epochs, values, title, ylabel, out_base, best_mode=None):
    """
    best_mode:
        'max' for Dice/IoU
        'min' for Loss/HD95
        None for no best marker
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(epochs, values, linewidth=2.2, label=title)

    if best_mode == "max":
        idx = values.idxmax()
        best_epoch = int(epochs.loc[idx])
        best_value = float(values.loc[idx])
        ax.scatter([best_epoch], [best_value], s=55, zorder=3)
        ax.axhline(best_value, linestyle="--", linewidth=1.3, alpha=0.8)
        ax.annotate(
            f"Best = {best_value:.4f}\nEpoch = {best_epoch}",
            xy=(best_epoch, best_value),
            xytext=(8, -28),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
        )
        ax.legend([f"Best: {best_value:.4f}"], frameon=False)

    elif best_mode == "min":
        idx = values.idxmin()
        best_epoch = int(epochs.loc[idx])
        best_value = float(values.loc[idx])
        ax.scatter([best_epoch], [best_value], s=55, zorder=3)
        ax.axhline(best_value, linestyle="--", linewidth=1.3, alpha=0.8)
        ax.annotate(
            f"Best = {best_value:.4f}\nEpoch = {best_epoch}",
            xy=(best_epoch, best_value),
            xytext=(8, 16),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
        )
        ax.legend([f"Best: {best_value:.4f}"], frameon=False)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure_all_formats(fig, out_base)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="outputs/results/logs/our_model_training_log.csv",
        help="Path to our model training log CSV."
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/visualization/outputs",
        help="Directory to save output figures."
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cannot find CSV file: {csv_path}\n"
            "Please put our_model_training_log.csv into outputs/results/logs/ "
            "or pass its path with --csv."
        )

    df = pd.read_csv(csv_path)
    print(f"Loaded: {csv_path}")
    print("Columns:", list(df.columns))

    epoch_col = find_column(df, ["epoch", "Epoch"])
    loss_col = find_column(df, ["train_loss", "loss", "Train Loss", "training_loss"])
    dice_col = find_column(df, ["val_dice", "dice", "Val Dice", "validation_dice"])
    iou_col = find_column(df, ["val_iou", "iou", "Val IoU", "validation_iou"])
    hd95_col = find_column(df, ["val_hd95_mean", "val_hd95", "hd95", "HD95", "validation_hd95"])

    for col in [epoch_col, loss_col, dice_col, iou_col, hd95_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[epoch_col, loss_col, dice_col, iou_col, hd95_col])
    df = df.sort_values(epoch_col).reset_index(drop=True)

    epochs = df[epoch_col]
    train_loss = df[loss_col]
    val_dice = df[dice_col]
    val_iou = df[iou_col]
    val_hd95 = df[hd95_col]

    # Single metric figures
    plot_single_curve(
        epochs,
        train_loss,
        title="DBDS-Net Training Loss Curve",
        ylabel="Training Loss",
        out_base=out_dir / "ours_training_loss_curve",
        best_mode="min",
    )

    plot_single_curve(
        epochs,
        val_dice,
        title="DBDS-Net Validation Dice Curve",
        ylabel="Dice Coefficient",
        out_base=out_dir / "ours_validation_dice_curve",
        best_mode="max",
    )

    plot_single_curve(
        epochs,
        val_iou,
        title="DBDS-Net Validation IoU Curve",
        ylabel="IoU",
        out_base=out_dir / "ours_validation_iou_curve",
        best_mode="max",
    )

    plot_single_curve(
        epochs,
        val_hd95,
        title="DBDS-Net Validation HD95 Curve",
        ylabel="HD95 (pixels)",
        out_base=out_dir / "ours_validation_hd95_curve",
        best_mode="min",
    )

    # Combined 2 x 2 panel
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    panels = [
        (axes[0, 0], train_loss, "Training Loss Curve", "Training Loss", "min"),
        (axes[0, 1], val_dice, "Validation Dice Curve", "Dice Coefficient", "max"),
        (axes[1, 0], val_iou, "Validation IoU Curve", "IoU", "max"),
        (axes[1, 1], val_hd95, "Validation HD95 Curve", "HD95 (pixels)", "min"),
    ]

    for ax, values, title, ylabel, mode in panels:
        ax.plot(epochs, values, linewidth=2.0)
        if mode == "max":
            idx = values.idxmax()
        else:
            idx = values.idxmin()

        best_epoch = int(epochs.loc[idx])
        best_value = float(values.loc[idx])

        ax.scatter([best_epoch], [best_value], s=45, zorder=3)
        ax.axhline(best_value, linestyle="--", linewidth=1.1, alpha=0.75)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend([f"Best: {best_value:.4f} @ Epoch {best_epoch}"], frameon=False)

    fig.tight_layout()
    save_figure_all_formats(fig, out_dir / "ours_training_curves_panel")
    plt.close(fig)

    best_loss_idx = train_loss.idxmin()
    best_dice_idx = val_dice.idxmax()
    best_iou_idx = val_iou.idxmax()
    best_hd95_idx = val_hd95.idxmin()

    print("\nSummary:")
    print(f"Best Loss: {train_loss.loc[best_loss_idx]:.4f} @ Epoch {int(epochs.loc[best_loss_idx])}")
    print(f"Best Dice: {val_dice.loc[best_dice_idx]:.4f} @ Epoch {int(epochs.loc[best_dice_idx])}")
    print(f"Best IoU : {val_iou.loc[best_iou_idx]:.4f} @ Epoch {int(epochs.loc[best_iou_idx])}")
    print(f"Best HD95: {val_hd95.loc[best_hd95_idx]:.4f} @ Epoch {int(epochs.loc[best_hd95_idx])}")
    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
