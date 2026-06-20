"""
Plot pixel-level ROC and PR curves for the final segmentation model.

Pixel-level means:
    each pixel is treated as a binary sample;
    GT mask pixel = 0/1;
    model output probability = score.

Output:
    visualization/outputs/ours_pixel_roc_curve.{png,pdf,svg}
    visualization/outputs/ours_pixel_pr_curve.{png,pdf,svg}
    visualization/outputs/ours_pixel_roc_points.csv
    visualization/outputs/ours_pixel_pr_points.csv
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

from viz_utils import (
    ensure_dir, read_split_csv, load_image_and_mask, load_ours_model,
    sigmoid_prob, resize_prob_to_mask_size, save_figure_all_formats,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", default="preprocess/test.csv")
    parser.add_argument("--weights", default="results/weights/best_our_model.pth")
    parser.add_argument("--out_dir", default="visualization/outputs")
    parser.add_argument("--input_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_pixels", type=int, default=2000000, help="Randomly subsample pixels to save memory; set 0 to use all.")
    parser.add_argument("--use_lee", action="store_true")
    parser.add_argument("--use_clahe", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = load_ours_model(args.weights, device=device)
    df, img_col, mask_col = read_split_csv(args.csv_file)

    y_true_parts = []
    y_score_parts = []
    with torch.no_grad():
        for i, row in df.iterrows():
            data = load_image_and_mask(row[img_col], row[mask_col], tuple(args.input_size), args.use_lee, args.use_clahe)
            outputs = model(data["tensor"].to(device))
            final_logits = outputs[-1]
            prob = sigmoid_prob(final_logits)
            prob_np = resize_prob_to_mask_size(prob, data["mask_resized"].shape)
            y_true_parts.append(data["mask_resized"].reshape(-1).astype(np.uint8))
            y_score_parts.append(prob_np.reshape(-1).astype(np.float32))
            if (i + 1) % 20 == 0:
                print(f"Processed {i+1}/{len(df)} images")

    y_true = np.concatenate(y_true_parts)
    y_score = np.concatenate(y_score_parts)

    if args.max_pixels and len(y_true) > args.max_pixels:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_true), size=args.max_pixels, replace=False)
        y_true = y_true[idx]
        y_score = y_score[idx]
        print(f"Subsampled to {args.max_pixels} pixels for ROC/PR curves")

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    out_dir = Path(args.out_dir)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": np.r_[roc_thresholds]}).to_csv(out_dir / "ours_pixel_roc_points.csv", index=False)
    # pr_thresholds is one item shorter than precision/recall.
    pr_th = np.r_[pr_thresholds, np.nan]
    pd.DataFrame({"precision": precision, "recall": recall, "threshold": pr_th}).to_csv(out_dir / "ours_pixel_pr_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(fpr, tpr, linewidth=2, label=f"Ours AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Pixel-level ROC Curve")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure_all_formats(fig, out_dir / "ours_pixel_roc_curve")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(recall, precision, linewidth=2, label=f"Ours AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Pixel-level Precision-Recall Curve")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure_all_formats(fig, out_dir / "ours_pixel_pr_curve")
    plt.close(fig)

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"Saved ROC/PR outputs to: {out_dir}")


if __name__ == "__main__":
    main()
