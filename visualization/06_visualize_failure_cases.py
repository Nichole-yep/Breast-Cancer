"""
Find and visualize typical failure cases based on per-image Dice.

Columns:
    Original | Ground Truth | Prediction | Boundary Overlay

Output:
    visualization/outputs/failure_cases.{png,pdf,svg}
    visualization/outputs/failure_cases_scores.csv
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from viz_utils import (
    ensure_dir, read_split_csv, load_image_and_mask, load_ours_model,
    sigmoid_prob, resize_prob_to_mask_size, make_boundary_overlay, dice_score,
    save_figure_all_formats,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", default="preprocess/test.csv")
    parser.add_argument("--weights", default="results/weights/best_our_model.pth")
    parser.add_argument("--out_dir", default="visualization/outputs")
    parser.add_argument("--num_cases", type=int, default=4)
    parser.add_argument("--input_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_lee", action="store_true")
    parser.add_argument("--use_clahe", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = load_ours_model(args.weights, device=device)
    df, img_col, mask_col = read_split_csv(args.csv_file)

    cases = []
    with torch.no_grad():
        for i, row in df.iterrows():
            data = load_image_and_mask(row[img_col], row[mask_col], tuple(args.input_size), args.use_lee, args.use_clahe)
            outputs = model(data["tensor"].to(device))
            prob = sigmoid_prob(outputs[-1])
            prob_np = resize_prob_to_mask_size(prob, data["mask_resized"].shape)
            pred = (prob_np > args.threshold).astype(np.uint8)
            dsc = dice_score(data["mask_resized"], pred)
            cases.append({
                "index": i,
                "img_path": row[img_col],
                "mask_path": row[mask_col],
                "dice": dsc,
                "original": data["processed_resized_gray"],
                "gt": data["mask_resized"],
                "pred": pred,
                "overlay": make_boundary_overlay(data["processed_resized_gray"], data["mask_resized"], pred),
            })

    score_df = pd.DataFrame([{k: c[k] for k in ["index", "img_path", "mask_path", "dice"]} for c in cases])
    score_df = score_df.sort_values("dice", ascending=True)
    score_df.to_csv(Path(args.out_dir) / "failure_cases_scores.csv", index=False)

    selected = [cases[int(i)] for i in score_df.head(args.num_cases)["index"].values]
    titles = ["Original", "Ground Truth", "Ours Prediction", "Boundary Overlay\nRed=GT, Green=Pred"]
    fig, axes = plt.subplots(len(selected), len(titles), figsize=(3.1 * len(titles), 3.0 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, casedata in enumerate(selected):
        imgs = [casedata["original"], casedata["gt"], casedata["pred"], casedata["overlay"]]
        cmaps = ["gray", "gray", "gray", None]
        for c in range(len(titles)):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap=cmaps[c], vmin=0 if c in [1, 2] else None, vmax=1 if c in [1, 2] else None)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(f"Case {r+1}\nDice={casedata['dice']:.3f}", fontsize=10)

    fig.tight_layout()
    out_base = Path(args.out_dir) / "failure_cases"
    save_figure_all_formats(fig, out_base)
    plt.close(fig)
    print(f"Saved: {out_base}.png/.pdf/.svg")
    print(f"Saved scores: {Path(args.out_dir) / 'failure_cases_scores.csv'}")


if __name__ == "__main__":
    main()
