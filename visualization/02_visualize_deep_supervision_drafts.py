"""
Visualize intermediate deep-supervision drafts of OurBreastCancerNet.

According to models/ours.py, forward() returns 4 prediction maps:
    output[0] = 1/8 draft
    output[1] = 1/4 draft
    output[2] = 1/2 draft
    output[3] = final 1/1 prediction

Output:
    visualization/outputs/deep_supervision_drafts.{png,pdf,svg}
"""
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

from viz_utils import (
    ensure_dir, read_split_csv, load_image_and_mask, load_ours_model,
    sigmoid_prob, resize_prob_to_mask_size, save_figure_all_formats,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", default="preprocess/test.csv")
    parser.add_argument("--weights", default="results/weights/best_our_model.pth")
    parser.add_argument("--out_dir", default="visualization/outputs")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--input_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use_lee", action="store_true")
    parser.add_argument("--use_clahe", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = load_ours_model(args.weights, device=device)
    df, img_col, mask_col = read_split_csv(args.csv_file)
    indices = np.linspace(0, len(df) - 1, min(args.num_samples, len(df)), dtype=int)

    rows = []
    with torch.no_grad():
        for idx in indices:
            row = df.iloc[idx]
            data = load_image_and_mask(row[img_col], row[mask_col], tuple(args.input_size), args.use_lee, args.use_clahe)
            outputs = model(data["tensor"].to(device))
            if not isinstance(outputs, (list, tuple)) or len(outputs) < 4:
                raise RuntimeError("Model does not return at least 4 deep-supervision outputs.")
            probs = [resize_prob_to_mask_size(sigmoid_prob(out), data["mask_resized"].shape) for out in outputs[-4:]]
            rows.append((data["processed_resized_gray"], probs[0], probs[1], probs[2], probs[3], data["mask_resized"]))

    titles = ["Original", "Draft 1\n1/8", "Draft 2\n1/4", "Draft 3\n1/2", "Final Prediction\n1/1", "Ground Truth"]
    n_rows, n_cols = len(rows), len(titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.9 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = axes[r, c]
            if c == 0:
                ax.imshow(img, cmap="gray")
            elif c == n_cols - 1:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img, cmap="magma", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=11)

    fig.tight_layout()
    out_base = Path(args.out_dir) / "deep_supervision_drafts"
    save_figure_all_formats(fig, out_base)
    plt.close(fig)
    print(f"Saved: {out_base}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
