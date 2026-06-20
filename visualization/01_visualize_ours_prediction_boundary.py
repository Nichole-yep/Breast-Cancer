"""
Create publication-style prediction panels for the final model.

Columns:
    Original | Ground Truth | Ours Probability | Ours Prediction | Boundary Overlay

Boundary overlay:
    Red   = Ground Truth boundary
    Green = Prediction boundary

Output:
    visualization/outputs/ours_prediction_panel.{png,pdf,svg}
"""
from pathlib import Path
import argparse

import numpy as np
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
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--input_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_lee", action="store_true", help="Use Lee filter before inference")
    parser.add_argument("--use_clahe", action="store_true", help="Use CLAHE before inference")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = load_ours_model(args.weights, device=device)
    df, img_col, mask_col = read_split_csv(args.csv_file)

    # Evenly choose samples from the test split to avoid only showing adjacent images.
    indices = np.linspace(0, len(df) - 1, min(args.num_samples, len(df)), dtype=int)
    panels = []

    with torch.no_grad():
        for idx in indices:
            row = df.iloc[idx]
            data = load_image_and_mask(
                row[img_col], row[mask_col], input_size=tuple(args.input_size),
                use_lee=args.use_lee, use_clahe=args.use_clahe,
            )
            img_tensor = data["tensor"].to(device)
            outputs = model(img_tensor)
            final_logits = outputs[-1]
            prob = sigmoid_prob(final_logits)
            prob_np = resize_prob_to_mask_size(prob, data["mask_resized"].shape)
            pred = (prob_np > args.threshold).astype(np.uint8)
            overlay = make_boundary_overlay(data["processed_resized_gray"], data["mask_resized"], pred)
            dsc = dice_score(data["mask_resized"], pred)
            panels.append((data["processed_resized_gray"], data["mask_resized"], prob_np, pred, overlay, dsc))

    col_titles = ["Original", "Ground Truth", "Ours Probability", "Ours Prediction", "Boundary Overlay\nRed=GT, Green=Pred"]
    n_rows, n_cols = len(panels), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.1 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, panel in enumerate(panels):
        original, gt, prob, pred, overlay, dsc = panel
        imgs = [original, gt, prob, pred, overlay]
        cmaps = ["gray", "gray", "magma", "gray", None]
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap=cmaps[c], vmin=0 if c in [1, 2, 3] else None, vmax=1 if c in [1, 2, 3] else None)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(f"Case {r+1}\nDice={dsc:.3f}", fontsize=10)

    fig.tight_layout()
    out_base = Path(args.out_dir) / "ours_prediction_panel"
    save_figure_all_formats(fig, out_base)
    plt.close(fig)
    print(f"Saved: {out_base}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
