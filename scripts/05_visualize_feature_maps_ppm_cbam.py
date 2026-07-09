"""
Visualize PPM output and CBAM-refined feature maps of OurBreastCancerNet.

This uses forward hooks:
    model.ppm      -> PPM feature map
    model.cbam_ppm -> CBAM-refined feature map

Output:
    outputs/visualization/outputs/ppm_cbam_feature_maps.{png,pdf,svg}
"""

# AUTO PATH FIX FOR FINAL GITHUB STRUCTURE
from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
for _p in [_PROJECT_ROOT, _PROJECT_ROOT / "src"]:
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# END AUTO PATH FIX
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.viz_utils import (
    ensure_dir, read_split_csv, load_image_and_mask, load_ours_model,
    normalize_map, save_figure_all_formats,
)


def feature_to_heatmap(feat):
    """Convert [B,C,H,W] feature tensor to a normalized 2D heatmap."""
    fmap = feat.detach().cpu()[0]
    heat = fmap.abs().mean(dim=0).numpy()
    return normalize_map(heat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", default="src/data/test.csv")
    parser.add_argument("--weights", default="outputs/results/weights/best_our_model.pth")
    parser.add_argument("--out_dir", default="outputs/visualization/outputs")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--input_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use_lee", action="store_true")
    parser.add_argument("--use_clahe", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = load_ours_model(args.weights, device=device)

    activations = {}
    def save_hook(name):
        def hook(module, inp, out):
            activations[name] = out
        return hook

    # These module names are confirmed by models/ours.py.
    h1 = model.ppm.register_forward_hook(save_hook("PPM output"))
    h2 = model.cbam_ppm.register_forward_hook(save_hook("After CBAM"))

    df, img_col, mask_col = read_split_csv(args.csv_file)
    indices = np.linspace(0, len(df) - 1, min(args.num_samples, len(df)), dtype=int)
    rows = []

    with torch.no_grad():
        for idx in indices:
            activations.clear()
            row = df.iloc[idx]
            data = load_image_and_mask(row[img_col], row[mask_col], tuple(args.input_size), args.use_lee, args.use_clahe)
            _ = model(data["tensor"].to(device))
            ppm_heat = feature_to_heatmap(activations["PPM output"])
            cbam_heat = feature_to_heatmap(activations["After CBAM"])
            rows.append((data["processed_resized_gray"], ppm_heat, cbam_heat, data["mask_resized"]))

    h1.remove(); h2.remove()

    titles = ["Original", "PPM Feature Map", "CBAM-refined Feature Map", "Ground Truth"]
    n_rows, n_cols = len(rows), len(titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = axes[r, c]
            if c == 0:
                ax.imshow(img, cmap="gray")
            elif c == 3:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img, cmap="magma", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=11)

    fig.tight_layout()
    out_base = Path(args.out_dir) / "ppm_cbam_feature_maps"
    save_figure_all_formats(fig, out_base)
    plt.close(fig)
    print(f"Saved: {out_base}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
