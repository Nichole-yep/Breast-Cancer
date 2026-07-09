"""
Shared utilities for BUSI segmentation visualization.
Put this file in: Breast-Cancer/visualization/viz_utils.py
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
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from src.data.dataset import resolve_data_path, imread_unicode


def add_project_root_to_path():
    """Allow scripts inside visualization/ to import models, preprocess, evaluate modules."""
    current = Path(__file__).resolve()
    project_root = current.parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_split_csv(csv_file):
    """Read a split CSV and normalize possible column names.

    Supported formats:
    - img_path, mask_path  (from preprocess/dataset.py)
    - images, masks        (from evaluate/eval.py)
    """
    df = pd.read_csv(resolve_data_path(csv_file))
    columns = set(df.columns)
    if {"img_path", "mask_path"}.issubset(columns):
        img_col, mask_col = "img_path", "mask_path"
    elif {"images", "masks"}.issubset(columns):
        img_col, mask_col = "images", "masks"
    else:
        raise ValueError(
            f"CSV must contain either ['img_path','mask_path'] or ['images','masks']; got columns: {list(df.columns)}"
        )
    return df, img_col, mask_col


def lee_filter(img, window=7):
    """Simple Lee filter for ultrasound speckle noise reduction.

    This mirrors the idea used in preprocess/dataset.py. It is optional for visualization;
    turn it off if your final evaluation pipeline did not use Lee filtering.
    """
    img_f = img.astype(np.float32)
    mean = cv2.blur(img_f, (window, window))
    mean_sq = cv2.blur(img_f * img_f, (window, window))
    var = mean_sq - mean * mean
    noise_var = np.mean(var)
    k = var / (var + noise_var + 1e-8)
    out = mean + k * (img_f - mean)
    return np.clip(out, 0, 255).astype(np.uint8)


def load_image_and_mask(img_path, mask_path, input_size=(256, 256), use_lee=False, use_clahe=False):
    """Load BUSI image/mask and prepare both display arrays and model input tensor."""
    img_gray = imread_unicode(resolve_data_path(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    mask = imread_unicode(resolve_data_path(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    mask = (mask > 127).astype(np.uint8)

    original_display = img_gray.copy()

    img_proc = img_gray.copy()
    if use_lee:
        img_proc = lee_filter(img_proc)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        img_proc = clahe.apply(img_proc)

    # Resize for model input.
    h, w = input_size
    img_resized = cv2.resize(img_proc, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Convert grayscale to 3-channel RGB and normalize like preprocess/dataset.py: mean=0.5, std=0.5.
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    img_rgb = (img_rgb - 0.5) / 0.5
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0)

    return {
        "original_gray": original_display,
        "processed_resized_gray": img_resized,
        "mask_original": mask,
        "mask_resized": mask_resized,
        "tensor": tensor,
    }


def load_ours_model(weights_path, device="cpu"):
    """Load the final OurBreastCancerNet model."""
    add_project_root_to_path()
    from src.models.ours import OurBreastCancerNet

    model = OurBreastCancerNet(pretrained=False, num_classes=1).to(device)
    state = torch.load(str(resolve_data_path(weights_path)), map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]

    # Support weights saved from DataParallel, where keys start with "module.".
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def sigmoid_prob(logits):
    return torch.sigmoid(logits)


def resize_prob_to_mask_size(prob_tensor, target_shape):
    """Resize a probability tensor [1,1,H,W] or [1,H,W] to target H,W and return numpy."""
    if prob_tensor.dim() == 3:
        prob_tensor = prob_tensor.unsqueeze(0)
    if prob_tensor.dim() == 2:
        prob_tensor = prob_tensor.unsqueeze(0).unsqueeze(0)
    target_h, target_w = target_shape
    resized = F.interpolate(prob_tensor.float(), size=(target_h, target_w), mode="bilinear", align_corners=False)
    return resized.squeeze().detach().cpu().numpy()


def normalize_map(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmin(x)
    denom = np.nanmax(x) + 1e-8
    return x / denom


def mask_boundary(mask, thickness=1):
    """Return binary boundary map of a 0/1 mask."""
    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=thickness)
    boundary = mask - eroded
    return (boundary > 0).astype(np.uint8)


def make_boundary_overlay(gray_img, gt_mask, pred_mask, gt_color=(255, 0, 0), pred_color=(0, 255, 0)):
    """Overlay GT and prediction boundaries on a grayscale image.

    Red = GT boundary, Green = prediction boundary by default.
    """
    if gray_img.ndim == 2:
        overlay = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    else:
        overlay = gray_img.copy()
    gt_mask = cv2.resize((gt_mask > 0).astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_mask = cv2.resize((pred_mask > 0).astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_b = mask_boundary(gt_mask)
    pred_b = mask_boundary(pred_mask)
    overlay[gt_b > 0] = gt_color
    overlay[pred_b > 0] = pred_color
    return overlay


def dice_score(gt, pred):
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)
    inter = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    return (2 * inter / (denom + 1e-8)) if denom > 0 else 1.0


def save_figure_all_formats(fig, out_base, dpi=300):
    """Save a Matplotlib figure as PNG, PDF, and SVG."""
    out_base = Path(out_base)
    ensure_dir(out_base.parent)
    fig.savefig(str(out_base) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", bbox_inches="tight")
