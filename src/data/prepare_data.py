import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent
BUSI_ROOT = DATA_DIR / "Dataset_BUSI_with_GT"
if not (BUSI_ROOT / "benign").exists() and (BUSI_ROOT / "Dataset_BUSI_with_GT" / "benign").exists():
    BUSI_ROOT = BUSI_ROOT / "Dataset_BUSI_with_GT"


def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return cv2.imread(str(path), flags)


def imwrite_unicode(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, image)
    if ok:
        buf.tofile(str(path))
    return ok


def merge_masks(mask_paths):
    merged_masks = None
    for p in mask_paths:
        mask = imread_unicode(p, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        binary = (mask > 127).astype(np.uint8)
        if merged_masks is None:
            merged_masks = binary
        else:
            merged_masks = np.logical_or(merged_masks, binary).astype(np.uint8)
    if merged_masks is None:
        return None
    return merged_masks * 255


def rel_to_data_dir(path):
    """Save portable CSV paths relative to src/data."""
    return Path(path).resolve().relative_to(DATA_DIR.resolve()).as_posix()


def collect_samples(root_path=BUSI_ROOT):
    root_path = Path(root_path)
    samples = []
    for category in ["benign", "malignant"]:
        dir_path = root_path / category
        if not dir_path.is_dir():
            print(f"skip missing dir: {dir_path}")
            continue
        all_files = os.listdir(dir_path)
        imgs = [f for f in all_files if f.endswith(".png") and "_mask" not in f]
        for f in imgs:
            img_path = dir_path / f
            basename = f[:-4]
            mask_files = [m for m in all_files if m.startswith(basename + "_mask") and m.endswith(".png")]
            mask_files.sort()
            if len(mask_files) == 0:
                print(f"no mask for {f}")
                continue
            if len(mask_files) == 1:
                mask_path = dir_path / mask_files[0]
            else:
                all_mask_paths = [dir_path / m for m in mask_files]
                merge_mask = merge_masks(all_mask_paths)
                if merge_mask is None:
                    continue
                mask_path = dir_path / f"{basename}merged_mask.png"
                imwrite_unicode(mask_path, merge_mask)
            samples.append((rel_to_data_dir(img_path), rel_to_data_dir(mask_path)))
    return samples


def save_csv(data, filename):
    out_path = DATA_DIR / filename
    df = pd.DataFrame(data, columns=["img_path", "mask_path"])
    df.to_csv(out_path, index=False)
    print(f"saved {len(data)} samples to {out_path}")


if __name__ == "__main__":
    print(f"BUSI root: {BUSI_ROOT}")
    samples = collect_samples(BUSI_ROOT)
    print(f"total samples: {len(samples)}")

    train_and_val, test = train_test_split(samples, test_size=0.15, random_state=42)
    train, val = train_test_split(train_and_val, test_size=0.15 / 0.85, random_state=42)

    save_csv(train, "train.csv")
    save_csv(val, "val.csv")
    save_csv(test, "test.csv")
