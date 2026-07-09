# 用于 BUSI 数据预处理与数据加载
# 支持最终 GitHub 目录结构：src/data/train.csv + src/data/Dataset_BUSI_with_GT
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import cv2
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path(__file__).resolve().parent
BUSI_ROOT = DATA_DIR / "Dataset_BUSI_with_GT"
if not (BUSI_ROOT / "benign").exists() and (BUSI_ROOT / "Dataset_BUSI_with_GT" / "benign").exists():
    BUSI_ROOT = BUSI_ROOT / "Dataset_BUSI_with_GT"


def resolve_data_path(path):
    """Resolve CSV paths after folders were moved.

    The CSV may contain portable paths such as:
        Dataset_BUSI_with_GT/benign/benign (1).png
    or absolute Windows paths. This function searches the final GitHub data folder first.
    """
    raw = str(path).strip().replace("\\", "/")
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p

    candidates = []
    # Most common final format: Dataset_BUSI_with_GT/...
    if raw.startswith("Dataset_BUSI_with_GT/"):
        candidates.append(DATA_DIR / raw)
        candidates.append(DATA_DIR / "Dataset_BUSI_with_GT" / raw)
        candidates.append(PROJECT_ROOT / raw)
    # If the path was saved as src/data/...
    candidates.append(PROJECT_ROOT / raw)
    candidates.append(DATA_DIR / raw)
    # If only category/filename was saved.
    candidates.append(BUSI_ROOT / raw)

    for c in candidates:
        if c.exists():
            return c
    # Return the most likely path for a clear error message.
    return candidates[0] if candidates else p


def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    """cv2.imread replacement that is more stable with Chinese/space paths on Windows."""
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return cv2.imread(str(path), flags)


class BUSIDataset(Dataset):
    def __init__(self, csv_file, ues_lee=True, ues_clahe=True, augment=True):
        csv_file = resolve_data_path(csv_file)
        self.df = pd.read_csv(csv_file)
        self.ues_lee = ues_lee
        self.ues_clahe = ues_clahe
        self.augment = augment

        if ues_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

        self.transform = self._get_transform(augment)

    def _get_transform(self, augment):
        base_transform = [
            A.Resize(256, 256),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
        if augment:
            aug_transform = [
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-45, 45), p=0.6),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(std_range=(10.0 / 255.0, 50.0 / 255.0), p=0.4),
            ]
            transforms = aug_transform + base_transform
        else:
            transforms = base_transform
        return A.Compose(transforms, additional_targets={"edge_mask": "mask"})

    def _lee_filter(self, img, window=7):
        img_float = img.astype(np.float64)
        mean = cv2.blur(img_float, (window, window))
        mean_sq = cv2.blur(img_float ** 2, (window, window))
        var = mean_sq - mean ** 2
        noise_var = np.var(img_float - mean)
        k = var / (var + noise_var + 1e-8)
        k = np.clip(k, 0.0, 1.0)
        out = mean + k * (img_float - mean)
        return np.clip(out, 0, 255).astype(np.uint8)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = resolve_data_path(row["img_path"])
        mask_path = resolve_data_path(row["mask_path"])

        img = imread_unicode(img_path, cv2.IMREAD_GRAYSCALE)
        mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise FileNotFoundError(f"无法读取图像/mask: {img_path} | {mask_path}")

        if self.ues_lee:
            img = self._lee_filter(img, window=7)
        if self.ues_clahe:
            img = self.clahe.apply(img)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        mask = (mask > 127).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.dilate(mask, kernel, iterations=1)
        erode = cv2.erode(mask, kernel, iterations=1)
        edge = (dilate - erode).astype(np.uint8)

        augmented = self.transform(image=img, mask=mask, edge_mask=edge)
        img_tensor = augmented["image"]
        mask_tensor = augmented["mask"].unsqueeze(0).float()
        edge_mask_tensor = augmented["edge_mask"].unsqueeze(0).float()
        return img_tensor, mask_tensor, edge_mask_tensor


def get_loaders(train_csv, val_csv, test_csv, batch_size=2, use_lee=True, use_clahe=True):
    train_data = BUSIDataset(train_csv, ues_lee=use_lee, ues_clahe=use_clahe, augment=True)
    val_data = BUSIDataset(val_csv, ues_lee=use_lee, ues_clahe=use_clahe, augment=False)
    test_data = BUSIDataset(test_csv, ues_lee=use_lee, ues_clahe=use_clahe, augment=False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders(
        "src/data/train.csv", "src/data/val.csv", "src/data/test.csv",
        batch_size=2,
        use_lee=True,
        use_clahe=True,
    )
    for images, masks, edges in train_loader:
        print("图像张量形状:", images.shape)
        print("Mask 形状:", masks.shape)
        print("Edge 形状:", edges.shape)
        print("图像值范围:", images.min().item(), images.max().item())
        break
