import numpy as np
import cv2
import pandas as pd
import os

# ---------- 预处理函数（复用自 BUSIDataset）----------
def lee_filter(img, window=7):
    """Lee 散斑滤波"""
    img_float = img.astype(np.float64)
    mean = cv2.blur(img_float, (window, window))
    mean_sq = cv2.blur(img_float ** 2, (window, window))
    var = mean_sq - mean ** 2
    noise_var = np.var(img_float - mean)
    k = var / (var + noise_var + 1e-8)
    k = np.clip(k, 0.0, 1.0)
    out = mean + k * (img_float - mean)
    return np.clip(out, 0, 255).astype(np.uint8)

def clahe_enhance(img, clip_limit=4.0, tile_size=(8,8)):
    """CLAHE 对比度受限自适应直方图均衡"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(img)

def preprocess_image(img_path, use_lee=True, use_clahe=True):
    """
    读取灰度图，依次进行 Lee 滤波和 CLAHE 增强（可选）
    返回预处理后的 uint8 灰度图
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{img_path}")
    if use_lee:
        img = lee_filter(img, window=7)
    if use_clahe:
        img = clahe_enhance(img)
    return img

# ---------- 流程1：Otsu + 形态学开运算 ----------
def segment_otsu(preprocessed_img):
    """Otsu 二值化 + 形态学开运算去除小噪点"""
    # Otsu 阈值
    _, binary = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 形态学开运算（先腐蚀后膨胀）
    kernel = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened

# ---------- 流程2：区域生长 ----------
def auto_seed_point(img, grid_size=5):
    """
    自动选择种子点：将图像划分为 grid_size x grid_size 的网格，
    选取平均灰度最高的网格的中心点作为种子（因为肿瘤通常较亮）
    """
    h, w = img.shape
    step_h = h // grid_size
    step_w = w // grid_size
    max_mean = -1
    seed = (h//2, w//2)  # 默认中心
    for i in range(grid_size):
        for j in range(grid_size):
            y1 = i * step_h
            y2 = (i+1) * step_h if i < grid_size-1 else h
            x1 = j * step_w
            x2 = (j+1) * step_w if j < grid_size-1 else w
            roi = img[y1:y2, x1:x2]
            mean_val = np.mean(roi)
            if mean_val > max_mean:
                max_mean = mean_val
                cy = (y1 + y2) // 2
                cx = (x1 + x2) // 2
                seed = (cy, cx)
    return seed

def region_growing(img, seed=None, threshold=10):
    """
    区域生长算法
    :param img:  灰度图 (uint8)
    :param seed: 种子点 (y, x)，若为 None 则自动选择
    :param threshold: 灰度差阈值
    :return: 二值掩模 (uint8, 0/255)
    """
    if seed is None:
        seed = auto_seed_point(img)
    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)
    # 使用栈实现
    stack = [seed]
    visited[seed] = True
    seed_val = int(img[seed])
    while stack:
        y, x = stack.pop()
        mask[y, x] = 255
        # 检查四邻域
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                if abs(int(img[ny, nx]) - seed_val) <= threshold:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
    return mask

# ---------- 批处理主函数 ----------
def process_and_save(csv_path, output_dir_otsu, output_dir_rg, use_lee=True, use_clahe=True):
    """
    从 CSV 中读取图片路径，分别用两种方法分割并保存结果
    CSV 需包含列 'img_path'（图像路径）和 'mask_path'（参考mask，这里仅用于命名）
    """
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir_otsu, exist_ok=True)
    os.makedirs(output_dir_rg, exist_ok=True)

    for idx, row in df.iterrows():
        img_path = row['img_path']
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # 预处理
        processed = preprocess_image(img_path, use_lee, use_clahe)

        # ---- Otsu 分割 ----
        otsu_mask = segment_otsu(processed)
        cv2.imwrite(os.path.join(output_dir_otsu, f"{base_name}_otsu.png"), otsu_mask)

        # ---- 区域生长分割 ----
        # 自动选取种子点（也可手动指定，例如 (100,100)）
        seed = auto_seed_point(processed)
        print(f"图像 {base_name} 自动种子点: {seed}")
        rg_mask = region_growing(processed, seed=seed, threshold=10)
        cv2.imwrite(os.path.join(output_dir_rg, f"{base_name}_rg.png"), rg_mask)

    print("所有分割结果已保存。")

# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 假设 test.csv 包含两列：img_path, mask_path（mask_path 可忽略）
    # 请修改为实际路径
    test_csv = r"D:\Nichole\Breast_cancer\Breast-Cancer-main\Breast-Cancer-main\preprocess\test.csv"
    out_otsu = "results_otsu"
    out_rg   = "results_region_growing"
    process_and_save(test_csv, out_otsu, out_rg, use_lee=True, use_clahe=True)