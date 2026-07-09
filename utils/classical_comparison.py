
# AUTO PATH FIX FOR FINAL GITHUB STRUCTURE
from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
for _p in [_PROJECT_ROOT, _PROJECT_ROOT / "src"]:
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# END AUTO PATH FIX
# utils/classical_comparison.py
import os
import sys
import cv2
import numpy as np
import pandas as pd
import glob

# ================================
# 1. 项目根目录与默认CSV路径
# ================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CSV = os.path.join(PROJECT_ROOT, 'src', 'data', 'test.csv')

# ================================
# 2. 预处理函数（与 dataset.py 一致）
# ================================

def lee_filter(img, window=7):
    img_float = img.astype(np.float64)
    mean = cv2.blur(img_float, (window, window))
    mean_sq = cv2.blur(img_float ** 2, (window, window))
    var = mean_sq - mean ** 2
    noise_var = np.var(img_float - mean)
    k = var / (var + noise_var + 1e-8)
    k = np.clip(k, 0.0, 1.0)
    out = mean + k * (img_float - mean)
    return np.clip(out, 0, 255).astype(np.uint8)

def preprocess_image(img_path, use_lee=True, use_clahe=True, target_size=(256, 256)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    if use_lee:
        img = lee_filter(img, window=7)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    return img

# ================================
# 3. 智能路径查找
# ================================
def find_image_path(rel_path, project_root, log_file=None):
    candidates = []
    cand1 = os.path.normpath(os.path.join(project_root, 'src', 'data', rel_path))
    candidates.append(cand1)
    cand2 = os.path.normpath(os.path.join(project_root, rel_path))
    candidates.append(cand2)
    filename = os.path.basename(rel_path)
    search_pattern = os.path.join(project_root, 'src', 'data', 'Dataset_BUSI_with_GT', '**', filename)
    matches = glob.glob(search_pattern, recursive=True)
    if matches:
        candidates.append(matches[0])
    for candidate in candidates:
        if os.path.exists(candidate):
            if log_file:
                log_file.write(f"找到图像: {candidate}\n")
            return candidate
    msg = f"警告: 未找到图像 {rel_path}, 已尝试: {candidates}"
    print(msg)
    if log_file:
        log_file.write(msg + "\n")
    return candidates[0]

# ================================
# 4. Otsu 方法（用于对比，与区域生长无关）
# ================================
def otsu_optimized(img, morph_kernel_size=3):
    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    return bin_img

# ================================
# 5. 独立稳健的区域生长算法（不依赖Otsu）
# ================================
def region_growing_advanced(img,
                            seed_radius=30,          # 种子搜索半径（中心区域）
                            threshold_ratio=1.8,     # 阈值系数（均值±ratio*标准差）
                            min_area=200):           # 最小面积阈值
    """
    纯区域生长算法，种子基于中心区域灰度最低的像素。
    不使用Otsu任何信息，完全独立。
    """
    h, w = img.shape

    # ---- 1. 在图像中心区域选取灰度最低的前5个点作为候选种子 ----
    cx, cy = w // 2, h // 2
    radius = min(seed_radius, w//2, h//2)
    x_min = max(0, cx - radius)
    x_max = min(w, cx + radius)
    y_min = max(0, cy - radius)
    y_max = min(h, cy + radius)
    roi = img[y_min:y_max, x_min:x_max]

    # 若 ROI 太小，则用全图
    if roi.size == 0:
        roi = img
        y_min, x_min = 0, 0

    # 获取 ROI 内灰度最低的前 5 个点坐标
    flat_indices = np.argsort(roi.ravel())[:5]  # 从小到大
    seed_coords = []
    for idx in flat_indices:
        y_local = idx // roi.shape[1]
        x_local = idx % roi.shape[1]
        seed_coords.append((y_local + y_min, x_local + x_min))

    # ---- 2. 生长函数（单阶段动态阈值） ----
    def grow_from_seed(seed):
        sy, sx = seed
        region_vals = [float(img[sy, sx])]
        region_mean = float(img[sy, sx])
        region_std = 0.0
        visited = np.zeros_like(img, dtype=bool)
        visited[sy, sx] = True
        result = np.zeros_like(img, dtype=np.uint8)
        result[sy, sx] = 255
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        stack = [(sy, sx)]

        max_iter = h * w  # 防止死循环
        iter_count = 0

        while stack and iter_count < max_iter:
            iter_count += 1
            y, x = stack.pop()
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w) or visited[ny, nx]:
                    continue
                pv = float(img[ny, nx])
                # 计算动态阈值
                thresh = max(5, int(region_std * threshold_ratio)) if region_std > 0 else 30
                lower = max(0, int(region_mean - thresh))
                upper = min(255, int(region_mean + thresh))
                if lower <= pv <= upper:
                    visited[ny, nx] = True
                    result[ny, nx] = 255
                    region_vals.append(pv)
                    stack.append((ny, nx))
                    # 更新区域统计量
                    region_mean = np.mean(region_vals)
                    region_std = np.std(region_vals) if len(region_vals) > 1 else 0.0
                else:
                    visited[ny, nx] = True  # 标记为已访问，避免重复检查

        return result, len(region_vals)

    # ---- 3. 对每个候选种子生长，选择像素数最多的结果 ----
    best_result = None
    best_count = 0
    for seed in seed_coords:
        res, cnt = grow_from_seed(seed)
        if cnt > best_count:
            best_count = cnt
            best_result = res

    # 若全部失败（极罕见），返回全零
    if best_result is None or best_count == 0:
        print("  警告: 区域生长未产生任何像素，返回全零。")
        return np.zeros_like(img)

    seg_result = best_result

    # ---- 4. 后处理 ----
    # 形态学闭运算填充空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    seg_result = cv2.morphologyEx(seg_result, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 保留最大连通域，并过滤小面积区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg_result, connectivity=8)
    if num_labels >= 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) > 0:
            max_idx = np.argmax(areas) + 1
            if areas[max_idx-1] >= min_area:
                seg_result = (labels == max_idx).astype(np.uint8) * 255
            else:
                # 若最大连通域太小，则保留所有（但面积小可能不理想）
                pass
    # 如果最终结果太小，直接返回原结果（至少不会全黑）
    return seg_result

# ================================
# 6. 主运行函数
# ================================

def run_classical_segmentation(test_csv_path=None,
                               output_otsu='outputs/figures/results_otsu',
                               output_rg='outputs/figures/results_region_growing',
                               use_lee=True, use_clahe=True):
    if test_csv_path is None:
        test_csv_path = DEFAULT_CSV
    if not os.path.exists(test_csv_path):
        print(f"错误：未找到测试集CSV文件：{test_csv_path}")
        print("请先运行 src/data/prepare_data.py 生成测试集划分文件。")
        sys.exit(1)

    output_otsu_abs = os.path.join(PROJECT_ROOT, output_otsu)
    output_rg_abs = os.path.join(PROJECT_ROOT, output_rg)

    os.makedirs(output_otsu_abs, exist_ok=True)
    os.makedirs(output_rg_abs, exist_ok=True)

    print(f"Otsu 结果将保存至: {output_otsu_abs}")
    print(f"区域生长结果将保存至: {output_rg_abs}")

    df = pd.read_csv(test_csv_path)
    print(f"测试集总样本数: {len(df)}")

    log_path = os.path.join(PROJECT_ROOT, 'processing_log.txt')
    with open(log_path, 'w', encoding='utf-8') as log:
        log.write(f"项目根目录: {PROJECT_ROOT}\n")
        log.write(f"CSV文件: {test_csv_path}\n")
        log.write(f"输出目录(绝对路径): {output_otsu_abs}, {output_rg_abs}\n\n")

        success_count = 0
        for idx, row in df.iterrows():
            img_rel_path = row['img_path']
            log.write(f"\n[{idx+1}/{len(df)}] 处理: {img_rel_path}\n")
            img_path = find_image_path(img_rel_path, PROJECT_ROOT, log)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            log.write(f"  最终使用图像路径: {img_path}\n")

            try:
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"文件不存在: {img_path}")

                img_proc = preprocess_image(img_path, use_lee=use_lee, use_clahe=use_clahe)
                otsu_bin = otsu_optimized(img_proc, morph_kernel_size=3)
                rg_bin = region_growing_advanced(img_proc)   # 完全独立

                otsu_out = os.path.join(output_otsu_abs, f"{base_name}_otsu.png")
                rg_out = os.path.join(output_rg_abs, f"{base_name}_rg.png")

                ret_otsu = cv2.imwrite(otsu_out, otsu_bin)
                ret_rg = cv2.imwrite(rg_out, rg_bin)

                if not ret_otsu:
                    raise RuntimeError(f"Otsu 结果写入失败: {otsu_out}")
                if not ret_rg:
                    raise RuntimeError(f"区域生长结果写入失败: {rg_out}")

                success_count += 1
                msg = f"  完成: {base_name} -> 已保存至\n    {otsu_out}\n    {rg_out}"
                print(msg)
                log.write(msg + "\n")

            except Exception as e:
                msg = f"  错误: {e}"
                print(msg)
                log.write(msg + "\n")
                continue

        final_msg = f"\n全部处理完成！成功处理 {success_count}/{len(df)} 张图像。"
        print(final_msg)
        log.write(final_msg + "\n")

if __name__ == "__main__":
    run_classical_segmentation()