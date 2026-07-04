import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

# ===================== 1. 读取数据 =====================
unet_df = pd.read_csv("results/logs/test_unet_per_sample_metrics.csv")
ours_df = pd.read_csv("results/metrics/test_per_sample_metrics.csv") 
deeplabv3plus_df = pd.read_csv("results/logs/test_deeplabv3plus_per_sample_metrics.csv")
attention_df = pd.read_csv("results/logs/test_attention_unet_per_sample_metrics.csv")

# ===================== 2. 数据对齐检查（最关键） =====================
print("=" * 60)
print("🔴 1. 检查数据是否对齐（致命错误检查）")
print("=" * 60)

# 自动寻找 ID 列（支持常见命名）
possible_ids = ['sample_id', 'image_id', 'file_name', 'filename', 'id']
id_col = None
for col in possible_ids:
    if col in unet_df.columns:
        id_col = col
        break

if id_col is None:
    print("❌ 警告：没找到 ID 列（case_id/file_name等）！无法验证对齐。")
    print("   请手动检查 CSV 文件，确保行顺序一致。")
else:
    print(f"✅ 使用列 '{id_col}' 进行对齐检查：")
    # 排序（防止乱序）
    unet_df = unet_df.sort_values(id_col).reset_index(drop=True)
    ours_df = ours_df.sort_values(id_col).reset_index(drop=True)
    deeplabv3plus_df = deeplabv3plus_df.sort_values(id_col).reset_index(drop=True)
    attention_df = attention_df.sort_values(id_col).reset_index(drop=True)
    
    # 验证
    print(f"   UNet vs Ours:      {(unet_df[id_col] == ours_df[id_col]).all()}")
    print(f"   DeepLab vs Ours:    {(deeplabv3plus_df[id_col] == ours_df[id_col]).all()}")
    print(f"   Attention vs Ours:  {(attention_df[id_col] == ours_df[id_col]).all()}")

# ===================== 3. 均值与胜率诊断（针对你的情况） =====================
print("\n" + "=" * 60)
print("🟡 2. 均值与胜率诊断（Dice 差 0.01 应有的样子）")
print("=" * 60)

def diagnose(name, base_df, ours_df):
    print(f"\n--- {name} ---")
    # 均值
    dice_diff = (ours_df['dice'] - base_df['dice']).mean()
    hd95_diff = (base_df['hd95'] - ours_df['hd95']).mean()  # 注意方向
    
    print(f"Dice 均值差 (Ours-Base): {dice_diff:.4f}")
    print(f"HD95 均值差 (Base-Ours): {hd95_diff:.4f}")
    
    # 胜率（100个样本）
    dice_wins = (ours_df['dice'] > base_df['dice']).sum()
    hd95_wins = (ours_df['hd95'] < base_df['hd95']).sum()
    
    print(f"Dice 胜率: {dice_wins} / {len(base_df)}")
    print(f"HD95 胜率: {hd95_wins} / {len(base_df)}")

diagnose("UNet vs Ours", unet_df, ours_df)
diagnose("DeepLab vs Ours", deeplabv3plus_df, ours_df)
diagnose("Attention vs Ours", attention_df, ours_df)

# ===================== 4. 正确的统计检验 =====================
print("\n" + "=" * 60)
print("🟢 3. 修正后的统计检验（仅当数据对齐时才可信）")
print("=" * 60)

def safe_stat_test(name, base_df, ours_df):
    print(f"\n{name}:")
    
    # Dice
    t_stat_d, p_t_d = ttest_rel(base_df['dice'], ours_df['dice'])
    w_stat_d, p_w_d = wilcoxon(base_df['dice'], ours_df['dice'])
    
    # HD95
    t_stat_h, p_t_h = ttest_rel(base_df['hd95'], ours_df['hd95'])
    w_stat_h, p_w_h = wilcoxon(base_df['hd95'], ours_df['hd95'])
    
    print(f"  Dice   - t-test: {p_t_d:.5e}, Wilcoxon: {p_w_d:.5e}")
    print(f"  HD95   - t-test: {p_t_h:.5e}, Wilcoxon: {p_w_h:.5e}")

safe_stat_test("UNet vs Ours", unet_df, ours_df)
safe_stat_test("DeepLab vs Ours", deeplabv3plus_df, ours_df)
safe_stat_test("Attention vs Ours", attention_df, ours_df)

# ===================== 5. 最终结论 =====================
print("\n" + "=" * 60)
print("📊 结果解读指南")
print("=" * 60)
print("""
1. 如果【对齐检查】出现 False：
   👉 之前所有 p 值作废！请先按 ID 重新对齐 CSV。

2. 如果【胜率】接近 100/100：
   👉 即使 Dice 差 0.01，也说明数据错位（不可能全赢）。

3. 如果【均值差 0.01】且【胜率 50~60】：
   👉 p 值应该在 0.05 ~ 0.5 之间。
   👉 如果此时 p 还是 0.00000，说明代码或数据有严重 Bug。
""")