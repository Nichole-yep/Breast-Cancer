
# AUTO PATH FIX FOR FINAL GITHUB STRUCTURE
from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
for _p in [_PROJECT_ROOT, _PROJECT_ROOT / "src"]:
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# END AUTO PATH FIX
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 基础设置
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelweight'] = 'bold'

fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')

# ==========================================
# 2. 真实数据
# ==========================================
models = ['Attention U-Net', 'DeepLabV3+', 'U-Net', 'My Model']
params = np.array([7.98, 2.30, 4.32, 6.72])      # M (用于气泡大小)
flops = np.array([29.04, 27.26, 20.25, 4.93])    # G (用于X轴)
dice = np.array([0.6898, 0.6792, 0.6615, 0.7814])

# 气泡大小：正比于参数量 (Params)
# 归一化到 150-800 之间
sizes = 150 + 650 * (params - params.min()) / (params.max() - params.min())

# ==========================================
# 3. 绘图 (X轴现在是FLOPs)
# ==========================================
# 基线模型：深蓝色圆点
baseline_scatter = ax.scatter(flops[:-1], dice[:-1], 
                           s=sizes[:-1], 
                           c='#1f77b4', 
                           alpha=0.8, 
                           edgecolors='white', 
                           linewidth=0.8,
                           label='Baselines')

# 你的模型：深红色星号
my_scatter = ax.scatter(flops[-1], dice[-1], 
                        s=sizes[-1],  # 大小同样由参数量决定
                        c='#d62728', 
                        marker='*', 
                        edgecolors='white', 
                        linewidth=1.2,
                        label='My Model')

# ==========================================
# 4. 解决图例图标过大的问题 (简化版)
# ==========================================
# 手动创建图例句柄，使用较小的固定大小
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', 
               markerfacecolor='#1f77b4', markersize=8, label='Baselines'),
    plt.Line2D([0], [0], marker='*', color='w', 
               markerfacecolor='#d62728', markersize=12, label='My Model')
]

# ==========================================
# 5. 标注与样式
# ==========================================
# 标注 - 因为X轴变了，需要调整偏移量
offsets = [
    (15, 20),    # Attention U-Net：右上
    (-20, -25),  # DeepLabV3+：左下
    (0, 25),     # U-Net：正上
    (0, -35)     # My Model：正下（你的点在最左边）
]

for i, (model, x, y) in enumerate(zip(models, flops, dice)):
    ax.annotate(model, (x, y), xytext=offsets[i], textcoords='offset points',
                fontsize=10, fontweight='bold' if i == 3 else 'normal',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# 坐标轴 - 现在X轴是FLOPs
ax.set_xlabel('Computational Cost (G FLOPs)', fontsize=12)
ax.set_ylabel('Dice Score', fontsize=12)

# 根据FLOPs范围设置X轴
ax.set_xlim(0, 35)
ax.set_ylim(0.64, 0.80)

# 使用手动创建的图例
ax.legend(handles=legend_handles, loc='lower left', frameon=True, framealpha=0.9)

#plt.title('Model Efficiency vs. Segmentation Accuracy', fontsize=14, pad=15)

# 网格和边框
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('medical_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()