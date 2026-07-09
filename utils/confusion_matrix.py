
# AUTO PATH FIX FOR FINAL GITHUB STRUCTURE
from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
for _p in [_PROJECT_ROOT, _PROJECT_ROOT / "src"]:
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# END AUTO PATH FIX
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 准备数据
# ==========================================
cm = np.array([[5782985, 57475],   
              [212967, 369101]])   # TN FP FN TP

# 【核心修改】直接计算百分比
# 这样所有的值都会变成 0 到 1 之间的小数
cm_percent = cm / cm.sum()

plt.figure(figsize=(6, 5))

# ==========================================
# 2. 绘制热图（使用百分比数据） 
# ==========================================
# vmin=0, vmax=1 设定固定的颜色范围
ax = sns.heatmap(cm_percent, annot=True, fmt='.2%', 
                 cmap='Reds', 
                 vmin=0, vmax=1,          # 强制颜色映射在 0% 到 100% 之间
                 xticklabels=['Background', 'Tumor'], 
                 yticklabels=['Background', 'Tumor'],
                 annot_kws={"size": 14, "weight": "bold"}, # 字体加大
                 linewidths=1, linecolor='white')

# ==========================================
# 3. 标题和标注
# ==========================================
#plt.title('Confusion Matrix: Exp D (Normalized %)', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# 手动添加召回率和精确率（从百分比矩阵中读取）
# Sensitivity = 真正例 / (真正例 + 假反例) -> 即第二行的值之和的百分比
sensitivity = cm_percent[1, :].sum()
# Precision = 真正例 / (真正例 + 假正例) -> 即第二列的值之和的百分比
precision = cm_percent[:, 1].sum()

#plt.text(0.5, -0.15, f'Sensitivity (Recall): {sensitivity:.2%}', 
#         horizontalalignment='center', transform=ax.transAxes, fontsize=12, color='darkred')
#plt.text(0.5, -0.22, f'Precision: {precision:.2%}', 
#         horizontalalignment='center', transform=ax.transAxes, fontsize=12, color='darkred')

plt.tight_layout()
import os
os.makedirs('outputs/figures', exist_ok=True)
plt.savefig('outputs/figures/confusion_matrix_C.png', dpi=300, bbox_inches='tight')
plt.show()