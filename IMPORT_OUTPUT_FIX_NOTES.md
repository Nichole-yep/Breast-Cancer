# Import and Output Path Fix Notes

本次修整针对组长最终 GitHub 结构：`src / scripts / utils / outputs`。

已处理：

1. 新增 `project_paths.py`，统一项目根目录、数据、输出位置。
2. 新增 `src/__init__.py`、`scripts/__init__.py`，让 Python 包导入更稳定。
3. 将旧结构路径：
   - `preprocess/*.csv` 改为 `src/data/*.csv`
   - `results/...` 改为 `outputs/results/...`
   - `visualization/outputs/...` 改为 `outputs/visualization/outputs/...`
4. 修复 import：
   - `models.*` 改为 `src.models.*`
   - `preprocess.dataset` 改为 `src.data.dataset`
   - `evaluate.eval` 改为 `utils.eval`
   - `viz_utils` 改为 `utils.viz_utils`
5. 修复 `src/data/prepare_data.py`：
   - 数据集默认读取 `src/data/Dataset_BUSI_with_GT`
   - CSV 默认输出到 `src/data/train.csv / val.csv / test.csv`
   - CSV 中保存相对路径，方便 GitHub 复现
6. 修复 `src/data/dataset.py`、`utils/viz_utils.py`、`utils/eval.py` 的路径解析和 Windows 中文路径读取问题。
7. 修改 `.gitignore`：保留数据集、CSV、outputs 结果；仍默认忽略 `.pth/.pt/.ckpt` 权重文件。

建议运行顺序：

```bash
python src/data/prepare_data.py
python utils/eval.py --device cpu
python scripts/01_visualize_ours_prediction_boundary.py --device cpu
python scripts/04_plot_pixel_roc_pr.py --device cpu
python scripts/06_visualize_failure_cases.py --device cpu
```

如需 baseline 评估：

```bash
python utils/eval_baseline_models.py --model unet --weights outputs/results/weights/best_unet.pth --device cpu
```
