# Breast-Cancer BUSI Segmentation Project

本仓库为整理后的最终 GitHub 结构：

```text
Breast-Cancer/
├── configs/
├── scripts/                 # 训练、测试、可视化入口脚本
├── src/
│   ├── core/                # backbone / PPM / CBAM / decoder
│   ├── data/                # BUSI 数据集、train/val/test.csv、prepare_data.py
│   └── models/              # DBDS-Net 与 baseline 模型
├── utils/                   # 评估、混淆矩阵、可视化工具
└── outputs/                 # 所有输出结果、图、日志、指标
```

## 1. 数据划分

数据集放在：

```text
src/data/Dataset_BUSI_with_GT/
```

重新生成划分：

```bash
python src/data/prepare_data.py
```

输出：

```text
src/data/train.csv
src/data/val.csv
src/data/test.csv
```

## 2. 测试模型结构

```bash
python src/models/baseline_models.py
python src/models/ours.py
```

## 3. 评估最终模型 DBDS-Net

权重默认读取：

```text
outputs/results/weights/best_our_model.pth
```

运行：

```bash
python utils/eval.py --device cpu
```

如需手动指定权重：

```bash
python utils/eval.py --weights outputs/results/weights/best_our_model.pth --csv_file src/data/test.csv --device cpu
```

输出保存到：

```text
outputs/results/
```

## 4. 生成可视化结果

```bash
python scripts/01_visualize_ours_prediction_boundary.py --device cpu
python scripts/02_visualize_deep_supervision_drafts.py --device cpu
python scripts/03_plot_training_curves.py
python scripts/04_plot_pixel_roc_pr.py --device cpu
python scripts/05_visualize_feature_maps_ppm_cbam.py --device cpu
python scripts/06_visualize_failure_cases.py --device cpu
python scripts/07_plot_ours_training_curves.py
```

输出统一保存到：

```text
outputs/visualization/outputs/
```

## 5. 评估 baseline 模型

```bash
python utils/eval_baseline_models.py --model unet --weights outputs/results/weights/best_unet.pth --device cpu
python utils/eval_baseline_models.py --model attention_unet --weights outputs/results/weights/best_attention_unet.pth --device cpu
python utils/eval_baseline_models.py --model deeplabv3plus --weights outputs/results/weights/best_deeplabv3plus.pth --device cpu
```

输出保存到：

```text
outputs/results/baseline_metrics/
```

## 6. 注意

所有脚本建议在项目根目录运行，例如：

```bash
cd E:\FinalGithubProject\Breast-Cancer
python scripts/04_plot_pixel_roc_pr.py --device cpu
```

不要在 `scripts/` 或 `src/` 子文件夹里直接运行，否则相对路径容易混乱。
