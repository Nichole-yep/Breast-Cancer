# Baseline Code Package

把这 3 个文件放到你们项目对应位置：

```text
Breast-Cancer/
├── models/
│   └── baseline_models.py
├── train_baseline_models.py
└── evaluate/
    └── eval_baseline_models.py
```

## 先测试模型是否能输出正确形状

```bash
python models/baseline_models.py
```

预期输出类似：

```text
unet torch.Size([2, 1, 256, 256])
attention_unet torch.Size([2, 1, 256, 256])
deeplabv3plus torch.Size([2, 1, 256, 256])
```

## 训练 U-Net baseline

CPU 建议先用小参数测试：

```bash
python train_baseline_models.py --model unet --device cpu --epochs 2 --batch_size 1 --base_channels 16
```

正式一点可以用：

```bash
python train_baseline_models.py --model unet --device cpu --epochs 50 --batch_size 2 --base_channels 32
```

## 评估 U-Net baseline

```bash
python evaluate/eval_baseline_models.py --model unet --weights results/weights/best_unet.pth --csv_file preprocess/test.csv --device cpu
```

## 训练和评估其他 baseline

```bash
python train_baseline_models.py --model attention_unet --device cpu --epochs 50 --batch_size 2
python evaluate/eval_baseline_models.py --model attention_unet --weights results/weights/best_attention_unet.pth --csv_file preprocess/test.csv --device cpu

python train_baseline_models.py --model deeplabv3plus --device cpu --epochs 50 --batch_size 2
python evaluate/eval_baseline_models.py --model deeplabv3plus --weights results/weights/best_deeplabv3plus.pth --csv_file preprocess/test.csv --device cpu
```

## 说明

- 训练和评估都使用你们自己的 `preprocess/dataset.py`。
- 评价指标使用你们自己的 `evaluate/eval.py` 里的 `SegmentationMetrics`。
- baseline loss 使用 `0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss`，不使用 Edge Loss，这样作为对比模型更清楚。
