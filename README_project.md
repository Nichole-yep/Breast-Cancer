

# 环境配置
建议使用 Python 3.8 及以上版本，将下载得到的压缩包解压至你的项目根目录下（pycharm）
```bash
conda create -n breast_cancer python=3.8
conda activate breast_cancer
pip install -r requirements.txt
```
# 数据准备
将BUSI数据集放在Breast-Cancer-main/src/data目录下（目录下已包含Dataset_BUSI_with_GT，也可删掉重新放置）

# 运行预处理脚本
```bash
python Breast-Cancer-main/src/data/prepare_data.py
```
生成 train.csv、val.csv、test.csv(划分数据集得到的样本路径)

# 训练模型
## 先测试模型是否能够输出正确形状
```bash
python Breast-Cancer-main/Breast-Cancer-main/src/models/baseline_models.py
```
预期输出类似：

```text
unet torch.Size([2, 1, 256, 256])
attention_unet torch.Size([2, 1, 256, 256])
deeplabv3plus torch.Size([2, 1, 256, 256])
```

## 训练基线模型（CPU建议使用小参数测试，readme中给出正式训练参数）
### 例：训练 U-Net baseline
```bash
python scripts/train_baseline_models.py --model unet --device cpu --epochs 50 --batch_size 2 --base_channels 32
```

## 训练、测试评估改进模型
```bash
python Breast-Cancer-main/scripts/train.py
```
## 显著性检验
```bash
phyton Breast-Cancer-main/Breast-Cancer-main/scripts/p-value.py
```






