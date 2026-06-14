#用于BUSI数据预处理与数据加载
#模型构建者可自行选择添加/删除散斑滤波、CLAHE增强、数据增强
import numpy as np
import cv2
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset ,DataLoader
# 输入参数：路径文件，是否启用lee、clahe、随机数据增强
class BUSIDataset(Dataset):
    def __init__(self, csv_file, ues_lee=True,ues_clahe=True,augment=True):

        self.df=pd.read_csv(csv_file)
        self.ues_lee=ues_lee
        self.ues_clahe=ues_clahe
        self.augment=augment

        if ues_clahe:  #使用常用值
            self.clahe=cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

        self.transform=self._get_transform(augment)
        #数据变换管道：基础变换+随机增强
    def _get_transform(self, augment):
        base_transform =[
            A.Resize(256, 256), #统一缩放为256×256
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2() #转为PyTorch张量，HWC→CHW
        ]
        if augment:
            aug_transform = [
                # 随机旋转正负 45 度，并随机缩放和平移 
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.6),
                # 随机水平和上下翻转
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                # 模拟超声探头接触不良时的亮度变化
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                # 模拟斑点噪声
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            ]
            transforms=aug_transform+base_transform
        else:
            transforms=base_transform
        return A.Compose(transforms,additional_targets={'edge_mask':'mask'})
    #lee滤波
    def _lee_filter(self,img,window=7):
        img_float=img.astype(np.float64)#防溢出
        mean=cv2.blur(img_float,(window,window)) #局部均值
        mean_sq=cv2.blur(img_float**2,(window,window))#局部平方均值
        var=mean_sq-mean**2 #局部方差
        noise_var=np.var(img_float-mean)
        #自适应增益k
        k=var/(var+noise_var+1e-8)
        k=np.clip(k,0.0,1.0)
        #滤波输出：均值+k*（原始值-均值）
        out=mean+k*(img_float-mean)
        return np.clip(out,0,255).astype(np.uint8)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        '''根据索引返回一个样本（图像、分割mask、边界mask）'''
        row=self.df.iloc[idx]
        img_path=row['img_path']
        mask_path=row['mask_path']
        #-------读取图像-------
        img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise FileNotFoundError(f'无法读取图像/mask！')
        #------lee滤波--------
        if self.ues_lee:
            img=self._lee_filter(img,window=7)
        #-----CLAHE增强-------
        if self.ues_clahe:
            img=self.clahe.apply(img)
        #---灰度图转RGB-----
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        #-----mask二值化----
        mask=(mask>127).astype(np.uint8)
        #-----生成edge mask（肿瘤边界）-----
        #使用形态学膨胀-腐蚀
        kernel=np.ones((3,3),np.uint8)
        dilate=cv2.dilate(mask,kernel,iterations=1)
        erode =cv2.erode(mask,kernel,iterations=1)
        edge=(dilate-erode).astype(np.uint8)
        #------数据变换-------
        augmented=self.transform(image=img,mask=mask,edge_mask=edge)
        img_tensor=augmented['image']
        mask_tensor=augmented['mask']
        edge_mask_tensor=augmented['edge_mask']
        #-------为mask,edge增加通道维度--------
        mask_tensor=mask_tensor.unsqueeze(0).float()
        edge_mask_tensor=edge_mask_tensor.unsqueeze(0).float()

        #--------返回张量---------
        return img_tensor, mask_tensor, edge_mask_tensor
#获取训练、验证、测试三个 DataLoader（训练集开随机增强）
def get_loaders(train_csv,val_csv,test_csv,batch_size=2,use_lee=True,use_clahe=True):
    train_data=BUSIDataset(train_csv)
    val_data=BUSIDataset(val_csv,augment=False)
    test_data=BUSIDataset(test_csv,augment=False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader,val_loader,test_loader


#-------测试---------
if __name__ == '__main__':
    #需要当前目录下已有路径文件
    train_loader, val_loader, test_loader = get_loaders(
        'train.csv', 'val.csv', 'test.csv',
        batch_size=2,
        use_lee=True,
        use_clahe=True
    )
    #取一个batch测试输出形状和值域
    for images, masks, edges in train_loader:
        print("图像张量形状:", images.shape)#预期:[2,3,256,256]
        print("Mask 形状:", masks.shape)#预期:[2,1,256,256]
        print("Edge 形状:", edges.shape)#预期:同上
        print("图像值范围:", images.min().item(), images.max().item())#应在-1到1之间
        break





