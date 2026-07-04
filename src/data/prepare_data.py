import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#合并mask函数
def merge_masks(mask_paths):
    merged_masks =None
    for p in mask_paths:
        mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if merged_masks is None:
            merged_masks = (mask>127).astype(np.uint8) #存入首个mask，大于127的肿瘤区域设为1
        else:
            merged_masks = np.logical_or(merged_masks, (mask>127).astype(np.uint8))
    if merged_masks is None:
        return None
    return merged_masks*255

#遍历benign和malignant，返回(img_path,mask_path)列表
def collect_samples(root_path):
    samples=[]
    for category in ['benign','malignant']:#循环处理2个文件夹
        dir_path = os.path.join(root_path, category)#获取完整路径
        if not os.path.isdir(dir_path): #若文件夹不存在则跳过
            continue
        all_files = os.listdir(dir_path) #列出当前类别文件夹下的所有文件名
        imgs=[f for f in all_files if f.endswith('.png') and '_mask'not in f]
        for f in imgs:
            img_path = os.path.join(dir_path, f)
            basename = f[:-4]#获得图像基础名
            f_mask=[m for m in all_files if m.startswith(basename+'_mask') and f.endswith('.png')]
            f_mask.sort()#按名字排序
            if len(f_mask)==0:
                print('no mask for {}'.format(f))
                continue
            if len(f_mask)==1:
                mask_path = os.path.join(dir_path, f_mask[0])
            else:
                all_mask_path=[os.path.join(dir_path, m) for m in f_mask]
                merge_mask=merge_masks(all_mask_path)
                if merge_mask is None:
                    continue
                merge_mask_path=os.path.join(dir_path,basename)+'merged_mask.png'
                cv2.imwrite(merge_mask_path, merge_mask)
                mask_path=merge_mask_path
            samples.append((img_path, mask_path))
    return samples

if __name__=='__main__':
    busi_root='Dataset_BUSI_with_GT'
    samples=collect_samples(busi_root)
    print(f'total samples: {len(samples)}')

#划分数据集：训练集70%，验证集15%，测试集15%
train_and_val,test=train_test_split(samples,test_size=0.15,random_state=42)
train,val=train_test_split(train_and_val,test_size=0.15/0.85,random_state=42)

#保存csv的函数：
def save_csv(data,filename):
    df = pd.DataFrame(data,columns=['img_path','mask_path'])
    df.to_csv(filename, index=False)#存为csv文件不保留索引
    print(f"saved {len(data)} samples to {filename}")

save_csv(train,'train.csv')
save_csv(val,'val.csv')
save_csv(test,'test.csv')






