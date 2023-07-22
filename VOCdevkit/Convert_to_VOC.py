import os
import random
from sklearn.model_selection import StratifiedKFold
from train_config import config
from PIL import Image
import numpy as np

segfilepath='./VOCdevkit/VOC2007/SegmentationClass'
saveBasePath="./VOCdevkit/VOC2007/ImageSets/Segmentation"

# 分割任务，正负样本均衡的数据集划分
temp_seg = os.listdir(segfilepath)
image_seg = []
neg_pos_label=[]
for seg in temp_seg:
    if seg.endswith(".png"):
        image_seg.append(seg)
        temp=np.array(Image.open(os.path.join(segfilepath,seg)))
        if len(np.unique(temp))==1:     # negative sample
            neg_pos_label.append(0)
        else: neg_pos_label.append(1)   # positive sample

# to judge the n_fold's setting and divide the dataset
if config.n_fold > 1:
    skf = StratifiedKFold(n_splits=config.n_fold)
    for i, (train_index, valid_index) in enumerate(skf.split(image_seg, neg_pos_label)):
        train_images = [image_seg[idx] for idx in train_index]
        valid_images = [image_seg[idx] for idx in valid_index]

        # 我们需要把 '.jpg' 后缀去掉，因为 VOC 的 ImageSets 文件只包含图像的文件名，不包含后缀
        train_images = [os.path.splitext(img)[0] for img in train_images]
        valid_images = [os.path.splitext(img)[0] for img in valid_images]

        with open(f'train_{i+1}.txt', 'w') as f:
            for img in train_images:
                f.write(img + '\n')

        with open(f'valid_{i+1}.txt', 'w') as f:
            for img in valid_images:
                f.write(img + '\n')
else:   # all need train
    with open(f'train_1.txt', 'w') as ftrain:
        for img in os.listdir(image_seg):
            name = img.split('.')[0] + '\n'
            ftrain.write(name)
    # you should then decide the val.txt for prediction


# trainval_percent=1
# train_percent=0.8
#
# temp_seg = os.listdir(segfilepath)
# total_seg = []
# for seg in temp_seg:
#     if seg.endswith(".png"):
#         total_seg.append(seg)
#
# num=len(total_seg)
# list=range(num)
# tv=int(num*trainval_percent)
# tr=int(tv*train_percent)
# trainval= random.sample(list,tv)
# train=random.sample(trainval,tr)
#
# print("train and val size",tv)
# print("traub suze",tr)
# ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')
# ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')
# ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')
# fval = open(os.path.join(saveBasePath,'val.txt'), 'w')
#
# for i in list:
#     name=total_seg[i].split('.')[0]+'\n'
#     if i in trainval:
#         ftrainval.write(name)
#         if i in train:
#             ftrain.write(name)
#         else:
#             fval.write(name)
#     else:
#         ftest.write(name)
#
# ftrainval.close()
# ftrain.close()
# fval.close()
# ftest .close()
