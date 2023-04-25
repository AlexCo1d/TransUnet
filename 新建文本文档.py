import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import shutil
import random
from torchinfo import summary

from nets.TransUnet import get_transNet
from nets.vit_seg_modeling import VisionTransformer
from nets.vit_seg_modeling_resnet_skip import *

path = r'D:/PycharmProjects/unet/transunet 癌细胞/transunet/VOCdevkit/VOC2007/SegmentationClass/'
newpath = r'D:\PycharmProjects\unet\transunet 癌细胞\transunet\VOCdevkit\VOC2007\lable'


# def turnto24(path):
#   fileList = []
#   files = os.listdir(path)
#   i=0
#   for f in files:
#     imgpath = path + '/' +f
#     img=Image.open(f).convert('RGB')
#     dirpath = newpath
#     file_name, file_extend = os.path.splitext(f)
#     dst = os.path.join(os.path.abspath(dirpath), file_name + '.jpg')
#     img.save(dst)
# turnto24(path)
def is_positive_sample(img):
    return img.getextrema() != (0, 0)


def main():
    image_folder = r'D:\learning\UNNC 科研\data\nnUNet\final_image_plus_video'
    folder_label = os.path.join(image_folder, 'label')
    folder_image = os.path.join(image_folder, 'image')
    output_folder_image = r'D:\learning\UNNC 科研\data\nnUNet\bal_image'
    output_folder_label = r'D:\learning\UNNC 科研\data\nnUNet\bal_label'
    output_folder_extra = r'D:\learning\UNNC 科研\data\nnUNet\extra_for_test'
    if not os.path.exists(output_folder_image):
        os.makedirs(output_folder_image)
    if not os.path.exists(output_folder_label):
        os.makedirs(output_folder_label)
    if not os.path.exists(output_folder_extra):
        os.makedirs(os.path.join(output_folder_extra))
        os.makedirs(os.path.join(output_folder_extra,'label'))
        os.makedirs(os.path.join(output_folder_extra,'image'))
    positive_samples = []
    negative_samples = []

    for img_file in os.listdir(folder_label):
        img_path = os.path.join(folder_label, img_file)
        try:
            img = Image.open(img_path)
        except IOError:
            print(f'Error opening image file: {img_path}')
            continue

        if is_positive_sample(img):
            positive_samples.append(img_path)
        else:
            negative_samples.append(img_path)

    min_samples = min(len(positive_samples), len(negative_samples))
    print(f"min_samples: {min_samples}")
    combined_samples = []

    for i in range(min_samples):
        combined_samples.append(positive_samples.pop(random.randrange(len(positive_samples))))
        combined_samples.append(negative_samples.pop(random.randrange(len(negative_samples))))

    for i, sample in enumerate(combined_samples):
        shutil.copy(sample.replace('.png', '.jpg').replace('\label', '\image'),
                    os.path.join(output_folder_image, os.path.basename(sample).replace('.png', '.jpg')))
        shutil.copy(sample, os.path.join(output_folder_label, os.path.basename(sample)))

    for extra_sample in positive_samples + negative_samples:
        if extra_sample not in combined_samples:
            shutil.copy(extra_sample, os.path.join(output_folder_extra, 'label', os.path.basename(extra_sample)))
            shutil.copy(extra_sample.replace('.png', '.jpg').replace('\label', '\image'),
                        os.path.join(output_folder_extra, 'image',
                                     os.path.basename(extra_sample).replace('.png', '.jpg')))

    print(f'Saved {min_samples * 2} images with 1:1 ratio of positive and negative samples.')


def txt():
    image_folder = './VOCdevkit/VOC2007/JPEGImages'

    # 指定要写入的txt文件路径
    output_txt_file = './VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt'

    all_files = os.listdir(image_folder)

    # 筛选出所有的JPEG图片
    jpeg_files = [file.replace('.jpg', '') for file in all_files if file.lower().endswith('.jpg')]

    # 以覆盖模式打开指定的txt文件
    with open(output_txt_file, 'w') as f:
        # 遍历所有的JPEG图片
        for jpeg_file in jpeg_files:
            # 将图片名称写入txt文件中，每个文件名占据一行
            f.write(jpeg_file + '\n')


def label_to_onehot(label_image, num_classes):
    one_hot_image = np.eye(num_classes)[label_image]
    one_hot_image = np.transpose(one_hot_image, (2, 0, 1))
    return one_hot_image


def observe():
    from nets.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
    img_size = 256
    vit_patches_size = 16
    vit_name = 'R50-ViT-B_16'

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    print(config_vit)
    net=ResNetV2_ASPP(block_units=config_vit.resnet.num_layers, width_factor=config_vit.resnet.width_factor)
    print(config_vit.resnet.width_factor)
    # net = VisionTransformer(config_vit, img_size=img_size, num_classes=1)
    summary(net,input_size=(2,3,256,256))
    #summary(VisionTransformer(config_vit, img_size=img_size, num_classes=1),input_size=(2,3,256,256))
    print(VisionTransformer(config_vit, img_size=img_size, num_classes=1))


if __name__ == '__main__':
    observe()
    # main()
    # txt()
