import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import shutil
import random

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
    folder_label = os.path.join(image_folder,'label')
    folder_image = os.path.join(image_folder, 'image')
    output_folder_image = r'D:\learning\UNNC 科研\data\nnUNet\bal_image'
    output_folder_label = r'D:\learning\UNNC 科研\data\nnUNet\bal_label'
    output_folder_extra = r'D:\learning\UNNC 科研\data\nnUNet\extra_for_test'
    if not os.path.exists(output_folder_image):
        os.makedirs(output_folder_image)
    if not os.path.exists(output_folder_label):
        os.makedirs(output_folder_label)

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

    min_samples = min(len(positive_samples), len(negative_samples))-100
    print(f"min_samples: {min_samples}")
    combined_samples = []

    for i in range(min_samples):
        combined_samples.append(positive_samples.pop(random.randrange(len(positive_samples))))
        combined_samples.append(negative_samples.pop(random.randrange(len(negative_samples))))

    for i, sample in enumerate(combined_samples):
        shutil.copy(sample.replace('.png','.jpg').replace('\label','\image'),os.path.join(output_folder_image,f'{i}.jpg'))
        shutil.copy(sample,os.path.join(output_folder_label,f'{i}.png'))

    for extra_sample in positive_samples + negative_samples:
        if extra_sample not in combined_samples:
            shutil.copy(extra_sample, os.path.join(output_folder_extra,'label', os.path.basename(extra_sample)))
            shutil.copy(extra_sample.replace('.png','.jpg').replace('\label','\image'), os.path.join(output_folder_extra,'image',os.path.basename(extra_sample).replace('.png','.jpg')))


    print(f'Saved {min_samples * 2} images with 1:1 ratio of positive and negative samples.')


if __name__ == '__main__':
    main()
