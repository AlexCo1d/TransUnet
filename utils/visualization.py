import os
from typing import Dict, List
import cv2
import numpy as np

#------------------
#参数
#------------------
from PIL import Image

from train_config import config

txt_path="./VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt"
image_path='./VOCdevkit/VOC2007/JPEGImages'
label_path='./VOCdevkit/VOC2007/SegmentationClass'
output_path='./BUSI_visualization/gt'
color_map = [(0, 0, 0), (0, 255, 0), (255, 0, 0), (128, 128, 0), (128, 0, 128), (0, 128, 128),
               (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
               (192, 0, 128),
               (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
               (0, 64, 128), (128, 64, 12)]
alpha=0.5

type=config.image_type

def main():
    blend_raw_images(label_path,image_path,output_path,color_map,alpha)


def convert_to_rgb(label_img, colormap):
    rgb_img = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)

    for label in range(len(colormap)):
        color = colormap[label]
        rgb_img[np.where(label_img == label)] = color

    return rgb_img


def blend_images(image1: Image.Image, image2: Image.Image, alpha: float) -> Image.Image:
    """
    融合两幅RGB图像，设定指定的alpha，对于第一张图像零像素位置，完全使用第二张图像，其余位置按alpha进行融合。

    Args:
        image1 (Image.Image): 第一张RGB图像,label
        image2 (Image.Image): 第二张RGB图像,image
        alpha (float): 融合时的权重，范围为 0.0 到 1.0。

    Returns:
        Image.Image: 融合后的图像。
    """

    # 将PIL.Image转换为numpy数组
    image1_np = np.array(image1)
    image2_np = np.array(image2)
    if len(image2_np.shape)==2:
        image2_np = np.stack((image2_np,) * 3, axis=-1)
    # 创建一个空白的输出图像（与输入图像大小相同）
    blended_np = np.zeros_like(image1_np)

    # 找到第一张图像中非零像素的位置
    non_zero_indices = np.any(image1_np != 0, axis=-1)

    # 对于第一张图像零像素位置，完全使用第二张图像
    blended_np[~non_zero_indices] = image2_np[~non_zero_indices]

    # 按alpha进行融合的其余位置
    blended_np[non_zero_indices] = (1 - alpha) * image1_np[non_zero_indices] + alpha * image2_np[non_zero_indices]

    # 将numpy数组转换回PIL.Image
    blended_image = Image.fromarray(blended_np.astype(np.uint8))

    return blended_image

def blend_raw_images(label_path,image_path,output_path,color_map,alpha=0.5):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # 将2个文件夹的图像融合输出进新文件夹
    with open(txt_path, "r") as file:
        lines = file.readlines()
    for label_name in os.listdir(label_path):
        label=np.array(Image.open(os.path.join(label_path,label_name)))
        label=convert_to_rgb(label,colormap=color_map)
        image=np.array(Image.open(os.path.join(image_path,label_name.replace('.png',type))))
        blend_image=blend_images(Image.fromarray(label),Image.fromarray(image),alpha)
        blend_image.save(os.path.join(output_path,label_name).replace('.png','.jpg'))


if __name__ == "__main__":
    main()