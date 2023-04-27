import os
from typing import Dict, List
import cv2
import numpy as np

#------------------
#参数
#------------------
from PIL import Image

label_path='./'
image_path=''
output_path=''
color_map = {
    0: [0, 0, 0],      # 背景
    1: [0, 255, 0],    # 类别1
    2: [0, 0, 255],    # 类别2
    # 更多类别...
}
alpha=0.5


def main():
    blend_raw_images()



def blend_images(image1: Image.Image, image2: Image.Image, alpha: float) -> Image.Image:
    """
    融合两幅RGB图像，设定指定的alpha，对于第一张图像零像素位置，完全使用第二张图像，其余位置按alpha进行融合。

    Args:
        image1 (Image.Image): 第一张RGB图像。
        image2 (Image.Image): 第二张RGB图像。
        alpha (float): 融合时的权重，范围为 0.0 到 1.0。

    Returns:
        Image.Image: 融合后的图像。
    """

    # 将PIL.Image转换为numpy数组
    image1_np = np.array(image1)
    image2_np = np.array(image2)

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

def blend_raw_images():
    pass

if __name__ == "__main__":
    main()