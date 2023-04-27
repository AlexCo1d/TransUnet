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
    batch_visualize(image_path,label_path,output_path,color_map,alpha)


def visualize_label_on_image(image: np.ndarray, label: np.ndarray, color_map: Dict[int, List[int]], alpha: float = 0.5) -> np.ndarray:
    """
    可视化标签图像，将标签叠加到原图像上。

    Args:
        image (np.ndarray): 原图像。
        label (np.ndarray): 标签图像。
        color_map (Dict[int, List[int]]): 每个类别标签对应的颜色映射字典，键为整数类别值，值为 [B, G, R] 彩色列表。
        alpha (float): 原图像和标签图像透明度的权重，范围为 0.0 到 1.0。默认值为 0.5。

    Returns:
        np.ndarray: 叠加了标签的原图像。
    """
    # 将灰度标签图像转换为彩色标签图像
    label_color = np.zeros_like(image)
    for label_value, color in color_map.items():
        label_color[label == label_value] = color

    # 将标签图像叠加到原图像上
    overlayed = cv2.addWeighted(image, 1 - alpha, label_color, alpha, 0)

    return overlayed


def batch_visualize(image_folder: str, label_folder: str, output_folder: str, color_map: Dict[int, List[int]], alpha: float = 0.5) -> None:
    image_names = os.listdir(image_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in image_names:
        image_path = os.path.join(image_folder, image_name)
        label_path = os.path.join(label_folder, image_name.replace('jpg','.png'))
        output=os.path.join(output_folder,image_name)
        if os.path.exists(label_path):
            image = cv2.imread(image_path)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            overlayed = visualize_label_on_image(image, label, color_map, alpha)

            cv2.imwrite(output,overlayed)

def blend_images(image: Image.Image, label: Image.Image, alpha: float) -> Image.Image:
    """
    融合两幅RGB图像，设定指定的alpha，对于第一张图像零像素位置，完全使用第二张图像，其余位置按alpha进行融合。

    Args:
        image (Image.Image): 第一张RGB图像。
        label (Image.Image): 第二张RGB图像。
        alpha (float): 融合时的权重，范围为 0.0 到 1.0。

    Returns:
        Image.Image: 融合后的图像。
    """

    # 将PIL.Image转换为numpy数组
    image = np.array(image)
    label = np.array(label)

    # 创建一个空白的输出图像（与输入图像大小相同）
    blended_np = np.zeros_like(image)

    # 找到第一张图像中非零像素的位置
    non_zero_indices = np.any(image != 0, axis=-1)

    # 对于第一张图像零像素位置，完全使用第二张图像
    blended_np[~non_zero_indices] = label[~non_zero_indices]

    # 按alpha进行融合的其余位置
    blended_np[non_zero_indices] = (1 - alpha) * image[non_zero_indices] + alpha * label[non_zero_indices]

    # 将numpy数组转换回PIL.Image
    blended_image = Image.fromarray(blended_np.astype(np.uint8))

    return blended_image



if __name__ == "__main__":
    main()