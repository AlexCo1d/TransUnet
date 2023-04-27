import os
from typing import Dict, List
import cv2
import numpy as np

#------------------
#参数
#------------------
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


if __name__ == "__main__":
    main()