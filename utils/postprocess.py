import cv2
import numpy as np
from train_config import config
from PIL import Image

def largest_k_connected_components(binary_image, k):
    """
    Finds the largest k connected components in a binary image.

    Args:
        binary_image (np.array): A grayscale binary image (2D array) with background 0 and objects non-zero.
        k (int): Number of largest connected components to find.

    Returns:
        np.array: A binary image with only the k largest connected components.
    """
    # Find all connected components and their stats
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Get the areas of all connected components (except the background at index 0)
    areas = stats[1:, cv2.CC_STAT_AREA]

    # Find the indices of the k largest connected components
    # (Note: In case there are less than k components, take all)
    k = min(k, len(areas))
    largest_k_indices = np.argpartition(areas, -k)[-k:]

    # Add 1 to the indices because we ignored the background
    largest_k_indices += 1

    # Create an empty image to store the result
    output = np.zeros_like(binary_image)

    for i in largest_k_indices:
        output[labels == i] = 1

    return output


def postprocess(image):
    '''

    Args:
        image:

    Returns:
        经过形态学处理的图片
    '''
    if config.component != -1:
        image=largest_k_connected_components(image,config.component)

    # 开运算，去除孤立点
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # 闭运算，去除空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return image




