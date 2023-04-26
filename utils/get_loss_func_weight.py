import numpy as np
import torch


def get_loss_weight(num_classes: int, pngs: torch.tensor):
    """

    Args:
        num_classes:
        pngs:

    Returns:
        w(numpy.ndarray): 损失函数权重

    """
    # 将张量转换为 NumPy 数组
    numpy_array = pngs.cpu().numpy()

    # 初始化一个一行 num_classes 列的数组，用于存储每个类别的数量
    class_counts = np.zeros(num_classes)

    # 遍历每个类别并统计它们在 NumPy 数组中的数量
    for i in range(num_classes):
        class_counts[i] = np.sum(numpy_array == i)
    t = class_counts / np.sum(class_counts)
    w = np.median(t) / t
    return w
