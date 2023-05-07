import torch
import torch.nn as nn
import functools
import torch.nn.functional as F

from nets.vit_cbam_modeling import Vit_CBAM, Vit_CBAM_ASPP
from .transunet_modeling import VisionTransformer as ViT_seg
from .transunet_modeling import CONFIGS as CONFIGS_ViT_seg
from train_config import config


def get_Net(n_classes, img_size=256):
    vit_patches_size = 16
    vit_name = 'R50-ViT-B_16'

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = n_classes
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    # net = ViT_seg(config_vit, img_size=img_size, num_classes=n_classes)
    global net
    if config.model == 'Vit_CBAM_ASPP':
        net = Vit_CBAM_ASPP(config_vit, img_size=img_size, num_classes=n_classes)
    elif config.model == 'Vit_CBAM':
        net = Vit_CBAM(config_vit, img_size=img_size, num_classes=n_classes)
    else:
        net = ViT_seg(config_vit, img_size=img_size, num_classes=n_classes)

    return net


if __name__ == '__main__':
    net = get_Net(1)
    img = torch.randn((2, 3, 512, 512))
    segments = net(img)
    print(segments.size())
    # for edge in edges:
    #     print(edge.size())
