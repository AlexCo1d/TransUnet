import torch
import torch.nn as nn
import functools
import torch.nn.functional as F

from nets.vit_cbam_aspp_modeling import Vit_CBAM, Vit_CBAM_ASPP
from nets.vit_cbam_modeling import Vit_CBAM, Vit_CBAM_CBAM
from .transunet_modeling import VisionTransformer as ViT_seg
from .transunet_modeling import CONFIGS as CONFIGS_ViT_seg
from train_config import config


def get_Net(n_classes, img_size=256):
    vit_patches_size = 16
    vit_name = 'R50-ViT-B_16'

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = n_classes
    config_vit.n_skip=config.n_skip
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    # net = ViT_seg(config_vit, img_size=img_size, num_classes=n_classes)

    if config_vit.n_skip == 3:
        config_vit.decoder_channels=(256, 128, 64, 16)
        config_vit.skip_channels = [512, 256, 64, 0]
    elif config_vit.n_skip == 4:
        config_vit.skip_channels = [1024, 512, 256, 64]
        config_vit.decoder_channels=(512,256, 64,16)

    global net

    if config.model == 'Vit_CBAM_ASPP':
        net = Vit_CBAM_ASPP(config_vit, img_size=img_size, num_classes=n_classes)
    elif config.model == 'Vit_CBAM':
        net = Vit_CBAM(config_vit, img_size=img_size, num_classes=n_classes)
    elif config.model == 'Vit_CBAM_CBAM':
        net = Vit_CBAM_CBAM(config_vit, img_size=img_size, num_classes=n_classes)
    else:
        # default BASIC
        config_vit.decoder_channels = (256, 128, 64, 16)
        config_vit.skip_channels = [512, 256, 64, 0]
        config_vit.n_skip = 3
        net = ViT_seg(config_vit, img_size=img_size, num_classes=n_classes)

    return net


if __name__ == '__main__':
    net = get_Net(1)
    img = torch.randn((2, 3, 512, 512))
    segments = net(img)
    print(segments.size())
    # for edge in edges:
    #     print(edge.size())
