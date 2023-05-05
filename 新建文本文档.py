import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import shutil
import random
from torchinfo import summary
import torch
from nets.TransUnet import get_transNet
from nets.transunet_modeling import *
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.init_weight import init_weights, traverse_unfreeze_block


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = unetConv2(out_size * 2, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)

class UNet_3Plus_DeepSup_CGM(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus_DeepSup_CGM, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear')###
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], n_classes, 3, padding=1)

        self.cls = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Conv2d(filters[4], 2, 1),
                    nn.AdaptiveMaxPool2d(1),
                    nn.Sigmoid())

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def dotProduct(self,seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        # -------------Classification-------------
        cls_branch = self.cls(hd5).squeeze(3).squeeze(2)  # (B,N,1,1)->(B,N)
        cls_branch_max = cls_branch.argmax(dim=1)
        cls_branch_max = cls_branch_max[:, np.newaxis].float()

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5) # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) # 128->256

        d1 = self.outconv1(hd1) # 256

        d1 = self.dotProduct(d1, cls_branch_max)
        d2 = self.dotProduct(d2, cls_branch_max)
        d3 = self.dotProduct(d3, cls_branch_max)
        d4 = self.dotProduct(d4, cls_branch_max)
        d5 = self.dotProduct(d5, cls_branch_max)

        return F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5)


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
        os.makedirs(os.path.join(output_folder_extra, 'label'))
        os.makedirs(os.path.join(output_folder_extra, 'image'))
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
    image_folder = './for_test/image'#'./VOCdevkit/VOC2007/JPEGImages'

    # 指定要写入的txt文件路径
    output_txt_file = './VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt'

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


def observe_model():
    from nets.TransUnet import get_transNet

    model = get_transNet(n_classes=2, img_size=512)
    # model=UNet_3Plus_DeepSup_CGM()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)
    # t=torch.rand(2,3, 256, 256)
    # x=model(t)
    # print(x.shape)
    summary(model, input_size=(2, 3, 512, 512))
    # summary(VisionTransformer(config_vit, img_size=img_size, num_classes=1),input_size=(2,3,256,256))
    #print(model)



def count_pos():
    path = r'D:\learning\UNNC 科研\data\DDTI\label'
    count = 0
    l = []
    for image in os.listdir(path):
        img = os.path.join(path, image)
        img = np.array(Image.open(img))
        if len(np.unique(img)) > 1:
            count += 1
            l.append(image)
    print(f'pos num: {count}')
    print(f'list:\n {l}')

def ob_weight():
    import datetime
    import os
    import time
    import torch.distributed as dist
    import numpy as np
    import torch
    from utils.Loss_utils import get_loss_weight, LossHistory, get_lr_scheduler, set_optimizer_lr
    from utils.metrics import CE_Loss, Dice_loss, Focal_Loss, f_score
    #     -------------------------------------------#
    #       权值文件的下载请看README
    #       权值和主干特征提取网络一定要对应
    #     -------------------------------------------#
    model = get_transNet(n_classes=2, img_size=256).train()
    # original_weights = model.state_dict()
    model_path = './model_data/pretrained_weight.pth'
    # pretrained_dict = torch.load(model_path)
    # # 加快模型训练的效率
    # print('Loading weights into state dict...')
    #
    model_dict = model.state_dict()

    # model.load_from(pretrained_dict)
    # loaded_weights = model.state_dict()
    # changed_weights = []
    # unchanged_weights = []
    for param in model.parameters():
        param.requires_grad = False
    print('first')
    # 观察权值矩阵！
    def print_requires_grad(module, prefix=''):
        for name, child in module.named_children():
            new_prefix = f"{prefix}.{name}" if prefix else name
            print_requires_grad(child, new_prefix)

        for name, param in module.named_parameters(prefix=prefix):
            print(f"{name}: {param.requires_grad}")

    # 使用这个函数遍历模型的所有子模块
    # print_requires_grad(model)
    print('second check')
    traverse_unfreeze_block(model,["cbam", "decoder", 'ASPP_unit3', 'segmentation_head', 'aspp_cbam'] )
    # for param in model.parameters():
    #     param.requires_grad = True
    # print_requires_grad(model)
    #
    # load_key, no_load_key, temp_dict = [], [], {}
    # for k, v in pretrained_dict.items():
    #     print(k)
    #     if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
    #         load_key.append(k)
    #     else:
    #         no_load_key.append(k)
    #
    # model_dict.update(temp_dict)
    # model.load_state_dict(model_dict)
    # for k in no_load_key:
    #     print(k)
    #
    # print(model)
    # # ------------------------------------------------------#
    # #   显示没有匹配上的Key
    # # ------------------------------------------------------#
    # print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    # print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    # print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

def preprocess():
    setlabel=2
    folder='malignant'  # benign:1, 'malignant':2
    path=os.path.join(r'D:\learning\UNNC 科研\data\BUSI\Dataset_BUSI_with_GT',folder)
    image_path=os.path.join(r'D:\learning\UNNC 科研\data\BUSI\Dataset_BUSI_with_GT\output','image')
    label_path=os.path.join(r'D:\learning\UNNC 科研\data\BUSI\Dataset_BUSI_with_GT\output','label')
    output_image_path=os.path.join(r'D:\learning\UNNC 科研\data\BUSI\Dataset_BUSI_with_GT','image')
    output_label_path=os.path.join(r'D:\learning\UNNC 科研\data\BUSI\Dataset_BUSI_with_GT','label')
    #
    # image_list=[]
    # for index,label in enumerate(os.listdir(path)):
    #     if label.endswith(').png'):
    #         image_list.append(label)
    for image in os.listdir(image_path):
        # # 寻找label并且合并他们
        # img=Image.open(os.path.join(path,image))
        # image_name=image.split('.')[0]
        # label_name=image_name+'_mask.png'
        # label=np.array(Image.open(os.path.join(path,label_name))).astype(np.uint8)*setlabel
        # if os.path.exists(os.path.join(path,label_name.replace('mask','mask_1'))):
        #     label1=np.array(Image.open(os.path.join(path,label_name).replace('mask','mask_1'))).astype(np.uint8)
        #     if len(label1.shape)!=2:
        #         label1=label1[:,:,0]
        #         label1[label1==255]=setlabel
        #     else:
        #         label1 = label1 * setlabel
        #
        #     label= label + label1
        # if os.path.exists(os.path.join(path, label_name.replace('mask', 'mask_2'))):
        #     label2 = np.array(Image.open(os.path.join(path, label_name).replace('mask', 'mask_2'))).astype(np.uint8)
        #     if len(label2.shape)!=2:
        #         label2=label2[:,:,0]
        #         label2[label2==255]=setlabel
        #     else:
        #         label2=label2*setlabel
        #
        #     label = label + label2
        # Image.fromarray(label).save(os.path.join(output_label_path,image))
        shutil.copy(os.path.join(image_path,image),os.path.join(output_image_path,image))
        shutil.copy(os.path.join(label_path, image), os.path.join(output_label_path, image))

def test():
    path=r'D:\learning\UNNC 科研\DeepLabV3Plus-Pytorch\best_deeplabv3plus_mobilenet_voc_os16.pth'
    pw=torch.load(path,map_location=torch.device('cpu'))['model_state']
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pw.items():
        print(k)


if __name__ == '__main__':
    # observe_model()
    # main()
    test()
    #txt()
    # count_pos()
    # ob_weight()
    # preprocess()






