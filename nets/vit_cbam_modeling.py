# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from nets import vit_seg_configs as configs
from nets.transunet_modeling import *
from nets.vit_seg_modeling_resnet_skip import *

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，输入BCHW -> 输出 B*C*1*1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 可以看到channel得被reduction整除，否则可能出问题
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 得到B*C*1*1,然后转成B*C，才能送入到FC层中。
        y = self.fc(y).view(b, c, 1, 1)  # 得到B*C的向量，C个值就表示C个通道的权重。把B*C变为B*C*1*1是为了与四维的x运算。
        return x * y.expand_as(x)  # 先把B*C*1*1变成B*C*H*W大小，其中每个通道上的H*W个值都相等。*表示对应位置相乘


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print('max_out:',max_out.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print('avg_out:',avg_out.shape)
        a = torch.cat([max_out, avg_out], dim=1)
        # print('a:',a.shape)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        # print('spatial:',spatial_out.shape)
        x = spatial_out * x
        # print('x:',x.shape)
        return x


class CBAM_ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, atrous_rates=(6, 12, 18), bn_mom=0.1):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=atrous_rates[0], dilation=atrous_rates[0], bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=atrous_rates[1], dilation=atrous_rates[1], bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=atrous_rates[2], dilation=atrous_rates[2], bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # print('dim_in:',dim_in)
        # print('dim_out:',dim_out)
        self.cbam = CBAMLayer(channel=dim_out * 5)

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # print('feature:',feature_cat.shape)
        # 加入cbam注意力机制
        cbamaspp = self.cbam(feature_cat)
        result = self.conv_cat(cbamaspp)
        return result


class Embeddings_CBAM(Embeddings):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super().__init__(config, img_size, in_channels)

        self.hybrid_model = ResNetV2_CBAM_4skip(block_units=config.resnet.num_layers,
                                          width_factor=config.resnet.width_factor)


# not change module, so save here
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer_CBAM(Transformer):
    def __init__(self, config, img_size, vis):
        super().__init__(config, img_size, vis)
        self.embeddings = Embeddings_CBAM(config, img_size=img_size)  ######读懂####
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)

        return encoded, attn_weights, features


# DecoderBlock_CBAM
class DecoderBlock_CBAM(DecoderBlock):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            skip_channels,
            use_batchnorm)

        self.cbam = CBAMLayer(out_channels)

    def forward(self, x, skip=None):
        print('in', x.size())
        # x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam(x)
        x = self.up(x)
        return x


class DecoderBlock_SE(DecoderBlock):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            skip_channels,
            use_batchnorm)

        self.se = SELayer(out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return x


class DecoderCup_3skip(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels[1:]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip - 1 != 0:
            skip_channels = self.config.skip_channels[1:]
            for i in range(5 - self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #####变矩阵###########3
        # --------------------2,1024,768------2,512,32,32 if 512
        # ------------------从transformer变成cnn
        # ----------------------
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip - 2) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)

        return x


class DecoderCup_4skip(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = config.skip_channels[0]  # 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            # skip_channels[self.config.n_skip:] = [0] * len(skip_channels[self.config.n_skip:])
            # for i in range(4 - self.config.n_skip):  # re-select the skip channels according to n_skip
            #     skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = []

        for i in range(self.config.n_skip):
            blocks.append(DecoderBlock_CBAM(in_channels[i], out_channels[i], skip_channels[i]))

        # blocks = [
        #     DecoderBlock_CBAM(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
        #     zip(in_channels, out_channels, skip_channels)
        # ]
        self.blocks = nn.ModuleList(blocks)


class DecoderCup_4skip_CBAM_ASPP_CBAM(DecoderCup_4skip):
    def __init__(self, config):
        super().__init__(config)
        head_channels = config.skip_channels[0]  # 512
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            self.skip_channels = self.config.skip_channels
            # skip_channels[self.config.n_skip:] = [0] * len(skip_channels[self.config.n_skip:])
            # for i in range(4 - self.config.n_skip):  # re-select the skip channels according to n_skip
            #     skip_channels[3 - i] = 0

        else:
            self.skip_channels = [0, 0, 0, 0]
        blocks = []

        for i in range(self.config.n_skip):
            blocks.append(DecoderBlock_CBAM(in_channels[i], out_channels[i], self.skip_channels[i]))

        # blocks = [
        #     DecoderBlock_CBAM(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
        #     zip(in_channels, out_channels, skip_channels)
        # ]
        self.blocks = nn.ModuleList(blocks)
        self.cbam_aspp = CBAM_ASPP(head_channels, head_channels)

    def forward(self, hidden_states, features=None):
        # for f in range(len(features)):
        #     print(features[f].size())
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #####变矩阵###########3
        # --------------------2,1024,768------2,512,32,32 if 512
        # ------------------从transformer变成cnn
        # ----------------------
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        x = self.cbam_aspp(x)

        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)

        return x


class DecoderCup_4skip_ASPP_CBAM(DecoderCup_4skip):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            self.skip_channels = self.config.skip_channels
            for i in range(4 - self.config.n_skip):  # re-select the skip channels according to n_skip
                self.skip_channels[3 - i] = 0
        else:
            self.skip_channels = [0, 0, 0, 0]
        blocks = []

        for i in range(len(in_channels)):
            blocks.append(DecoderBlock_SE(in_channels[i], out_channels[i], self.skip_channels[i]))

        # blocks = [
        #     DecoderBlock_CBAM(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
        #     zip(in_channels, out_channels, skip_channels)
        # ]
        self.blocks = nn.ModuleList(blocks)
        self.aspp = ASPP(head_channels, head_channels)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #####变矩阵###########3
        # --------------------2,1024,768------2,512,32,32 if 512
        # ------------------从transformer变成cnn
        # ----------------------
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        x = self.aspp(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)

        return x


class DecoderCup_3skip_CBAM_ASPP_CBAM(DecoderCup_3skip):
    def __init__(self, config):
        super().__init__(config)
        head_channels = 512
        decoder_channels = config.decoder_channels[1:]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        if self.config.n_skip - 1 != 0:
            skip_channels = self.config.skip_channels[1:]
            for i in range(5 - self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]
        blocks = [
            DecoderBlock_CBAM(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
            zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.aspp = ASPP(head_channels, head_channels)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #####变矩阵###########3
        # --------------------2,1024,768------2,512,32,32 if 512
        # ------------------从transformer变成cnn
        # ----------------------
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        x = self.aspp(x)

        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip - 1) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)

        return x


class Vit_CBAM(VisionTransformer):
    def __init__(self, config, img_size=256, num_classes=21843, zero_head=False, vis=False, cgm=True):
        super(Vit_CBAM, self).__init__(config, img_size, num_classes, zero_head, vis, )
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer_CBAM(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

        # self.BinaryClassifier = ReducedBinaryClassifier(config.hidden_size, num_classes)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)

        logits = self.segmentation_head(x)

        return logits


class Vit_CBAM_ASPP(VisionTransformer):
    def __init__(self, config, img_size=256, num_classes=21843, zero_head=False, vis=False, cgm=True):
        super().__init__(config, img_size, num_classes, zero_head, vis, )
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer_CBAM(config, img_size, vis)
        self.decoder = DecoderCup_4skip_CBAM_ASPP_CBAM(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][config.n_skip-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

        # self.BinaryClassifier = ReducedBinaryClassifier(config.hidden_size, num_classes)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)

        logits = self.segmentation_head(x)

        return logits


class Vit_CBAM_3skip(VisionTransformer):
    def __init__(self, config, img_size=256, num_classes=21843, zero_head=False, vis=False, cgm=True):
        super().__init__(config, img_size, num_classes, zero_head, vis, )
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer_CBAM(config, img_size, vis)
        self.decoder = DecoderCup_3skip_CBAM_ASPP_CBAM(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][config.n_skip-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

        # self.BinaryClassifier = ReducedBinaryClassifier(config.hidden_size, num_classes)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)

        logits = self.segmentation_head(x)

        return logits
