import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],
            ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                '''这个 forward 函数中的 if 语句的目的是检查当前特征图 x 的空间尺寸（宽度和高度）是否等于期望的 right_size。如果不等于， 它将创建一个填充零的新特征图 
                feat，并将原始特征图 x 复制到这个新特征图的左上角，以使其空间尺寸与期望的尺寸相符。这样做主要是为了确保特征图的尺寸在之后的操作中是正确的，例如在特征金字塔网络中对特征图进行上采样和融合时。 '''
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        features.append(x)
        # print('features:',len(features))
        return x, features[::-1]


class ASPP(nn.Module):
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
        # self.cbam = CBAMLayer(channel=dim_out * 5)

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
        result = self.conv_cat(feature_cat)
        return result


class ResNetV2_ASPP(ResNetV2):
    def __init__(self, block_units, width_factor):
        super(ResNetV2_ASPP, self).__init__(block_units, width_factor)
        width = int(64 * width_factor)
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2]+1)] +
                [('ASPP_unit10', ASPP(width * 16, width * 16, atrous_rates=(6, 12, 18)))],
            ))),
        ]))


class ResNetV2_ASPP_1(ResNetV2):
    def __init__(self, block_units, width_factor):
        super(ResNetV2_ASPP_1, self).__init__(block_units, width_factor)
        width = int(64 * width_factor)
        self.width = width
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2]-1)] +
                [('ASPP_unit3', ASPP(width * 16, width * 16, atrous_rates=(6, 12, 18)))] +
                [(f'unit9', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4))],
            ))),
        ]))


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


class ASPP_SE(nn.Module):
    def __init__(self, dim_in, dim_out, atrous_rates=(6,12,18), bn_mom=0.1):
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
        self.se = SELayer(dim_out * 5)

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
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear',align_corners=True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # print('feature:',feature_cat.shape)
        seaspp1 = self.se(feature_cat)  # 加入通道注意力机制
        # print('seaspp1:',seaspp1.shape)
        se_feature_cat = seaspp1 * feature_cat
        result = self.conv_cat(se_feature_cat)
        # print('result:',result.shape)
        return result


class CBAM_ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, atrous_rates=(6,12,18), bn_mom=0.1):
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


class PreActBottleneck_CBAM(PreActBottleneck):
    """Pre-activation (v2) bottleneck block.
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__(cin, cout, cmid, stride)
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAMLayer(cmid)
        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.cbam(y)
        y = self.gn3(self.conv3(y))
        y = self.relu(residual + y)
        return y


class PreActBottleneck_SE(PreActBottleneck):
    """Pre-activation (v2) bottleneck block.
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__(cin, cout, cmid, stride)
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.se=SELayer(cmid)
        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.se(y)
        y = self.gn3(self.conv3(y))
        y = self.relu(residual + y)
        return y

class ResNetV2_CBAM(ResNetV2):
    def __init__(self, block_units, width_factor):
        super().__init__(block_units, width_factor)
        width = int(64 * width_factor)
        self.width = width
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit2', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width))] +
                [(f'unit3', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width))]

            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck_CBAM(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit2', PreActBottleneck_CBAM(cin=width * 8, cout=width * 8, cmid=width * 2))] +
                [(f'unit3', PreActBottleneck_CBAM(cin=width * 8, cout=width * 8, cmid=width * 2))] +
                [(f'unit4', PreActBottleneck_CBAM(cin=width * 8, cout=width * 8, cmid=width * 2))],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck_CBAM(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit2', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit3', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit4', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit5', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit6', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit7', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit8', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                # [('ASPP_unit3', ASPP(width * 16, width * 16, atrous_rates=(6, 12, 18)))] +
                [(f'unit9', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))]

            ))),
        ]))


class ResNetV2_ASPP_SE(ResNetV2):
    def __init__(self, block_units, width_factor):
        super().__init__(block_units, width_factor)
        width = int(64 * width_factor)
        self.width = width
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck_SE(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit2', PreActBottleneck_SE(cin=width * 4, cout=width * 4, cmid=width))] +
                [(f'unit3', PreActBottleneck_SE(cin=width * 4, cout=width * 4, cmid=width))]

            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck_SE(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit2', PreActBottleneck_SE(cin=width * 8, cout=width * 8, cmid=width * 2))] +
                [(f'unit3', PreActBottleneck_SE(cin=width * 8, cout=width * 8, cmid=width * 2))] +
                [(f'unit4', PreActBottleneck_SE(cin=width * 8, cout=width * 8, cmid=width * 2))],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck_SE(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit2', PreActBottleneck_SE(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit3', PreActBottleneck_SE(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit4', PreActBottleneck_SE(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit5', PreActBottleneck_SE(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit6', PreActBottleneck_SE(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit7', PreActBottleneck_SE(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit8', PreActBottleneck_SE(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                # [('ASPP_unit3', ASPP(in_channels=width * 16, out_channels=width * 16, atrous_rates=(6, 12, 18)))] +
                [(f'unit9', PreActBottleneck_SE(cin=width * 16, cout=width * 16, cmid=width * 4))],
            ))),
        ]))


class ResNetV2_SE_ASPP_SE(ResNetV2):
    def __init__(self, block_units, width_factor):
        super().__init__(block_units, width_factor)
        width = int(64 * width_factor)
        self.width = width
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit2', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width))] +
                [(f'unit3', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width))]

            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck_SE(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit2', PreActBottleneck_SE(cin=width * 8, cout=width * 8, cmid=width * 2))] +
                [(f'unit3', PreActBottleneck_SE(cin=width * 8, cout=width * 8, cmid=width * 2))] +
                [(f'unit4', PreActBottleneck_SE(cin=width * 8, cout=width * 8, cmid=width * 2))],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck_SE(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit2', PreActBottleneck_SE(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit3', PreActBottleneck_SE(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit4', PreActBottleneck_SE(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit5', PreActBottleneck_SE(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit6', PreActBottleneck_SE(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit7', PreActBottleneck_SE(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                # [(f'unit8', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [('ASPP_unit3', ASPP_SE(width * 16, width * 16, atrous_rates=(6, 12, 18)))] +
                [(f'unit9', PreActBottleneck_SE(cin=width * 16, cout=width * 16, cmid=width * 4))],
            ))),
        ]))


class ResNetV2_CBAM_ASPP_CBAM(ResNetV2):
    def __init__(self, block_units, width_factor):
        super().__init__(block_units, width_factor)
        width = int(64 * width_factor)
        self.width = width
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit2', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width))] +
                [(f'unit3', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width))]

            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck_CBAM(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit2', PreActBottleneck_CBAM(cin=width * 8, cout=width * 8, cmid=width * 2))] +
                [(f'unit3', PreActBottleneck_CBAM(cin=width * 8, cout=width * 8, cmid=width * 2))] +
                [(f'unit4', PreActBottleneck_CBAM(cin=width * 8, cout=width * 8, cmid=width * 2))],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck_CBAM(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit2', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit3', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit4', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit5', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit6', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit7', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                [(f'unit8', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))] +
                # [('ASPP_unit3', CBAM_ASPP(width * 16, width * 16, atrous_rates=(6, 12, 18)))] +
                [(f'unit9', PreActBottleneck_CBAM(cin=width * 16, cout=width * 16, cmid=width * 4))],
            ))),
        ]))