from __future__ import division, absolute_import, print_function

import copy

import numpy
import numpy as np
import torch
from torch import nn
from torch.nn import LayerNorm

from nets.transunet_modeling import DecoderBlock, Conv2dReLU, VisionTransformer, SegmentationHead, Embeddings, Block, \
    Transformer, Encoder
from nets.vit_seg_modeling_resnet_skip import ResNetV2_CBAM_4skip, ResNetV2_CBAM, SELayer


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
        # 加一层BN layer
        # self.BN = nn.BatchNorm2d(*)
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


class Embeddings_CBAM(Embeddings):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super().__init__(config, img_size, in_channels)
        if config.n_skip == 4:
            self.hybrid_model = ResNetV2_CBAM_4skip(block_units=config.resnet.num_layers,
                                                    width_factor=config.resnet.width_factor)
        else:
            self.hybrid_model = ResNetV2_CBAM(block_units=config.resnet.num_layers,
                                              width_factor=config.resnet.width_factor)


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


class DecoderBlock_4skip_CBAM(DecoderBlock):
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
        # print('in', x.size())
        # x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam(x)
        x = self.up(x)
        return x


class DecoderBlock_4skip(DecoderBlock):
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

    def forward(self, x, skip=None):
        # x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.up(x)
        return x


class DecoderBlock_3skip_CBAM(DecoderBlock):
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
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam(x)
        # x = self.up(x)
        return x


class DecoderBlock_3skip(DecoderBlock):
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

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.up(x)
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
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip - 1 != 0:
            self.skip_channels = self.config.skip_channels
        else:
            self.skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock_3skip(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
            zip(in_channels, out_channels, self.skip_channels)
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
                skip = features[i] if (i < self.config.n_skip) else None
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
            self.skip_channels = self.config.skip_channels
            # skip_channels[self.config.n_skip:] = [0] * len(skip_channels[self.config.n_skip:])
            # for i in range(4 - self.config.n_skip):  # re-select the skip channels according to n_skip
            #     skip_channels[3 - i] = 0

        else:
            self.skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock_4skip(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
            zip(in_channels, out_channels, self.skip_channels)
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
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)

        return x


class DecoderCup_4skip_CBAM(DecoderCup_4skip):
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
            blocks.append(DecoderBlock_4skip_CBAM(in_channels[i], out_channels[i], self.skip_channels[i]))

        # blocks = [
        #     DecoderBlock_CBAM(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
        #     zip(in_channels, out_channels, skip_channels)
        # ]
        self.blocks = nn.ModuleList(blocks)

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

        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)

        return x


class DecoderCup_3skip_CBAM(DecoderCup_3skip):
    def __init__(self, config):
        super().__init__(config)
        head_channels = 512
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock_3skip_CBAM(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
            zip(in_channels, out_channels, skip_channels)
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
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)

        return x


class Vit_CBAM(VisionTransformer):
    def __init__(self, config, img_size=256, num_classes=21843, zero_head=False, vis=False, cgm=True):
        super().__init__(config, img_size, num_classes, zero_head, vis, )
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer_CBAM(config, img_size, vis)
        if config.n_skip == 3:
            self.decoder = DecoderCup_3skip(config)
        else:
            self.decoder = DecoderCup_4skip(config)
        self.segmentation_head = SegmentationHead(
            in_channels=16,
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


class Vit_CBAM_CBAM(VisionTransformer):
    def __init__(self, config, img_size=256, num_classes=21843, zero_head=False, vis=False, cgm=True):
        super().__init__(config, img_size, num_classes, zero_head, vis, )
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer_CBAM(config, img_size, vis)
        if config.n_skip == 3:
            self.decoder = DecoderCup_3skip_CBAM(config)
        else:
            self.decoder = DecoderCup_4skip_CBAM(config)
        self.segmentation_head = SegmentationHead(
            in_channels=16,
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
