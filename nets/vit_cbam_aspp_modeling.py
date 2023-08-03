# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.transunet_modeling import *
from nets.vit_cbam_modeling import DecoderBlock_4skip_CBAM, DecoderBlock_3skip_CBAM, DecoderCup_3skip, DecoderCup_4skip, \
    CBAMLayer, Transformer_CBAM
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


###  CBAM_ASPP和CBAM_Layer的区别是什么
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


# not change module, so save here


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
            blocks.append(DecoderBlock_4skip_CBAM(in_channels[i], out_channels[i], self.skip_channels[i]))

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


class DecoderCup_3skip_CBAM_ASPP_CBAM(DecoderCup_3skip):
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
        self.cbam_aspp = ASPP(head_channels, head_channels)

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
        x = self.cbam_aspp(x)

        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)

        return x


class Vit_CBAM_ASPP(VisionTransformer):
    def __init__(self, config, img_size=256, num_classes=21843, zero_head=False, vis=False, cgm=True):
        super().__init__(config, img_size, num_classes, zero_head, vis, )
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer_CBAM(config, img_size, vis)
        if config.n_skip == 3:
            self.decoder = DecoderCup_3skip_CBAM_ASPP_CBAM(config)
        else:
            self.decoder = DecoderCup_4skip_CBAM_ASPP_CBAM(config)
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
