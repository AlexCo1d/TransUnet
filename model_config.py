class Config:
    def __init__(self):
        self.inputs_size = [512, 512, 3]
        self.log_dir = "logs/"
        # ---------------------#
        #   分类个数+1
        #   2+1
        # ---------------------#
        self.NUM_CLASSES = 2
        # --------------------------------------------------------------------#
        #   建议选项：
        #   种类少（几类）时，设置为True
        #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
        #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
        # ---------------------------------------------------------------------#

        # --------------#
        # BATCH_SIZE
        # --------------#
        self.Batch_size = 6
        # (focal or ce) + dice
        self.focal_loss = True
        self.cls_weights = True
        # -------------------------------#
        #   主干网络预训练权重的使用
        #
        # -------------------------------#
        self.pretrained = True
        # backbone = "ECAresnet"
        # ---------------------#
        #   是否使用辅助分支
        #   会占用大量显存
        # ---------------------#
        self.aux_branch = False
        # ---------------------#
        #   下采样的倍数
        #   8和16
        # ---------------------#
        self.downsample_factor = 16
        # -------------------------------#
        #   Cuda的使用
        # -------------------------------#
        self.Cuda = True
        # ---------------------------------------------------------------------#
        #   distributed     用于指定是否使用单机多卡分布式运行
        #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
        #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
        #   DP模式：
        #       设置            distributed = False
        #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
        #   DDP模式：
        #       设置            distributed = True
        #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
        # ---------------------------------------------------------------------#
        self.distributed = True
        # ---------------------------------------------------------------------#
        #   sync_bn     是否使用sync_bn，DDP模式多卡可用
        # ---------------------------------------------------------------------#
        self.sync_bn = True
        self.model_path = './model_data/pretrained_weight.pth'
        # no_load_dict,加载预训练时不加载解码器部分
        self.no_load_dict = ['decoder', 'segmentation_head']
        self.frozen_modules = ["cbam", "decoder", 'ASPP_unit1', 'ASPP_unit2', 'ASPP_unit3', 'segmentation_head',
                          'cls']  # removed: cls

        # ----------------------#
        #   记录Loss
        # ----------------------#
        self.save_dir = 'logs'
        self.Init_Epoch = 0
        self.Interval_Epoch = 120
        # 设置冻结的epoch
        self.Freeze_Epoch = 40

