import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from nets.TransUnet import get_transNet
from torch.utils.data import DataLoader
from dataloader import unetDataset, unet_dataset_collate
from utils.fit_one import fit_one_epoch

if __name__ == "__main__":
    inputs_size = [256, 256, 3]
    log_dir = "logs/"
    # ---------------------#
    #   分类个数+1
    #   2+1
    # ---------------------#
    NUM_CLASSES = 2
    # --------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    # ---------------------------------------------------------------------#
    dice_loss = True
    focal_loss= True
    # -------------------------------#
    #   主干网络预训练权重的使用
    #
    # -------------------------------#
    pretrained = False
    backbone = "ECAresnet"
    # ---------------------#
    #   是否使用辅助分支
    #   会占用大量显存
    # ---------------------#
    aux_branch = False
    # ---------------------#
    #   下采样的倍数
    #   8和16
    # ---------------------#
    downsample_factor = 16
    # -------------------------------#
    #   Cuda的使用
    # -------------------------------#
    Cuda = True

    model = get_transNet(n_classes=NUM_CLASSES, img_size=inputs_size[0]).train()

    # -------------------------------------------#
    #   权值文件的下载请看README
    #   权值和主干特征提取网络一定要对应
    # -------------------------------------------#
    # model_path = r"model_data/pspnet_mobilenetv2.pth"
    # # 加快模型训练的效率
    # print('Loading weights into state dict...')
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # print('Finished!')

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 打开数据集的txt
    with open(r"VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt", "r") as f:
        train_lines = f.readlines()

    # 打开数据集的txt
    with open(r"VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt", "r") as f:
        val_lines = f.readlines()

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        lr = 1e-3
        Init_Epoch = 0
        Interval_Epoch = 1
        Batch_size = 5
        optimizer = optim.Adam(model.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        train_dataset = unetDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = unetDataset(val_lines, inputs_size, NUM_CLASSES, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate)

        epoch_size = max(1, len(train_lines) // Batch_size)
        epoch_size_val = max(1, len(val_lines) // Batch_size)

        for epoch in range(Init_Epoch, Interval_Epoch):
            fit_one_epoch(model, epoch, epoch_size, epoch_size_val, gen, gen_val, Interval_Epoch, Cuda, aux_branch,num_classes=NUM_CLASSES,focal_loss=focal_loss)
            lr_scheduler.step()

    # if True:
    #     lr = 1e-5
    #     Interval_Epoch = 50
    #     Epoch = 100
    #     Batch_size = 2
    #     optimizer = optim.Adam(model.parameters(),lr)
    #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)
    #
    #     train_dataset = unetDataset(train_lines, inputs_size, NUM_CLASSES, True)
    #     val_dataset = unetDataset(val_lines, inputs_size, NUM_CLASSES, False)
    #     gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
    #                             drop_last=True, collate_fn=unet_dataset_collate)
    #     gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=2,pin_memory=True,
    #                             drop_last=True, collate_fn=unet_dataset_collate)
    #
    #     epoch_size      = max(1, len(train_lines)//Batch_size)
    #     epoch_size_val  = max(1, len(val_lines)//Batch_size)
    #
    #     # for param in model.backbone.parameters():
    #     #     param.requires_grad = True
    #
    #     for epoch in range(Interval_Epoch,Epoch):
    #         fit_one_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Epoch,Cuda,aux_branch)
    #         lr_scheduler.step()



