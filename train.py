import datetime
import os
import time
import torch.distributed as dist
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tqdm import tqdm

from nets.TransUnet import get_transNet
from torch.utils.data import DataLoader
from dataloader import unetDataset, unet_dataset_collate
from utils.Loss_utils import get_loss_weight, LossHistory
from utils.metrics import CE_Loss, Dice_loss, Focal_Loss, f_score


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(model_train,model,loss_history, optimizer, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda, aux_branch, num_classes, dice_loss, focal_loss, local_rank=0, cls_weights=True):

    total_loss = 0
    total_f_score = 0

    val_toal_loss = 0
    val_total_f_score = 0

    start_time = time.time()

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_size:
            break
        imgs, pngs, labels = batch

        with torch.no_grad():
            imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
            pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
            labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))

            if cls_weights is False:
                # 类别均衡
                cls_weights = np.ones([num_classes],np.float32)
            else:
                cls_weights = get_loss_weight(num_classes, pngs)

            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                cls_weights=torch.tensor(cls_weights).cuda(local_rank)

        optimizer.zero_grad()
        #-------------------------------#
        #   判断是否使用辅助分支并回传
        #-------------------------------#
        # not use now
        if aux_branch:
            aux_outputs, outputs = model_train(imgs)
            aux_loss  = CE_Loss(aux_outputs, pngs, num_classes = num_classes)
            main_loss = CE_Loss(outputs, pngs, num_classes = num_classes)
            loss      = aux_loss + main_loss
            if dice_loss:
                aux_dice  = Dice_loss(aux_outputs, labels)
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + aux_dice + main_dice

        else:
            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, cls_weights=cls_weights,num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, cls_weights=cls_weights, num_classes=num_classes)
            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

        with torch.no_grad():
            #-------------------------------#
            #   计算f_score
            #-------------------------------#
            _f_score = f_score(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        waste_time = time.time() - start_time
        if local_rank==0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score'   : total_f_score / (iteration + 1),
                                's/step'    : waste_time,
                                'lr'        : get_lr(optimizer)})

            pbar.update(1)

    start_time = time.time()
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()

    for iteration, batch in enumerate(genval):
        if iteration >= epoch_size_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
            pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
            labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
            if cls_weights is False:
                # 类别均衡
                cls_weights = np.ones([num_classes],np.float32)
            else:
                cls_weights = get_loss_weight(num_classes, pngs)

            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                cls_weights=torch.tensor(cls_weights).cuda(local_rank)
            #-------------------------------#
            #   判断是否使用辅助分支 not use
            #-------------------------------#
            if aux_branch:
                aux_outputs, outputs = model_train(imgs)
                aux_loss  = CE_Loss(aux_outputs, pngs, num_classes = num_classes)
                main_loss = CE_Loss(outputs, pngs, num_classes = num_classes)
                val_toal_loss  = aux_loss + main_loss
                if dice_loss:
                    aux_dice  = Dice_loss(aux_outputs, labels)
                    main_dice = Dice_loss(outputs, labels)
                    val_toal_loss  = val_toal_loss + aux_dice + main_dice
            else:
                outputs = model_train(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, cls_weights=cls_weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, cls_weights=cls_weights, num_classes=num_classes)
                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice
            #-------------------------------#
            #   计算f_score
            #-------------------------------#
            _f_score = f_score(outputs, labels)

            val_toal_loss += loss.item()
            val_total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_toal_loss / (iteration + 1),
                                'f_score'   : val_total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_size, val_toal_loss / epoch_size_val)
        # eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_size, val_toal_loss / epoch_size_val))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % 5 == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
            (epoch + 1), total_loss / epoch_size, val_toal_loss / epoch_size_val)))

        if len(loss_history.val_loss) <= 1 or (val_toal_loss / epoch_size_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
    # print('Finish Validation')
    # print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    # print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))
    #
    # print('Saving state, iter:', str(epoch + 1))
    # totalBig_loss = ('%.4f' % (total_loss / (epoch_size + 1)))
    # val_loss1232 = ('%.4f' % (val_toal_loss / (epoch_size_val + 1)))
    # file_handle2 = open('train_loss.csv', mode='a+')
    #
    # file_handle2.write(totalBig_loss + ',' + val_loss1232 + '\n')
    # file_handle2.close()
    # score = ('%.4f' % (val_total_f_score / (iteration + 1)))
    # file_handle2 = open('acc.csv', mode='a+')
    # file_handle2.write(score + '\n')
    # file_handle2.close()
    # torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    # (epoch + 1), total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))


if __name__ == "__main__":
    inputs_size = [512, 512, 3]
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
    cls_weights=True
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
    #---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    #---------------------------------------------------------------------#
    sync_bn         = False
    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    print(f'ngpus: {ngpus_per_node}')
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0

    model = get_transNet(n_classes=NUM_CLASSES, img_size=inputs_size[0]).train()

    # -------------------------------------------#
    #   权值文件的下载请看README
    #   权值和主干特征提取网络一定要对应
    # -------------------------------------------#
    # model_path = r"model_data/pspnet_mobilenetv2.pth"
    # # 加快模型训练的效率
    # print('Loading weights into state dict...')
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path, map_location=device)
    # load_key, no_load_key, temp_dict = [], [], {}
    # for k, v in pretrained_dict.items():
    #     if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
    #         temp_dict[k] = v
    #         load_key.append(k)
    #     else:
    #         no_load_key.append(k)
    # model_dict.update(temp_dict)
    # model.load_state_dict(model_dict)
    # # ------------------------------------------------------#
    # #   显示没有匹配上的Key
    # # ------------------------------------------------------#
    # if local_rank == 0:
    #     print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    #     print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    #     print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    #----------------------#
    #   记录Loss
    #----------------------#
    save_dir='logs'
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=inputs_size[:2])
    else:
        loss_history    = None

    model_train = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   多卡平行运行
            #----------------------------#
            net = model.cuda(local_rank)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], find_unused_parameters=True)
        else:
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
        Interval_Epoch = 300
        Batch_size = 2
        # set opt
        optimizer = optim.Adam(model.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        train_dataset = unetDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = unetDataset(val_lines, inputs_size, NUM_CLASSES, False)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            # batch_size      = Batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen     = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate,sampler=train_sampler)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate,sampler=val_sampler)

        epoch_size = max(1, len(train_lines) // Batch_size)
        epoch_size_val = max(1, len(val_lines) // Batch_size)

        for epoch in range(Init_Epoch, Interval_Epoch):

            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler)
            if distributed:
                train_sampler.set_epoch(epoch)

            fit_one_epoch(model_train,model,loss_history, optimizer,epoch, epoch_size, epoch_size_val, gen, gen_val, Interval_Epoch, Cuda, aux_branch,num_classes=NUM_CLASSES,focal_loss=focal_loss,dice_loss=dice_loss,cls_weights=cls_weights,local_rank=local_rank)
            lr_scheduler.step()

            if distributed:
                dist.barrier()
    if local_rank == 0:
        loss_history.writer.close()
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


