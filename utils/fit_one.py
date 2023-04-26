import time

import torch
from torch.autograd import Variable
from tqdm import tqdm

from train import optimizer, NUM_CLASSES, dice_loss, model
from utils.metrics import CE_Loss, Dice_loss, f_score, Focal_Loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda,aux_branch,num_classes,focal_loss,cls_weights):
    net = net.train()
    total_loss = 0
    total_f_score = 0

    val_toal_loss = 0
    val_total_f_score = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

            #-------------------------------#
            #   判断是否使用辅助分支并回传
            #-------------------------------#
            optimizer.zero_grad()
            if aux_branch:
                aux_outputs, outputs = net(imgs)
                aux_loss  = CE_Loss(aux_outputs, pngs, num_classes = NUM_CLASSES)
                main_loss = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                loss      = aux_loss + main_loss
                if dice_loss:
                    aux_dice  = Dice_loss(aux_outputs, labels)
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + aux_dice + main_dice

            else:
                outputs = net(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, cls_weights=cls_weights,num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, num_classes=num_classes)
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
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score'   : total_f_score / (iteration + 1),
                                's/step'    : waste_time,
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                #-------------------------------#
                #   判断是否使用辅助分支
                #-------------------------------#
                if aux_branch:
                    aux_outputs, outputs = net(imgs)
                    aux_loss  = CE_Loss(aux_outputs, pngs, num_classes = NUM_CLASSES)
                    main_loss = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                    val_loss  = aux_loss + main_loss
                    if dice_loss:
                        aux_dice  = Dice_loss(aux_outputs, labels)
                        main_dice = Dice_loss(outputs, labels)
                        val_loss  = val_loss + aux_dice + main_dice

                else:
                    outputs  = net(imgs)
                    val_loss = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                    if dice_loss:
                        main_dice = Dice_loss(outputs, labels)
                        val_loss  = val_loss + main_dice

                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

                val_toal_loss += val_loss.item()
                val_total_f_score += _f_score.item()


            pbar.set_postfix(**{'total_loss': val_toal_loss / (iteration + 1),
                                'f_score'   : val_total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    totalBig_loss = ('%.4f' % (total_loss / (epoch_size + 1)))
    val_loss1232 = ('%.4f' % (val_toal_loss / (epoch_size_val + 1)))
    file_handle2 = open('train_loss.csv', mode='a+')

    file_handle2.write(totalBig_loss + ',' + val_loss1232 + '\n')
    file_handle2.close()
    score = ('%.4f' % (val_total_f_score / (iteration + 1)))
    file_handle2 = open('acc.csv', mode='a+')
    file_handle2.write(score + '\n')
    file_handle2.close()
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    (epoch + 1), total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))