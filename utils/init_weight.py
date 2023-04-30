import torch
import torch.nn as nn
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data, gain=1)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data, gain=1)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Conv2d):
        init.orthogonal_(m.weight.data, gain=1)
    elif isinstance(m, nn.Linear):
        init.orthogonal_(m.weight.data, gain=1)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def traverse_unfreeze_block(model,freeze_block,local_rank=0):
    for name, child in model.named_children():
        # 如果需要，可以在此处对子模块执行某些操作
        if name in freeze_block:
            if local_rank==0:
                print(f'find block {name}! set rg TRUE!')
            for param in child.parameters():
                param.requires_grad = True

        # 继续递归遍历子模块的子模块
        traverse_unfreeze_block(child,freeze_block=freeze_block,local_rank=local_rank)