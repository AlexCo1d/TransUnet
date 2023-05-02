import numpy as np
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
        print(name)
        if name in freeze_block:
            if local_rank==0:
                print(f'find block {name}! set rg TRUE!')
            for param in child.parameters():
                param.requires_grad = True

        # 继续递归遍历子模块的子模块
        traverse_unfreeze_block(child,freeze_block=freeze_block,local_rank=local_rank)


def load_pretrained_weights(model,model_path,no_load_dict,local_rank):
    # 加快模型训练的效率
    print('Loading weights into state dict...')
    model_dict = model.state_dict()
    device = torch.device("cuda", local_rank)
    pretrained_dict = torch.load(model_path, map_location=device)

    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            flag=False
            for prefix in no_load_dict:
                if k.startswith(prefix):
                    flag=True
            if flag:
                no_load_key.append(k)
            else:
                temp_dict[k] = v
                load_key.append(k)

        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    # ------------------------------------------------------#
    #   显示没有匹配上的Key
    # ------------------------------------------------------#
    if local_rank == 0:
        print("\nSuccessful Load Key Num:", len(load_key))
        for key in load_key:
            print(key)
        print("\nFail To Load Key num:", len(no_load_key))
        for key in no_load_key:
            print(key)
        # print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
        # 定义需要冻结的模块名称