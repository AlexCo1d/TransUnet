import random

import torch
from math import exp
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.autograd import Variable
from random import shuffle
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image
import cv2
from torch.nn import functional as F
'''
    git config --global --unset https.proxy
    git config --global --unset http.proxy
'''

def _one_hot_encoder(pngs,num_classes):
    c,h,w=pngs.size()
    temp=pngs.cpu().numpy()
    seg_labels = np.eye(num_classes + 1)[temp.reshape([-1])]
    # print(f'seg_labels{seg_labels.shape}')
    seg_labels = seg_labels.reshape((int(h), int(w), num_classes + 1))
    return seg_labels.float()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True