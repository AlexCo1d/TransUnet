import cv2
from numpy.core.numeric import False_, True_
from nets.TransUnet import get_transNet
from torch import nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import colorsys
import torch
import copy
import os

from utils.visualization import blend_images


class uNet(object):
    # -----------------------------------------#
    #   注意修改model_path、num_classes
    #   和backbone
    #   使其符合自己的模型
    # -----------------------------------------#
    _defaults = {
        "model_path": './logs/Epoch300-Total_Loss0.5099-Val_Loss0.5715.pth',
        "model_image_size": (512, 512, 3),
        "backbone": "ECAresnet",
        "downsample_factor": 16,
        "num_classes": 2,
        "cuda": True,
        # --------------------------------#
    }

    # ---------------------------------------------------#
    #   初始化UNET
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.net = get_transNet(n_classes=self.num_classes, img_size=self.model_image_size[0])
        self.net = self.net.eval()

        state_dict = torch.load(self.model_path)
        self.net.load_state_dict(state_dict, strict=False)
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (0, 255, 0), (0, 128, 0), (128, 128, 0), (128, 0, 128), (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                           (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128), (128, 64, 12)]
        else:
            # 画框设置不同的颜色
            hsv_tuples = [(x / 32, 1., 1.)
                          for x in range(32)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))

    def letterbox_image(self, image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image, nw, nh

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, mix=0):
        self.mix = mix
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image, nw, nh = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        images = np.array(image, np.float32) / 255.0
        images = np.expand_dims(np.transpose(images, (2, 0, 1)), 0)

        with torch.no_grad():
            images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            pr = pr[int((self.model_image_size[0] - nh) // 2):int((self.model_image_size[0] - nh) // 2 + nh),
                 int((self.model_image_size[1] - nw) // 2):int((self.model_image_size[1] - nw) // 2 + nw)]
            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)
        if self.mix == 2:
            # 混合图像
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c) * (self.colors[c][0])).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c) * (self.colors[c][1])).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c) * (self.colors[c][2])).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            # ------------------------------------------------#
            #   将新图片转换成Image的形式
            # ------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
            # ------------------------------------------------#
            #   将新图与原图及进行混合
            # ------------------------------------------------#
            image = blend_images(image,old_img, 0.5)
        elif self.mix == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image = Image.fromarray(np.uint8(seg_img))
        elif self.mix == 0:
            image = Image.fromarray(np.uint8(pr)).resize((orininal_w, orininal_h), Image.NEAREST)
            # ------------------------------------------------#
            #   将新图片转换成Image的形式
            # ------------------------------------------------#
            # seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            # image   = Image.fromarray(np.uint8(seg_img))

        return image
