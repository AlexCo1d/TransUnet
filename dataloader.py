from random import shuffle
import random
import numpy as np
import torch
import torch.nn as nn
import math
from imgaug import augmenters as iaa
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import torchvision.transforms as transforms
import cv2

from train_config import config


def letterbox_image(image, label, size):
    label = Image.fromarray(np.array(label))
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    label = label.resize((nw, nh), Image.NEAREST)
    new_label = Image.new('L', size, (0))
    new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))

    return np.array(new_image, np.uint8), np.array(new_label, np.uint8)


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


class unetDataset(Dataset):
    def __init__(self, train_lines, image_size, num_classes, random_data):
        super(unetDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.num_classes = num_classes
        self.random_data = random_data

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=.7, val=.3, fil=.4):
        label = Image.fromarray(np.array(label))
        # crop image or not
        if rand() < .6:
            height, width, _ = np.array(image).shape
            pc = rand(0.8, 1)
            crop_height = min(int(pc * input_shape[0]), height)
            crop_width = min(int(pc * input_shape[0]), width)
            left = random.randint(0, width - crop_width)
            upper = random.randint(0, height - crop_height)
            image = image.crop((left, upper, left + crop_width, upper + crop_height))
            label = label.crop((left, upper, left + crop_width, upper + crop_height))

        # resize image
        h, w = input_shape
        rand_jit1 = rand(1 - jitter, 1 + jitter)
        rand_jit2 = rand(1 - jitter, 1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2

        scale = rand(0.5, 1.5)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        label = label.convert("L")

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), 0)
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # flip image or not
        if rand() < .6:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if rand() < .6:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)

        # rotate image:
        if rand() < .3:
            angle = np.random.uniform(3, 10)
            image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(128, 128, 128))
            label = label.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=0)

        # gaussian filter and sharpen filter choose one
        if rand() < .4:
            if rand() < .5:
                radius = np.random.uniform(1 - fil, 1)
                image = image.filter(ImageFilter.GaussianBlur(radius))
            else:
                factor = np.random.uniform(1, 1 + fil)
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(factor)

        image_data = np.array(image, np.uint8)
        label = np.array(label, np.uint8)

        # distort image slightly
        if rand() < .3:
            augmenter = iaa.ElasticTransformation(alpha=8, sigma=5)
            image_data, label = augmenter(images=[image_data, label])

        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        if rand() < .6:
            r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
            # ---------------------------------#
            #   将图像转到HSV上
            # ---------------------------------#
            hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
            dtype = image_data.dtype
            # ---------------------------------#
            #   应用变换
            # ---------------------------------#
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # image_data = image
        # if rand() < .7:
        #     hue = rand(-hue, hue)
        #     sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        #     val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        #     x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        #     x[..., 0] += hue * 360
        #     x[..., 0][x[..., 0] > 1] -= 1
        #     x[..., 0][x[..., 0] < 0] += 1
        #     x[..., 1] *= sat
        #     x[..., 2] *= val
        #     x[x[:, :, 0] > 360, 0] = 360
        #     x[:, :, 1:][x[:, :, 1:] > 1] = 1
        #     x[x < 0] = 0
        #     image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        return image_data, label

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)

        annotation_line = self.train_lines[index]
        name = annotation_line.replace('\n','')
        # 从文件中读取图像
        jpg = Image.open(r"./VOCdevkit/VOC2007/JPEGImages" + '/' + name + config.image_type).convert('RGB')
        png = Image.open(r"./VOCdevkit/VOC2007/SegmentationClass" + '/' + name + ".png").convert('L')

        if self.random_data:
            jpg, png = self.get_random_data(jpg, png, (int(self.image_size[1]), int(self.image_size[0])))
            # print('!!', np.array(jpg).shape, np.array(png).shape)
        else:
            jpg, png = letterbox_image(jpg, png, (int(self.image_size[1]), int(self.image_size[0])))

        # 从文件中读取图像
        png[png >= self.num_classes] = self.num_classes

        # 转化成one_hot的形式
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        # print(f'seg_labels{seg_labels.shape}')
        seg_labels = seg_labels.reshape((int(self.image_size[1]), int(self.image_size[0]), self.num_classes + 1))

        # 归一化和维度交换
        jpg = np.transpose(np.array(jpg), [2, 0, 1]) / 255
        # jpg image,
        # png label ,
        # seg_labels: one-hot encoded label
        return jpg, png, seg_labels


# DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = np.array(images)
    pngs = np.array(pngs)
    seg_labels = np.array(seg_labels)
    return images, pngs, seg_labels
