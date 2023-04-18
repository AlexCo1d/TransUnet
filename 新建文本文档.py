import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import shutil
path=r'D:/PycharmProjects/unet/transunet 癌细胞/transunet/VOCdevkit/VOC2007/SegmentationClass/'
newpath=r'D:\PycharmProjects\unet\transunet 癌细胞\transunet\VOCdevkit\VOC2007\lable'
def turnto24(path):
  fileList = []
  files = os.listdir(path)
  i=0
  for f in files:
    imgpath = path + '/' +f
    img=Image.open(f).convert('RGB')
    dirpath = newpath
    file_name, file_extend = os.path.splitext(f)
    dst = os.path.join(os.path.abspath(dirpath), file_name + '.jpg')
    img.save(dst)
turnto24(path)