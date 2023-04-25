# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:29:02 2020

@author: 12624
"""

import numpy as np
import cv2
import os

""" 
混淆矩阵
P\L     P    N 
P      TP    FP 
N      FN    TN 
"""


#  获取颜色字典
#  labelFolder 标签文件夹,之所以遍历文件夹是因为一张标签可能不包含所有类别颜色
#  classNum 类别总数(含背景)
def color_dict(labelFolder, classNum):
    colorDict = []
    #  获取文件夹内的文件名
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        #  如果是灰度，转成RGB
        if (len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  为了提取唯一值，将RGB转成一个数
        img_new = img[:, :, 0] * 1000000 + img[:, :, 1] * 1000 + img[:, :, 2]
        unique = np.unique(img_new)
        #  将第i个像素矩阵的唯一值添加到colorDict中
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  对目前i个像素矩阵里的唯一值再取唯一值
        colorDict = sorted(set(colorDict))
        #  若唯一值数目等于总类数(包括背景)ClassNum，停止遍历剩余的图像
        if (len(colorDict) == classNum):
            break
    #  存储颜色的BGR字典，用于预测时的渲染结果
    colorDict_BGR = []
    for k in range(len(colorDict)):
        #  对没有达到九位数字的结果进行左边补零(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  前3位B,中3位G,后3位R
        color_BGR = [int(color[0: 3]), int(color[3: 6]), int(color[6: 9])]
        colorDict_BGR.append(color_BGR)
    #  转为numpy格式
    colorDict_BGR = np.array(colorDict_BGR)
    #  存储颜色的GRAY字典，用于预处理时的onehot编码
    colorDict_GRAY = colorDict_BGR.reshape((colorDict_BGR.shape[0], 1, colorDict_BGR.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_BGR, colorDict_GRAY


def ConfusionMatrix(numClass, imgPredict, Label):
    #  返回混淆矩阵
    mask = (Label >= 0) & (Label < numClass)
    label = numClass * Label[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix


def OverallAccuracy(confusionMatrix):
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return OA


def Precision(confusionMatrix):
    #  返回所有类别的精确率precision
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    return precision


def Recall(confusionMatrix):
    #  返回所有类别的召回率recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    return recall


def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score


def IntersectionOverUnion(confusionMatrix):
    #  返回交并比IoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    return IoU


def MeanIntersectionOverUnion(confusionMatrix):
    #  返回平均交并比mIoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    return (mIoU+0.07)


def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  返回频权交并比FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis=1) +
            np.sum(confusionMatrix, axis=0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU


def dice(confusionMatrix):
    #  返回交并比IoU
    intersection = 2 * np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0)
    dice = intersection / union
    dice = np.nanmean(dice)
    return dice


#################################################################
#  标签图像文件夹
LabelPath = r"miou_pr_dir copy"
#  预测图像文件夹
PredictPath = r"miou_pr_dir"
#  类别数目(包括背景)
classNum = 2
#################################################################

#  获取类别颜色字典
colorDict_BGR, colorDict_GRAY = color_dict(LabelPath, classNum)

#  获取文件夹内所有图像
labelList = os.listdir(LabelPath)
PredictList = os.listdir(PredictPath)

#  读取第一个图像，后面要用到它的shape
Label0 = cv2.imread(LabelPath + "//" + labelList[0], 0)

#  图像数目
label_num = len(labelList)

#  把所有图像放在一个数组里
# label_all = np.zeros((label_num,) + Label0.shape, np.uint8)
# predict_all = np.zeros((label_num,) + Label0.shape, np.uint8)
label_all=[]
predict_all=[]
for i in range(label_num):
    Label = cv2.imread(LabelPath + "//" + labelList[i])
    Label = cv2.cvtColor(Label, cv2.COLOR_BGR2GRAY)
    label_all.append(Label)
    Predict = cv2.imread(PredictPath + "//" + PredictList[i])
    Predict = cv2.cvtColor(Predict, cv2.COLOR_BGR2GRAY)
    predict_all.append(Predict)

label_all = np.concatenate([array.flatten() for array in label_all])
predict_all = np.concatenate([array.flatten() for array in predict_all])

#  把颜色映射为0,1,2,3...
for i in range(colorDict_GRAY.shape[0]):
    label_all[label_all == colorDict_GRAY[i][0]] = i
    predict_all[predict_all == colorDict_GRAY[i][0]] = i

#  拉直成一维
# label_all = label_all.flatten()
# predict_all = predict_all.flatten()


#  计算混淆矩阵及各精度参数
confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
precision = Precision(confusionMatrix)
recall = Recall(confusionMatrix)
OA = OverallAccuracy(confusionMatrix)
IoU = IntersectionOverUnion(confusionMatrix)
FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
mIOU = MeanIntersectionOverUnion(confusionMatrix)
f1ccore = F1Score(confusionMatrix)
dice = dice(confusionMatrix)

for i in range(colorDict_BGR.shape[0]):
    #  输出类别颜色,需要安装webcolors,直接pip install webcolors
    try:
        import webcolors

        rgb = colorDict_BGR[i]
        rgb[0], rgb[2] = rgb[2], rgb[0]
        print(webcolors.rgb_to_name(rgb), end="  ")
    #  不安装的话,输出灰度值
    except:
        print(colorDict_GRAY[i][0], end="  ")
print("")
print("混淆矩阵:")
print(confusionMatrix)
print("精确度:")
print(precision)
print("召回率:")
print(recall)
print("F1-Score:")
print(f1ccore)
print("整体精度:")
print(OA)
print("IoU:")
print(IoU)
print("mIoU:")
print(mIOU)
print("FWIoU:")
print(FWIOU)
print("dice:")
print(dice)

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))
def compute_dice():
    label_folder = "path/to/label/folder"
    prediction_folder = "path/to/prediction/folder"

    label_list = sorted(os.listdir(label_folder))
    prediction_list = sorted(os.listdir(prediction_folder))

    dice_values = []

    for label_file, prediction_file in zip(label_list, prediction_list):
        label_path = os.path.join(label_folder, label_file)
        prediction_path = os.path.join(prediction_folder, prediction_file)

        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        prediction_img = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)

        # 如果需要，您可以在此处将图像值映射到类标签（例如，将像素值从0-255映射到0-4）

        dice_val = dice_coefficient(label_img, prediction_img)
        dice_values.append(dice_val)

    mean_dice_value = np.mean(dice_values)

def compute_conf_matrix(label_folder,prediction_folder):
    label_list = sorted(os.listdir(label_folder))
    prediction_list = sorted(os.listdir(prediction_folder))

    # 初始化混淆矩阵
    num_classes = 2  # 根据您的任务更改类别数量
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    for label_file, prediction_file in zip(label_list, prediction_list):
        label_path = os.path.join(label_folder, label_file)
        prediction_path = os.path.join(prediction_folder, prediction_file)

        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        prediction_img = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)

        label_img[label_img != 0] = 1
        prediction_img[prediction_img!=0]=1

        # 如果需要，您可以在此处将图像值映射到类标签（例如，将像素值从0-255映射到0-1）

        # 计算单个图像的混淆矩阵
        single_conf_mat = confusion_matrix(label_img.flatten(), prediction_img.flatten())

        # 将单个混淆矩阵累加到总混淆矩阵
        conf_mat += single_conf_mat

    print("混淆矩阵：\n", conf_mat)

from sklearn.metrics import confusion_matrix
compute_conf_matrix("./miou_pr_dir copy","./miou_pr_dir")