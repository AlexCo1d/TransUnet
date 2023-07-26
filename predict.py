# -------------------------------------#
#       对单张图片进行预测
# -------------------------------------#

from PIL import Image
import os
import time
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch

from seg_metrics import seg_metrics, Get_ROC
from train_config import config
from utils.postprocess import postprocess

txt_path = "./VOCdevkit/VOC2007/ImageSets/Segmentation/valid"
image_path = './VOCdevkit/VOC2007/JPEGImages'
label_path = './VOCdevkit/VOC2007/SegmentationClass'

# class miou_Pspnet(psp):
#     def detect_image(self, image):
#         orininal_h = np.array(image).shape[0]
#         orininal_w = np.array(image).shape[1]
#
#         image, nw, nh = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
#         images = [np.array(image) / 255]
#         images = np.transpose(images, (0, 3, 1, 2))
#
#         with torch.no_grad():
#             images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
#             if self.cuda:
#                 images = images.cuda()
#             pr = self.net(images)[0]
#             pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
#
#         pr = pr[int((self.model_image_size[0] - nh) // 2):int((self.model_image_size[0] - nh) // 2 + nh),
#              int((self.model_image_size[1] - nw) // 2):int((self.model_image_size[1] - nw) // 2 + nw)]
#
#         image = Image.fromarray(np.uint8(pr)).resize((orininal_w, orininal_h), Image.NEAREST)
#
#         return image

type = config.image_type


def show_image():
    from unet import uNet
    uNet = uNet()
    uNet.net.eval()

    with open(txt_path, "r") as file:
        lines = file.readlines()

    for jpg in lines:
        jpg = jpg.replace('\n', '') + type
        img = Image.open(os.path.join(image_path, jpg)).convert('RGB')
        start_time = time.time()
        image, _ = uNet.detect_image(img, mix=config.output_type)
        image = postprocess(np.array(image))
        duration = time.time() - start_time
        print("预测时间", duration)
        image.save("./img_out/" + jpg)


def predict_and_eval():
    from unet import uNet
    label_all = []
    predict_all = []

    for fold in range(config.n_fold if config.n_fold>=1 else 1):
        if not os.path.exists(f'pr_dir/fold_{fold+1}'):
            os.mkdir(f'pr_dir/fold_{fold+1}')

        unet = uNet(fold=fold)
        unet.net.eval()
        with open(txt_path+f'_{fold+1}.txt', "r") as file:
            lines = file.readlines()
        for image_name in lines:
            # 测试集原标签
            image_name = image_name.replace('\n', '') + type
            label_name = image_name.replace(type, '.png')

            label_truth = Image.open(os.path.join(label_path, label_name))
            # label_truth.save(f"pr_dir copy/{label_name}")

            # 测试集生成标签
            image = Image.open(os.path.join(image_path, image_name))
            label, score = unet.detect_image(image, mix=0) # label is classification result, pr is prediction score
            label = postprocess(np.array(label))
            label.save(f"pr_dir/fold_{fold+1}/{label_name}")
            # print(label_name, " done!")

            # only for 1 label, this work should be further accomplished, ROC for multiple labels
            if config.NUM_CLASSES==2:
                label_all.append(np.array(label_truth))
                predict_all.append(np.array(score[..., 1]))

        # evaluate the prediction by using the metrics
        seg_metrics(fold=fold+1)

        # get ROC func
        Get_ROC(predict_all,label_all,config.NUM_CLASSES)

if __name__ == '__main__':
    # show_image()
    predict_and_eval()

