# -------------------------------------#
#       对单张图片进行预测
# -------------------------------------#

from PIL import Image
import os
import time
from unet import uNet as psp
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch


class miou_Pspnet(psp):
    def detect_image(self, image):
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image, nw, nh = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        images = [np.array(image) / 255]
        images = np.transpose(images, (0, 3, 1, 2))

        with torch.no_grad():
            images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)

        pr = pr[int((self.model_image_size[0] - nh) // 2):int((self.model_image_size[0] - nh) // 2 + nh),
             int((self.model_image_size[1] - nw) // 2):int((self.model_image_size[1] - nw) // 2 + nw)]

        image = Image.fromarray(np.uint8(pr)).resize((orininal_w, orininal_h), Image.NEAREST)

        return image


def show_image():
    from unet import uNet
    uNet = uNet()
    fpath="./for_test/image"
    imgs = os.listdir(fpath)

    for jpg in imgs:
        img = Image.open(os.path.join(fpath,jpg)).convert('RGB')
        start_time = time.time()
        image = uNet.detect_image(img)
        duration = time.time() - start_time
        print("预测时间", duration)
        image.save("./img_out/" + jpg)


def transfer_image():
    psp = miou_Pspnet()
    test_path = './for_test'
    label_path = os.path.join(test_path, 'label')
    image_path = os.path.join(test_path, 'image')
    if not os.path.exists("./miou_pr_dir"):
        os.makedirs("./miou_pr_dir")

    for image_name in os.listdir(label_path):
        # 测试集原标签
        image = Image.open(os.path.join(label_path, image_name))
        image=np.array(image)
        #
        image[image!=0]=set_label
        image=Image.fromarray(image)
        # image = image.resize((512, 512))
        image.save(f"miou_pr_dir copy/{image_name}")
        image.save(f'{label_path}/{image.name}')

        # 测试集生成标签
        image = Image.open(os.path.join(image_path, image_name.replace('.png', '.jpg')))
        image=np.array(image)
        #
        image[image!=0]=set_label
        image=Image.fromarray(image)
        image = psp.detect_image(image)
        # image = image.resize((512, 512))
        image.save(f"miou_pr_dir/{image_name}")
        print(image_name, " done!")


if __name__ == '__main__':
    set_label = 1

    show_image()
    transfer_image()
