import numpy as np
from PIL import Image
import cv2
import os
# 1,反色变换
# 假设原始图像的灰度范围是[0,L],L表示该图像最大的灰度值
# 则反色变换为output = L - input
def image_inverse(input):
    value_max = np.max(input)
    output = value_max - input
    return output
def read_path(file_pathname):
    #遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        gray_img = np.asarray(Image.open(file_pathname+'/'+filename).convert('L'))
        inv_img = image_inverse(gray_img)
        cv2.imwrite(r'E:\labelImgWork\VOC\img_promotion\edge_threshold_reverse'+"/"+filename,inv_img)
read_path(r"E:\labelImgWork\VOC\img_promotion\edge_threshold")
