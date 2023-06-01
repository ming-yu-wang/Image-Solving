import numpy as np
import cv2
import os
def read_path_grey(file_pathname):
    # 遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        img = cv2.imread(file_pathname + '/' + filename)
        if len(img.shape) == 2:
            print(filename,'This image is grayscale')
        else:
            print(filename,'This image is color')
read_path_grey('E:/labelImgWork/VOC/img_promotion/gray')
