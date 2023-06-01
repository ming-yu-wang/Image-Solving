import numpy as np
import cv2
import math
import os
"""# 1.固定阈值法二值化
#阈值难以确定 效果不佳
# 图片二值化
from PIL import Image
import matplotlib.pyplot as plt
img = Image.open('E://labelImgWork//VOC//Before//1.jpg')
# 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
Img = img.convert('L')
Img.save("E://labelImgWork//VOC//After//test1.jpg")
# 自定义灰度界限，大于这个值为黑色，小于这个值为白色
threshold = 200
table = []
for i in range(256):
    if i < threshold:
        table.append(0)
    else:
        table.append(1)
# 图片二值化
photo = Img.point(table, '1')
#photo.save("test2.jpg")"""

#2.OTSU大律法
def read_path(file_pathname):
    #遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        print(filename)
        image = cv2.imread(file_pathname+'/'+filename,0)
        rows, cols = image.shape[:2]
        gray_hist = np.zeros([256], np.uint64)
        # 首先获取图像的灰度图
        for i in range(rows):
            for j in range(cols):
                gray_hist[image[i][j]] += 1
        uniformGrayHist = gray_hist / float(rows * cols)
        # 计算零阶累积距和一阶累积距
        zeroCumuMomnet = np.zeros(256, np.float32)
        oneCumuMomnet = np.zeros(256, np.float32)
        for k in range(256):
            if k == 0:
                # k等于0时，作为初始状态要进行初始化赋值，要不会out of index
                zeroCumuMomnet[k] = uniformGrayHist[0]
                oneCumuMomnet[k] = (k) * uniformGrayHist[0]
            else:
                # 零阶和一阶累积距的计算都是一个累加的结果
                zeroCumuMomnet[k] = zeroCumuMomnet[k - 1] + uniformGrayHist[k]
                oneCumuMomnet[k] = oneCumuMomnet[k - 1] + k * uniformGrayHist[k]
        # 计算类间方差
        variance = np.zeros(256, np.float32)
        for k in range(255):
            if zeroCumuMomnet[k] == 0 or zeroCumuMomnet[k] == 1:
                variance[k] = 0
            else:
                # 将每一个灰度值作为阈值时，前景区域的平均灰度、背景区域的平均灰度与整幅图像的平均灰度的方差
                variance[k] = math.pow(oneCumuMomnet[255] * zeroCumuMomnet[k] - oneCumuMomnet[k], 2) / (
                        zeroCumuMomnet[k] * (1.0 - zeroCumuMomnet[k]))
        # 找到阈值
        threshLoc = np.where(variance[0:255] == np.max(variance[0:255]))
        thresh = threshLoc[0][0]
        # 阈值处理
        threshold = np.copy(image)
        threshold[threshold > thresh] = 255
        threshold[threshold <= thresh] = 0
        cv2.imwrite(r'E:\labelImgWork\VOC\img_promotion\edge_threshold'+"/"+filename,threshold)
read_path(r"E:\labelImgWork\VOC\img_promotion\edge")