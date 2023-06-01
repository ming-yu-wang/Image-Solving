# 实验时间 2023/4/20 11:10
import cv2
import numpy as np
import os
def read_path_edge(file_pathname):
    # 遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        print(filename)
        img = cv2.imread(file_pathname + '/' + filename,0)
        # 构造水平和垂直Sobel算子
        sobel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_vertical = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        # 使用filter2D函数分别应用水平和垂直Sobel算子进行卷积操作
        horizontal_edges = cv2.filter2D(img, -1, sobel_horizontal)
        vertical_edges = cv2.filter2D(img, -1, sobel_vertical)
        # 将两个方向的边缘检测结果取绝对值后合并
        edges = cv2.addWeighted(np.absolute(horizontal_edges),
                                0.5, np.absolute(vertical_edges), 0.5, 0)
        #写入的路径
        cv2.imwrite(r'E:\labelImgWork\VOC\img_promotion\edge' + "/" + filename, edges)
read_path_edge(r'E:\labelImgWork\VOC\img_promotion\threshold_middlewave_gausswave')