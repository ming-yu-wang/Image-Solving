import numpy as np
import cv2
import os
# def padding_0(img):
#     #均值滤波用得到
#     a = np.insert(img, 0, 0, 0)
#     b = np.insert(a, 0, 0, 0)
#     c = np.insert(b, 0, 0, 1)
#     d = np.insert(c, 0, 0, 1)
#     h_zeros = np.zeros((d.shape[0], 1))
#     e = np.hstack((d, h_zeros))
#     f = np.hstack((e, h_zeros))
#     w_zeros = np.zeros((1, f.shape[1]))
#     g = np.vstack((f, w_zeros))
#     img = np.vstack((g, w_zeros))
#     return img
import pywt


def medianBlur(image, ksize=2, ):
    '''
    中值滤波，去除椒盐噪声
    args:
        image：输入图片数据,要求为灰度图片
        ksize：滤波窗口大小
    return：
        中值滤波之后的图片
    '''
    rows, cols = image.shape[:2]
    # 输入校验
    half = ksize // 2
    startSearchRow = half
    endSearchRow = rows - half - 1
    startSearchCol = half
    endSearchCol = cols - half - 1
    dst = np.zeros((rows, cols), dtype=np.uint8)
    # 中值滤波
    for y in range(startSearchRow, endSearchRow):
        for x in range(startSearchCol, endSearchCol):
            window = []
            for i in range(y - half, y + half + 1):
                for j in range(x - half, x + half + 1):
                    window.append(image[i][j])
            # 取中间值
            window = np.sort(window, axis=None)
            if len(window) % 2 == 1:
                medianValue = window[len(window) // 2]
            else:
                medianValue = int((window[len(window) // 2] + window[len(window) // 2 + 1]) / 2)
            dst[y][x] = medianValue
    return dst

# def mean_fliter(img):
#     """均值滤波"""
#     img = padding_0(img)
#     for i in range(2, img.shape[0] - 2):
#         for j in range(2, img.shape[1] - 2):
#             mat = np.array(
#                 [[img[i - 2, j - 2], img[i - 2, j - 1], img[i - 2, j], img[i - 2, j + 1], img[i - 2, j + 1]],
#                  [img[i - 1, j - 2], img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1], img[i - 1, j + 1]],
#                  [img[i, j - 2], img[i, j - 1], img[i, j], img[i, j + 1], img[i, j + 1]],
#                  [img[i + 1, j - 2], img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1], img[i + 1, j + 1]],
#                  [img[i + 2, j - 2], img[i + 2, j - 1], img[i + 2, j], img[i + 2, j + 1], img[i + 2, j + 1]]])
#             mean = np.sum(mat) / 25
#             img[i, j] = mean
#     return img

def read_path_bymean(file_pathname):
    #遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        print(filename)
        ksize=5
        image = cv2.imread(file_pathname+'/'+filename,0)
        #med = mean_fliter(image/255)
        med=cv2.blur(image, (ksize, ksize))
        # 写入的路径
        cv2.imwrite(r'E:\labelImgWork\VOC\img_promotion\mean_wave' + "/" + filename, med)
def read_path_bymiddle(file_pathname):
    # 遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        print(filename)
        image = cv2.imread(file_pathname + '/' + filename)
        med = medianBlur(image)
        cv2.imwrite(r'E:\labelImgWork\VOC\img_promotion\threshold_middlewave' + "/" + filename, med)

def read_path_bygauss(file_pathname):
    # 遍历该目录下的所有图片文件
    # 高斯模糊
    for filename in os.listdir(file_pathname):
        print(filename)
        Gn = cv2.imread(file_pathname+'/'+filename)
        Gf = cv2.GaussianBlur(Gn, (3, 3), 0, 0)
        #写入的路径
        cv2.imwrite(r'E:\labelImgWork\VOC\img_promotion\threshold_middlewave_gausswave' + "/" + filename, Gf)

def read_path_bybilateral(file_pathname):
    # 遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        print(filename)
        image = cv2.imread(file_pathname + '/' + filename,0)
        imgBiFilter = cv2.bilateralFilter(image, d=5, sigmaColor=100, sigmaSpace=2)
        #写入的路径
        cv2.imwrite(r'E:\labelImgWork\VOC\img_promotion\threshold_middlewave_gausswave_bilateralwave' + "/" + filename, imgBiFilter)

#read_path_bymiddle(r"E:\labelImgWork\VOC\img_promotion\threshold_test")

#read_path_bymean(r"E:\labelImgWork\VOC\img_promotion\gray")

#read_path_bygauss(r"E:\labelImgWork\VOC\img_promotion\threshold_middlewave")

read_path_bybilateral(r"E:\labelImgWork\VOC\img_promotion\threshold_middlewave_gausswave")
