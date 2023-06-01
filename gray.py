import cv2
import os
def read_path_grey(file_pathname):
    # 遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        img = cv2.imread(file_pathname + '/' + filename)
        # gray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         gray[i, j] = 0.299 * img[i, j, 2] + 0.587 * img[i, j, 1] + 0.114 * img[i, j, 0]
        #         #gray[i, j] = np.mean(img[i, j]) 均值法
        #         #gray[i, j] = np.max(img[i, j]) 最大值法
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if len(gray.shape) == 2:
            print(filename,'This image is grayscale')
        else:
            print(filename,'This image is color')
        cv2.imwrite(r'E:\labelImgWork\VOC\img_promotion\gray' + "/" + filename, gray)
read_path_grey('E:/labelImgWork/VOC/initial_operation/partone_after_manage')