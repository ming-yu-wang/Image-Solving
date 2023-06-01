# 实验时间 2023/4/9 11:02
import cv2
import numpy as np

# 读取二值化后的图像
img = cv2.imread('E:/labelImgWork/VOC/img_promotion/Before/single.jpg', cv2.IMREAD_GRAYSCALE)

# 定义k近邻算法的参数
k = 1
kernel = np.ones((2,2), np.uint8)

# 对图像进行膨胀操作，以便去除噪声
dilation = cv2.dilate(img, kernel, iterations=1)

# 初始化k近邻算法模型
model = cv2.ml.KNearest_create()

# 将二维坐标（x,y）作为特征，将像素值作为标签训练模型
X_train = np.argwhere(dilation == 255)
y_train = img[dilation == 255].ravel()
model.train(X_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32))

# 对图像上所有黑色像素点（即噪声点）进行预测，并将预测结果赋值给对应像素点
X_test = np.argwhere(img == 0)
_, y_pred, _, _ = model.findNearest(X_test.astype(np.float32), k)
img[X_test[:, 0], X_test[:, 1]] = y_pred.ravel().astype(np.uint8)

# 显示去噪后的图像
cv2.imshow('Denoised Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
