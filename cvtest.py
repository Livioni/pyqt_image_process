import cv2
import numpy as np
import random
from copy import deepcopy

# 读一个图片并进行显示(图片路径需自己指定)
logo=cv2.imread("figures/lena.jpg")
# logo=cv2.imread("figures/icon.jpeg")
# logo=cv2.imread("figures/logo_01.png")
# logo=cv2.imread("figures/source.jpg")
# cv2.imshow("image",logo)
# cv2.waitKey(0)

def __saturated(pix):
    if pix > 255:
        pix = 255
    elif pix < 0:
        pix = 0
    else:
        pix=pix
    return pix

def rgb2gray(img):
    new_figure = np.zeros([img.shape[0],img.shape[1]],dtype=np.uint8)
    b,g,r = cv2.split(img)   #提取 BGR 颜色通道
    for row in range(img.shape[0]):
        for colmun in range(img.shape[1]):
            new_figure[row][colmun] = int(0.3*r[row][colmun]+0.59*g[row][colmun]+0.11*b[row][colmun])
    return new_figure



import pysift

image = cv2.imread('figures/lena.jpg', 0)
keypoints, descriptors = pysift.computeKeypointsAndDescriptors(image)
cv2.imshow("figure1",keypoints)
cv2.imshow("figure2",descriptors)
# cv2.imshow("figure1",logo)
cv2.waitKey(0)

