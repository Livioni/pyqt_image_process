import imp
import cv2
import numpy as np
import random
from copy import deepcopy

# 读一个图片并进行显示(图片路径需自己指定)
# logo=cv2.imread("figures/lena.jpg")
# logo=cv2.imread("figures/icon.jpeg")
logo=cv2.imread("figures/logo_01.png")
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

def laplace(img):
    img = rgb2gray(img)
    edge = np.zeros(img.shape,dtype=np.uint8)
    for row in range(1,img.shape[0]-1):
        for colmun in range(1,img.shape[1]-1):
            edge[row][colmun] = __saturated(int(img[row-1][colmun])+int(img[row][colmun-1])+int(img[row][colmun+1])+img[row+1][colmun]-4*int(img[row][colmun]))
    return edge

def krisch(img):
    img = rgb2gray(img)
    edge = np.zeros(img.shape,dtype=np.uint8)
    f1_kernal = np.array([[5,5,5], [-3,0,-3], [-3,-3,-3]],dtype=np.int8)
    f2_kernal = np.array([[-3, 5,5], [-3,0,5], [-3,-3,-3]],dtype=np.int8)
    f3_kernal = np.array([[-3,-3,5], [-3,0,5], [-3,-3,5]],dtype=np.int8)
    f4_kernal = np.array([[-3,-3,-3], [-3,0,5], [-3,5,5]],dtype=np.int8)
    f5_kernal = np.array([[-3,-3,-3], [-3,0,-3], [5,5,5]],dtype=np.int8)
    f6_kernal = np.array([[-3, -3, -3], [5,0,-3], [5,5,-3]],dtype=np.int8)
    f7_kernal = np.array([[5, -3, -3], [5,0,-3], [5,-3,-3]],dtype=np.int8)
    f8_kernal = np.array([[5, 5, -3], [5,0,-3], [-3,-3,-3]],dtype=np.int8)
    for row in range(1,img.shape[0]-1):
        for colmun in range(1,img.shape[1]-1):
            feild = np.array([[int(img[row-1][colmun-1]),int(img[row-1][colmun]),int(img[row-1][colmun+1])],\
                              [int(img[row][colmun-1]),int(img[row][colmun]),int(img[row][colmun+1])],\
                              [int(img[row+1][colmun-1]),int(img[row+1][colmun]),int(img[row+1][colmun+1])]],dtype=np.uint8)
            edge[row][colmun] = max(__saturated(np.sum(f1_kernal*feild)),__saturated(np.sum(f2_kernal*feild)),\
                                       __saturated(np.sum(f3_kernal*feild)),__saturated(np.sum(f4_kernal*feild)),\
                                       __saturated(np.sum(f5_kernal*feild)),__saturated(np.sum(f6_kernal*feild)),\
                                       __saturated(np.sum(f7_kernal*feild)),__saturated(np.sum(f8_kernal*feild)))
    return edge


def erode_demo(pic):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    dst = cv2.erode(pic, kernel)
    return dst

def dilate_demo(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dst = cv2.dilate(binary, kernel)
    return dst


def diy_gaussian_filter(img,template = '[[1,2,1],[2,8,2],[1,2,1]]'):
    filtered_img = deepcopy(img)
    template = eval(template)
    kernal = np.array(template)/np.sum(template)
    radius = int(np.floor(kernal.shape[0]/2))
    for ch in range(filtered_img.shape[2]):
        for row in range(radius,img.shape[0]-radius):
            for colmun in range(radius,img.shape[1]-radius):
                filtered_img[row][colmun][ch] = np.sum(np.multiply(img[row-radius:row+radius+1,colmun-radius:colmun+radius+1:,ch],kernal))
    return filtered_img


new_img = diy_gaussian_filter(logo,'[[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]')
cv2.imshow("figure",new_img)
cv2.imshow("figure1",logo)
cv2.waitKey(0)

