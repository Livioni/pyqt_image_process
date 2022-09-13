import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读一个图片并进行显示(图片路径需自己指定)
logo=cv2.imread("logo_01.png")
# cv2.imshow("image",logo)
# cv2.waitKey(0)

def rgb2gray(img):
    new_figure = np.zeros([img.shape[0],img.shape[1]],dtype=np.uint8)
    b,g,r = cv2.split(img)   #提取 BGR 颜色通道
    for row in range(img.shape[0]):
        for colmun in range(img.shape[1]):
            new_figure[row][colmun] = int(0.3*r[row][colmun]+0.59*g[row][colmun]+0.11*b[row][colmun])
    return new_figure

def gray_histogram(img):
    img = rgb2gray(img)
    x = np.linspace(0,255,256)
    gh = np.zeros(256,dtype=np.float64)
    pixsum = 0
    for row in range(img.shape[0]):
        for colmun in range(img.shape[1]):
            gh[img[row][colmun]] += 1
            pixsum += 1
    gh /= pixsum
    fig, ax = plt.subplots()
    ax.plot(x, gh, linewidth=2.0)
    plt.show()  
    return

def histogram_equalize(img):
    img = rgb2gray(img)
    x = np.linspace(0,255,256)
    gh = np.zeros(256,dtype=np.float64)
    pixsum = 0
    for row in range(img.shape[0]):
        for colmun in range(img.shape[1]):
            gh[img[row][colmun]] += 1
            pixsum += 1


    gh_acc = np.zeros(256,dtype=np.float64)
    for i in range(256):
        gh_acc[i] = 255*np.sum(gh[:i+1])/pixsum

    for row in range(img.shape[0]):
        for colmun in range(img.shape[1]):
            img[row][colmun] = gh_acc[img[row][colmun]]

    return img
 


new_img = histogram_equalize(logo)
cv2.imshow("image",new_img)
cv2.waitKey(0)

# image = cv2.imread("logo_01.png")
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
# hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
# # 直方图均衡化
# gray_image_eq = cv2.equalizeHist(gray_image)
# cv2.imshow("image",gray_image_eq)
# cv2.waitKey(0)

# new_fi = rgb2gray(image)
# cv2.imshow("image",new_fi)
# cv2.waitKey(0)

