import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb2gray(img):
    new_figure = np.zeros([img.shape[0],img.shape[1]],dtype=np.uint8)
    b,g,r = cv2.split(img)   #提取 BGR 颜色通道
    for row in range(img.shape[0]):
        for colmun in range(img.shape[1]):
            new_figure[row][colmun] = int(0.3*r[row][colmun]+0.59*g[row][colmun]+0.11*b[row][colmun])
    return new_figure

def gray_histogram(img):
    plt.style.use('seaborn-darkgrid')
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