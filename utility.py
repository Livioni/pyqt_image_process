from hashlib import new
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sci
from copy import deepcopy
import random

def __saturated(pix):
    if pix > 255:
        pix = 255
    elif pix < 0:
        pix = 0
    else:
        pix=pix
    return pix

def rgb2gray(img):
    #灰度化
    gray_img = np.zeros([img.shape[0],img.shape[1]],dtype=np.uint8)
    b,g,r = cv2.split(img)   #提取 BGR 颜色通道
    for row in range(img.shape[0]):
        for colmun in range(img.shape[1]):
            gray_img[row][colmun] = int(0.3*r[row][colmun]+0.59*g[row][colmun]+0.11*b[row][colmun])
    return gray_img

def gray_histogram(img):
    #灰度图直方图
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
    #直方图均衡化
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

def gradient_sharpening(img):
    #梯度锐化
    img = rgb2gray(img)
    new_img = np.zeros(img.shape,dtype=np.uint8)
    grad_img = np.zeros(img.shape,dtype=np.uint8)
    for row in range(img.shape[0]-1):
        for colmun in range(img.shape[1]-1):
            f1 = abs(int(img[row+1][colmun])-int(img[row][colmun]))
            f2 = abs(int(img[row][colmun+1])-int(img[row][colmun]))
            grad = __saturated(f1+f2)
            grad_img[row][colmun] = grad
            new_img[row][colmun] = __saturated(img[row][colmun] - grad)
    return new_img, grad_img

def laplace_sharpening(img):
    #Laplace 锐化
    img = rgb2gray(img)
    new_img = np.zeros(img.shape,dtype=np.uint8)
    grad_img = np.zeros(img.shape,dtype=np.uint8)
    for row in range(1,img.shape[0]-1):
        for colmun in range(1,img.shape[1]-1):
            grad = __saturated(int(img[row+1][colmun])+int(img[row-1][colmun])+int(img[row][colmun+1])+int(img[row][colmun-1])-4*int(img[row][colmun]))
            grad_img[row][colmun] = grad
            new_img[row][colmun] = __saturated(5*int(img[row][colmun])-int(img[row+1][colmun])-int(img[row-1][colmun])-int(img[row][colmun+1])-int(img[row][colmun-1]))
    return new_img, grad_img

def roberts(img):
    #Roberts 边缘检测
    img = rgb2gray(img)
    edge = np.zeros(img.shape,dtype=np.uint8)
    for row in range(img.shape[0]-1):
        for colmun in range(img.shape[1]-1):
            edge[row][colmun] = __saturated(abs(int(img[row][colmun])-int(img[row+1][colmun+1]))+abs(int(img[row+1][colmun])-int(img[row][colmun+1])))
    return edge

def sobel(img):
    #Sobel
    img = rgb2gray(img)
    edge = np.zeros(img.shape,dtype=np.uint8)
    for row in range(1,img.shape[0]-1):
        for colmun in range(1,img.shape[1]-1):
            fx = abs(-int(img[row-1][colmun-1])-2*int(img[row-1][colmun])-int(img[row-1][colmun+1])+int(img[row+1][colmun-1])+2*int(img[row+1][colmun])+int(img[row+1][colmun+1]))
            fy = abs(-int(img[row-1][colmun-1])-2*int(img[row][colmun-1])-int(img[row+1][colmun-1])+int(img[row-1][colmun+1])+2*int(img[row+1][colmun])+int(img[row+1][colmun+1]))
            edge[row][colmun] = __saturated(fx+fy)
    return edge 

def laplace(img):
    #laplace
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


def __SobelFilter(image):
    convolved = np.zeros(image.shape)
    G_x = np.zeros(image.shape)
    G_y = np.zeros(image.shape)
    size = image.shape
    kernel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    kernel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            G_x[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], kernel_x))
            G_y[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], kernel_y))
    
    convolved = np.sqrt(np.square(G_x) + np.square(G_y))
    convolved = np.multiply(convolved, 255.0 / convolved.max())

    angles = np.rad2deg(np.arctan2(G_y, G_x))
    angles[angles < 0] += 180
    convolved = convolved.astype('uint8')
    return convolved, angles

def canny(img):
    filtered_img = rgb2gray(img) 
    gradient, direction = __SobelFilter(filtered_img)

    edge_point = np.zeros(filtered_img.shape,dtype=np.uint8)
    edge_candiate = np.zeros(filtered_img.shape,dtype=np.uint8)
    gradient_copy = gradient

    for row in range(1,filtered_img.shape[0]-1):
        for colmun in range(1,filtered_img.shape[1]-1): 
            if (0 <= direction[row][colmun] < 22.5) or (157.5 <= direction[row][colmun] <= 180):
                if (gradient[row][colmun] > gradient[row][colmun-1]) and (gradient[row][colmun] > gradient[row][colmun+1]):
                    continue
                else:
                    gradient_copy[row][colmun] = 1
            elif (22.5 <= direction[row][colmun]) and (direction[row][colmun] < 67.5):
                if (gradient[row][colmun] > gradient[row+1][colmun+1]) and (gradient[row][colmun] > gradient[row-1][colmun-1]):
                    continue
                else:
                    gradient_copy[row][colmun] = 1

            elif (67.5 <= direction[row][colmun] < 112.5):
                if (gradient[row][colmun] > gradient[row-1][colmun]) and (gradient[row][colmun] > gradient[row+1][colmun]):
                    continue    
                else:
                    gradient_copy[row][colmun] = 1
            else:            
                if (gradient[row][colmun] > gradient[row+1][colmun-1]) and (gradient[row][colmun] > gradient[row-1][colmun+1]):
                    continue
                else:
                    gradient_copy[row][colmun] = 1

    for ele in zip(np.where(gradient_copy==1)[0],np.where(gradient_copy==1)[1]):
        row = ele[0]
        col = ele[1]
        gradient[row][col] = 0

    var = np.sqrt(np.var(gradient))
    mean = np.mean(gradient)                             
    high_threshold =  var + mean      
    low_threshold = 0.4 * high_threshold
    for row in range(filtered_img.shape[0]):
        for colmun in range(filtered_img.shape[1]):
            if (gradient[row][colmun] >= high_threshold):
                edge_point[row][colmun] = 255        #strong point
            elif (low_threshold < gradient[row][colmun]) and (gradient[row][colmun] <= high_threshold):
                edge_candiate[row][colmun] = 1       # weak point
            else:
                edge_point[row][colmun] = 0

    add_edge = np.zeros(filtered_img.shape,dtype=np.uint8)
    for ele in zip(np.where(edge_candiate==1)[0],np.where(edge_candiate==1)[1]):
        row = ele[0]
        col = ele[1]
        if ((edge_point[row-1][col-1]==255) or (edge_point[row-1][col]==255) or (edge_point[row-1][col+1]==255) or 
            (edge_point[row][col-1]==255) or (edge_point[row][col]==255) or (edge_point[row][col+1]==255) or 
            (edge_point[row+1][col-1]==255) or (edge_point[row+1][col]==255) or (edge_point[row+1][col+1]==255)):
            add_edge[row][col] = 255

    edge_point += add_edge       
    return edge_point

def mean_filter(img):
    filtered_img = deepcopy(img)
    for ch in range(filtered_img.shape[2]):
        for row in range(1,img.shape[0]-1):
            for colmun in range(1,img.shape[1]-1):
                filtered_img[row][colmun][ch] = np.mean([[img[row-1][colmun-1][ch],img[row-1][colmun][ch],img[row-1][colmun+1][ch]],\
                                                    [img[row][colmun-1][ch],img[row][colmun][ch],img[row][colmun+1][ch]],\
                                                    [img[row+1][colmun-1][ch],img[row+1][colmun][ch],img[row+1][colmun+1][ch]]])
    return filtered_img

def median_filter(img):
    filtered_img = deepcopy(img)
    for ch in range(filtered_img.shape[2]):
        for row in range(1,img.shape[0]-1):
            for colmun in range(1,img.shape[1]-1):
                field = np.array([[img[row-1][colmun-1][ch],img[row-1][colmun][ch],img[row-1][colmun+1][ch]],\
                                                    [img[row][colmun-1][ch],img[row][colmun][ch],img[row][colmun+1][ch]],\
                                                    [img[row+1][colmun-1][ch],img[row+1][colmun][ch],img[row+1][colmun+1][ch]]])
                filtered_img[row][colmun][ch] = np.sort(field.flatten())[4]
    return filtered_img    
 
def __ones_kernel(kernel,size=(1,1),loc=(0,0),value = 1):
    tmp = np.ones(size)
    kernel_tmp = deepcopy(kernel)
    kernel_tmp[loc[0]:(loc[0]+size[0]),loc[1]:(loc[1]+size[1])] = tmp
    return kernel_tmp

def s_meanfilter(img,radius,iteration = 1):
    r = radius
    zero_kernel = np.zeros([2*r+1,2*r+1])
    k_L = __ones_kernel(zero_kernel,size= (2*r+1,r+1),loc= (0,0))/((2*r+1)*(r+1))
    k_R = __ones_kernel(zero_kernel,size= (2*r+1,r+1),loc= (0,r))/((2*r+1)*(r+1))
    k_U = k_L.T
    k_D = k_U[::-1]
    k_NW = __ones_kernel(zero_kernel,size= (r+1,r+1),loc= (0,0))/((r+1)*(r+1))
    k_NE = __ones_kernel(zero_kernel,size= (r+1,r+1),loc= (0,r))/((r+1)*(r+1))
    k_SW = k_NW[::-1]
    k_SE = k_NE[::-1]
    kernels = [k_L,k_R,k_U,k_D,k_NW,k_NE,k_SW,k_SE]

    m = img.shape[0]+2*r
    n = img.shape[1]+2*r
    dis = np.zeros([8,m,n]);
    result = np.zeros_like(img)
    
    for ch in range(img.shape[2]):
        U = np.pad(img[:,:,ch],(r,r),'edge');
        for i in range(iteration):
            for id,kernel in enumerate(kernels):
                conv2 = sci.correlate2d(U,kernel,'same')
                dis[id] = conv2 - U
            norm = []
            for m in range(8):
                norm.append(np.linalg.norm(dis[m]))
            min_indx = np.argmin(np.min(norm))
            U = U + dis[min_indx]
        result[:,:,ch] = U[r:-r,r:-r]
    return result

def impluse_noise(image: np.ndarray, prob=0.01):
    new_img = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if random.random() < prob:
                new_img[i][j] = 0 if random.random() < 0.5 else 255
            else:
                new_img[i][j] = image[i][j]
    return new_img

def gaussian_noise(image, mean=0, sigma=0.1):
    image = np.asarray(image / 255, dtype=np.float32)  # 图片灰度标准化
    noise = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 产生高斯噪声
    output = image + noise  # 将噪声和图片叠加
    output = np.clip(output, 0, 1)
    output = np.uint8(output * 255)
    return output   

def __erode_demo(pic):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    dst = cv2.erode(pic, kernel)
    return dst

def __dilate_demo(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dst = cv2.dilate(binary, kernel)
    return dst

def morphological_filter(img):
    new_img1 = __erode_demo(img)
    new_img2 = __dilate_demo(new_img1)
    return new_img1,new_img2

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

def affine_transform(img):
    edge = img.shape
    # 原图像中的三组坐标 根据[x,y] 表示的，而并非数组的行列
    pts1 = np.float32([[0, 0] , [edge[1], 0], [0, edge[0]]])
    # 转换后的三组对应坐标
    pts2 = np.float32([[0,edge[0]*0.33], [edge[1]*0.85, edge[0]*0.25], [edge[1]*0.15, edge[0]*0.7]])
    # 计算仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    # 执行变换
    new_img = cv2.warpAffine(img, M ,edge[0:2])
    return new_img

def perspective_transform(img):
    edge = img.shape
    # 原图的四组顶点坐标
    pts3D1 = np.float32([[0, 0], [edge[1], 0], [0, edge[0]], [edge[0]-1, edge[1]-1]])
    # 转换后的四组坐标
    pts3D2 = np.float32([[edge[1]*0.05, edge[0]*0.33], [edge[1]*0.9, edge[0]*0.25], [edge[1]*0.2, edge[0]*0.7], [edge[0]*0.8, edge[1]*0.9]])
    # 计算透视放射矩阵
    M = cv2.getPerspectiveTransform(pts3D1, pts3D2)
    # 执行变换
    new_img = cv2.warpPerspective(img, M, edge[0:2])
    return new_img
