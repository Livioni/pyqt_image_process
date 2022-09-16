import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    img = rgb2gray(img)
    edge = np.zeros(img.shape,dtype=np.uint8)
    for row in range(1,img.shape[0]-1):
        for colmun in range(1,img.shape[1]-1):
            fx = abs(-int(img[row-1][colmun-1])-2*int(img[row-1][colmun])-int(img[row-1][colmun+1])+int(img[row+1][colmun-1])+2*int(img[row+1][colmun])+int(img[row+1][colmun+1]))
            fy = abs(-int(img[row-1][colmun-1])-2*int(img[row][colmun-1])-int(img[row+1][colmun-1])+int(img[row-1][colmun+1])+2*int(img[row+1][colmun])+int(img[row+1][colmun+1]))
            edge[row][colmun] = __saturated(fx+fy)
    return edge 

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
            if (0 <= direction[row, colmun] < 22.5) or (157.5 <= direction[row, colmun] <= 180):
                if (gradient[row][colmun] > gradient[row][colmun-1]) and (gradient[row][colmun] > gradient[row][colmun+1]):
                    continue
                else:
                    gradient_copy[row][colmun] = 1
            elif (22.5 <= direction[row][colmun]) and (direction[row][colmun] < 67.5):
                if (gradient[row][colmun] > gradient[row+1][colmun+1]) and (gradient[row][colmun] > gradient[row-1][colmun-1]):
                    continue
                else:
                    gradient_copy[row][colmun] = 1

            elif (67.5 <= direction[row, colmun] < 112.5):
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