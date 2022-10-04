import cv2,os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sci
from copy import deepcopy
import random,pysift

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
    new_img = cv2.warpAffine(img, M ,(edge[1],edge[0]))
    return new_img

def perspective_transform(img):
    edge = img.shape
    # 原图的四组顶点坐标
    pts3D1 = np.float32([[214,73],[356,73] , [83,159], [491,159 ]])
    # 转换后的四组坐标
    pts3D2 = np.float32([[83,17], [491,17], [83,159], [491,159]])
    # 计算透视放射矩阵
    M = cv2.getPerspectiveTransform(pts3D1, pts3D2)
    # 执行变换
    new_img = cv2.warpPerspective(img, M, (edge[1],edge[0]))
    return new_img

def calib_camera(calib_dir, pattern_size=(9, 6), draw_points=False):
    """
    calibrate camera
    :param calib_dir: str
    :param pattern_size: (x, y), the number of points in x, y axes in the chessboard
    :param draw_points: bool, whether to draw the chessboard points
    """
    object_points = []
    image_points = []

    xl = np.linspace(0, pattern_size[0], pattern_size[0], endpoint=False)
    yl = np.linspace(0, pattern_size[1], pattern_size[1], endpoint=False)
    xv, yv = np.meshgrid(xl, yl)
    object_point = np.insert(np.stack([xv, yv], axis=-1), 2, 0, axis=-1).astype(np.float32).reshape([-1, 3])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img_dir = calib_dir
    assert os.path.isdir(img_dir), 'Path {} is not a dir'.format(img_dir)
    imagenames = os.listdir(img_dir)
    for imagename in imagenames:
        if not os.path.splitext(imagename)[-1] in ['.jpg', '.png', '.bmp', '.tiff', '.jpeg']:
            continue
        img_path = os.path.join(img_dir, imagename)
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(img_gray, patternSize=pattern_size)
        if ret:

            object_points.append(object_point)
            corners_refined = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners.reshape([-1, 2]))
            if draw_points:
                cv2.drawChessboardCorners(img, pattern_size, corners_refined, ret)
                if img.shape[0] * img.shape[1] > 1e6:
                    scale = round((1. / (img.shape[0] * img.shape[1] // 1e6)) ** 0.5, 3)
                    img_draw = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                else:
                    img_draw = img

                cv2.imshow('img', img_draw)
                cv2.waitKey(0)

    assert len(image_points) > 0, 'Cannot find any chessboard points, maybe incorrect pattern_size has been set'
    reproj_err, k_cam, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points,
                                                                       image_points,
                                                                       img_gray.shape[::-1],
                                                                       None,
                                                                       None,
                                                                       criteria=criteria)
    print("相机内参矩阵：",k_cam)
    print("畸变系数：",dist_coeffs)
    return 

def __getImageList(img_dir):
    imgPath = []
    if os.path.exists(img_dir) is False:
        print('error dir')
    else:
        for parent, dirNames, fileNames in os.walk(img_dir):
            for fileName in fileNames:
                imgPath.append(os.path.join(parent, fileName))
    return imgPath

def __getObjectPoints(m, n, k):
    objP = np.zeros(shape=(m * n, 3), dtype=np.float32)
    for i in range(m * n):
        objP[i][0] = i % m
        objP[i][1] = int(i / m)
    return objP * k

def bicalib_camera():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objPoint = __getObjectPoints(9, 6, 10)
    
    objPoints = []
    imgPointsL = []
    imgPointsR = []
    imgPathL = 'chesspad/chesspadL'
    imgPathR = 'chesspad/chesspadR'
    filePathL = __getImageList(imgPathL)
    filePathR = __getImageList(imgPathR)
    
    for i in range(len(filePathL)):
        imgL = cv2.imread(filePathL[i])
        imgR = cv2.imread(filePathR[i])
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        retL, cornersL = cv2.findChessboardCorners(grayL, (9, 6), None)
        retR, cornersR = cv2.findChessboardCorners(grayR, (9, 6), None)
        if (retL & retR) is True:
            objPoints.append(objPoint)
            cornersL2 = cv2.cornerSubPix(grayL, cornersL, (5, 5), (-1, -1), criteria)
            cornersR2 = cv2.cornerSubPix(grayR, cornersR, (5, 5), (-1, -1), criteria)
            imgPointsL.append(cornersL2)
            imgPointsR.append(cornersR2)
    retL, cameraMatrixL, distMatrixL, RL, TL = cv2.calibrateCamera(objPoints, imgPointsL, (640, 480), None, None)
    retR, cameraMatrixR, distMatrixR, RR, TR = cv2.calibrateCamera(objPoints, imgPointsR, (640, 480), None, None)

    retS, mLS, dLS, mRS, dRS, R, T, E, F = cv2.stereoCalibrate(objPoints, imgPointsL, 
    imgPointsR, cameraMatrixL,
                                                            distMatrixL, cameraMatrixR, 
                                                            distMatrixR, (640, 480),
                                                            criteria_stereo, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    # R， T为相机2与相机1旋转平移矩阵
    print("左相机内参数矩阵:\n",cameraMatrixL)
    print('*' * 20)
    print("右相机内参数矩阵:\n",cameraMatrixR)
    print('*' * 20)
    print("第一和第二相机坐标系之间的旋转矩阵:\n",R)
    print('*' * 20)
    print("第一和第二相机坐标系之间的平移向量:\n",T)
    print('*' * 20)
    print("本征矩阵:\n",E)
    print('*' * 20)
    print("基础矩阵:\n",F)
    print('*' * 20)
    return

def threshold(img,thres):
    if thres != '':
        thres = eval(thres)
        gray_img = rgb2gray(img)

        gray_img[gray_img < thres] = 0
        gray_img[gray_img >= thres] = 255 
    else:
        pass
    return gray_img

def otsu(img):
    gray_img = rgb2gray(img)
    h = gray_img.shape[0]
    w = gray_img.shape[1]
    N = h * w
    threshold_t = 0
    max_g = 0

    # 遍历每一个灰度级
    for t in range(256):
        # 使用numpy直接对数组进行运算
        n0 = gray_img[np.where(gray_img < t)]
        n1 = gray_img[np.where(gray_img >= t)]
        w0 = len(n0) / N
        w1 = len(n1) / N
        u0 = np.mean(n0) if len(n0) > 0 else 0.
        u1 = np.mean(n1) if len(n0) > 0 else 0.

        g = w0 * w1 * (u0 - u1) ** 2
        if g > max_g:
            max_g = g
            threshold_t = t
    # print('类间方差最大阈值：', threshold_t)
    gray_img[gray_img < threshold_t] = 0
    gray_img[gray_img >= threshold_t] = 255
    return gray_img

def kittle(img):
    gray_img = rgb2gray(img)
    edge = np.zeros(gray_img.shape,dtype=np.uint16)
    uend = np.zeros(gray_img.shape,dtype=np.uint16)
    for r in range(1,edge.shape[0]-1):
        for c in range(1,edge.shape[1]-1):
            fi = gray_img[r+1][c]-gray_img[r-1][c]
            fj = int(gray_img[r][c+1]-gray_img[r][c-1])
            edge[r][c] = max(abs(fi),abs(fj))
            uend[r][c] = edge[r][c] * gray_img[r][c]

    thres = np.sum(uend)/np.sum(edge)
    gray_img[gray_img < thres] = 0
    gray_img[gray_img >= thres] = 255
    return gray_img

def bulidbg(path):
    cap = cv2.VideoCapture(path)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # 创建混合高斯模型
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while(True):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        # print(frame)
        #形态学开运算去噪点（先腐蚀，再膨胀）
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('fgmask', fgmask)
        #寻找视频中的轮廓
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            perimeter = cv2.arcLength(c,True)
            if perimeter > 188:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow('frame',frame)

        k = cv2.waitKey(150) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return

def single_gauss(path):
    T = 2 # 前后景区分常数
    lr=0.95 # 学习率
    cap=cv2.VideoCapture(path)
    isFirst=True

    while cap.isOpened():
        ret,frame=cap.read()
        if frame is not None:
            gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            # 使用第一帧来初始化参数
            if isFirst:
                mean = gray
                std = 20 * np.ones_like(gray)
                var = std * std
                isFirst = False
            else: 
                # 标识目标
                cv2.imshow('orginal',frame)
                frontground = np.zeros_like(gray)
                for i in range(gray.shape[0]):
                    for j in range(gray.shape[1]):
                        if abs(int(gray[i][j]) - int(mean[i][j])) < T * std[i][j]:
                            mean[i][j] = __saturated(lr * int(mean[i][j]) + (1-lr) * int(gray[i][j]))
                            var[i][j] = __saturated(lr * var[i][j]+(1-lr) * (int(gray[i][j])-int(mean[i][j]))**2)
                            std[i][j] = __saturated(np.sqrt(var[i][j]))                
                        else:
                            frontground[i][j] = gray[i][j]
                kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                dst = cv2.erode(frontground, kernel1)
                kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                dst = cv2.dilate(dst, kernel2)
                cv2.imshow('tracking',dst)
                if cv2.waitKey(50) and 0xFF ==ord('q'):
                    break
        else:
            break

def histmatch():
    img1 = cv2.imread('figures/lena.jpg')
    img2 = cv2.imread('figures/lena-1.jpg')
    hsv_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    max_ = 0
    temp = np.zeros_like(hsv_img2)
    for i in range(0,hsv_img1.shape[0]-hsv_img2.shape[0]):
        for j in range(0,hsv_img1.shape[1]-hsv_img2.shape[1]):
            for x in range(i,hsv_img2.shape[0]+i):
                for y in range(j,hsv_img2.shape[1]+j):
                    temp[x-i][y-j] = hsv_img1[x][y]
            mat1 = cv2.calcHist([temp], [0], None, [256], [0, 256])
            mat1 = cv2.normalize(mat1,None,0,255,cv2.NORM_MINMAX)
            mat2 = cv2.calcHist([hsv_img2], [0], None, [256], [0, 256])
            mat2 = cv2.normalize(mat2,None,0,255,cv2.NORM_MINMAX)
            match = cv2.compareHist(mat1, mat2,cv2.HISTCMP_CORREL)
            print("src compare with src correlation value :",match)
            if match >= max_:
                max_ = match
                x_ray = i
                y_ray = j
    rect = np.array([x_ray,y_ray,img2.shape[1],img2.shape[0]])
    final_img = cv2.rectangle(hsv_img1,rect,(255,0,0),1,cv2.LINE_8,0)
    return final_img,hsv_img2

def tempmatch():
    img1 = cv2.imread('figures/lena.jpg')
    img2 = cv2.imread('figures/lena-1.jpg')
    result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED, mask=None)
    min_v, max_v, min_pt, max_pt = cv2.minMaxLoc(result)
    final_img = cv2.rectangle(img1,[max_pt[0],max_pt[1],img2.shape[1],img2.shape[0]],(255,0,0),1,cv2.LINE_8,0)
    
    return final_img,img2

def __rotation_invariant_LBP(img, radius=3, neighbors=8):
    h,w=img.shape
    dst = np.zeros((h-2*radius, w-2*radius),dtype=img.dtype)
    for i in range(radius,h-radius):
        for j in range(radius,w-radius):
            # 获得中心像素点的灰度值
            center = img[i,j]
            for k in range(neighbors):
                # 计算采样点对于中心点坐标的偏移量rx，ry
                rx = radius * np.cos(2.0 * np.pi * k / neighbors)
                ry = -(radius * np.sin(2.0 * np.pi * k / neighbors))
                # 为双线性插值做准备
                # 对采样点偏移量分别进行上下取整
                x1 = int(np.floor(rx))
                x2 = int(np.ceil(rx))
                y1 = int(np.floor(ry))
                y2 = int(np.ceil(ry))
                # 将坐标偏移量映射到0-1之间
                tx = rx - x1
                ty = ry - y1
                # 根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
                w1 = (1-tx) * (1-ty)
                w2 =    tx  * (1-ty)
                w3 = (1-tx) *    ty
                w4 =    tx  *    ty
                # 根据双线性插值公式计算第k个采样点的灰度值
                neighbor = img[i+y1,j+x1] * w1 + img[i+y2,j+x1] *w2 + img[i+y1,j+x2] *  w3 +img[i+y2,j+x2] *w4
                # LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
                dst[i-radius,j-radius] |= (neighbor>center)  <<  (np.uint8)(neighbors-k-1)
    # 进行旋转不变处理
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            currentValue = dst[i,j]
            minValue = currentValue
            for k in range(1, neighbors):
                # 对二进制编码进行循环左移，意思即选取移动过程中二进制码最小的那个作为最终值
                temp = (np.uint8)(currentValue>>(neighbors-k)) |  (np.uint8)(currentValue<<k)
                if temp < minValue:
                    minValue = temp
            dst[i,j] = minValue
    return dst

def lbp():
    img1 = cv2.imread('figures/redball.jpg')
    img2 = cv2.imread('figures/redball-1.jpg')
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    lbp_data1 = __rotation_invariant_LBP(gray_img1)
    lbp_data2 = __rotation_invariant_LBP(gray_img2)

    result = cv2.matchTemplate(lbp_data1, lbp_data2, cv2.TM_CCORR, mask=None)
    min_v, max_v, min_pt, max_pt = cv2.minMaxLoc(result)
    final_img = cv2.rectangle(img1,[max_pt[0],max_pt[1],img2.shape[1],img2.shape[0]],(255,0,0),1,cv2.LINE_8,0)

    return lbp_data1,final_img,img2

def sift():
    MIN_MATCH_COUNT = 10

    img1 = cv2.imread('figures/1.1.jpg', 0)  # queryImage
    img2 = cv2.imread('figures/1.2.jpg', 0)  # trainImage

    # Compute SIFT keypoints and descriptors
    kp1, des1 = pysift.computeKeypointsAndDescriptors(img1)
    kp2, des2 = pysift.computeKeypointsAndDescriptors(img2)

    # Initialize and use FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        # Estimate homography between template and scene
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

        # Draw detected template in scene image
        h, w = img1.shape
        pts = np.float32([[0, 0],
                        [0, h - 1],
                        [w - 1, h - 1],
                        [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        h1, w1 = img1.shape
        h2, w2 = img2.shape
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

        for i in range(3):
            newimg[hdif:hdif + h1, :w1, i] = img1
            newimg[:h2, w1:w1 + w2, i] = img2

        # Draw SIFT keypoint matches
        for m in good:
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))

        plt.imshow(newimg)    
        plt.savefig('figures/result.jpg', dpi=100) 
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

    return img1,img2

def svm_predict():
    rand1 = np.array([[501,300],[10,501]])
    rand2 = np.array([[255,10],[501,255],[10,501]])
    label = np.array([[0],[0],[1],[1],[1]])

    data = np.vstack((rand1,rand2))
    data = np.array(data,dtype='float32')
    svm = cv2.ml.SVM_create() 

    svm.setType(cv2.ml.SVM_C_SVC) # svm type
    svm.setKernel(cv2.ml.SVM_LINEAR) # line
    svm.setC(0.01)
    # 训练
    result = svm.train(data,cv2.ml.ROW_SAMPLE,label)
    # 预测
    predicted_labels = np.zeros([501,501],dtype=np.int0)
    for i in range(predicted_labels.shape[0]):
        for j in range(predicted_labels.shape[1]):
            predicted_labels[i][j] = svm.predict(np.array([[i,j]],dtype=np.float32))[1].item()
    predicted_labels[np.where(predicted_labels==1)] = 255
    predicted_labels = np.array(predicted_labels,dtype=np.uint8)

    cv2.imshow("SVM",predicted_labels)
    cv2.waitKey(0)
    return

def hog_svm(path):
    winSize = (14,14)
    blockSize = (7,7) 
    blockStride = (1,1) 
    cellSize = (7,7)
    nBin = 9 
    paths_list = ['sample/0/0.png','sample/1/0.png','sample/2/0.png','sample/3/0.png','sample/4/0.png',\
                    'sample/5/0.png','sample/6/0.png','sample/7/0.png','sample/8/0.png','sample/9/0.png']

    # 第二步：创造一个HOG描述子和检测器
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nBin)
    # 第三步：启动SVM分离器
    svm = cv2.ml.SVM_create()

    # 第四步：计算Hog
    featureNum = 5184
    # 窗口对应的一维特征向量维数n
    featureArray = np.zeros(((10),featureNum),np.float32)
    # 创建Hog特征矩阵
    labelArray = np.zeros(((10),1),np.int32)

    for i in range(10):
        fileName = paths_list[i]
        # 导入正样本图片
        img = cv2.imread(fileName)
        img = cv2.resize(img, (28, 28))
        hist = hog.compute(img,(5,5))
        for j in range(0,featureNum):
            featureArray[i,j] = hist[j]

        labelArray[i,0] = i
        
    # SVM属性设置    
    svm.setType(cv2.ml.SVM_C_SVC)
    # SVM模型类型：C_SVC表示SVM分类器，C_SVR表示SVM回归
    svm.setKernel(cv2.ml.SVM_LINEAR)
    # 核函数类型： LINEAR：线性核函数（linear kernel），POLY:多项式核函数（ploynomial kernel），RBF:径向机核函数(radical basis function)，SIGMOID: 神经元的非线性作用函数核函数(Sigmoid tanh)，PRECOMPUTED：用户自定义核函数 
    svm.setC(0.01)
    # SVM类型（C_SVC/ EPS_SVR/ NU_SVR）的参数C,C表示惩罚因子，C越大表示对错误分类的惩罚越大

    # 第六步：训练函数
    ret = svm.train(featureArray,cv2.ml.ROW_SAMPLE,labelArray)

    imageSrc = cv2.imread(path)
    resized_img = cv2.resize(imageSrc, (28, 28))
    test_hist = hog.compute(resized_img,(5,5))
    testArray = np.zeros(((1),featureNum),np.float32)
    for j in range(0,featureNum):
        testArray[0,j] = test_hist[j]
    result1 = svm.predict(testArray)
    labels = int(result1[1].item())
    print("预测结果为:",labels)
    return