import cv2
import numpy as np

def rgb2gray(img):
    new_figure = np.zeros([img.shape[0],img.shape[1]],dtype=np.uint8)
    b,g,r = cv2.split(img)   #提取 BGR 颜色通道
    for row in range(img.shape[0]):
        for colmun in range(img.shape[1]):
            new_figure[row][colmun] = int(0.3*r[row][colmun]+0.59*g[row][colmun]+0.11*b[row][colmun])
    return new_figure

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
    lrimg = cv2.Canny(filtered_img,low_threshold,high_threshold)

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

if __name__ == "__main__": 
    logo=cv2.imread("figures/logo_01.png")
    # logo=cv2.imread("figures/lena.jpg")
    new_img = canny(logo)
    cv2.imshow("image",new_img)
    cv2.waitKey(0)