#coding:utf-8
#*********************************************************************************************************
'''
说明：利用python/numpy/opencv提取图像Haar特征
算法思路:
        1)以灰度图的方式加载图片,并计算可获取的Haar特征尺度数;
        2)计算对应的积分图像，以便于Haar特征的快速计算;
		3)利用积分图像计算图像X2类型的不同尺度Haar特征；
		4)获得的feature这个特征可用于Adaboost级联检测。
具体参数：输入图像大小（640,480），积分图像大小（641,481），
          初始尺度Haar_block_size = (24, 24), 宽和高同等放大，可获取20个尺度Haar特征，
          对于初始尺度：Haar特征图的大小（640-24+1， 480-24+1）= （617,457），该尺度Haar特征维度:281969
          总的Haar特征维度为：2164660
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

#获取积分图像
def integral( img ):
    #积分图像比原始图像多一行一列，积分图像第一行第一列为0
    integimg = np.zeros( shape = (img.shape[0] + 1, img.shape[1] + 1), dtype = np.int32 )
    for i in range( 1, img.shape[0] ):
        for j in range( 1, img.shape[1] ):
            integimg[i][j] = img[i][j] + integimg[i-1][j] + integimg[i][j-1] - integimg[i-1][j-1]
    # plt.imshow( integimg )
    # plt.show()
    print( 'Done!' )
    return integimg

#获取单一尺度的Haar特征
def haar_onescale( img, integimg, haarblock_width, haarblock_height  ):
    #步长为1， no padding
    haarimg = np.zeros( shape = ( img.shape[0] - haarblock_width + 1, img.shape[1] - haarblock_height + 1 ), dtype = np.int32 )
    # plt.imshow( haarimg )
    # plt.show()
    haar_feature_onescale = []
    for i in range( haarimg.shape[0] ):
        for j in range( haarimg.shape[1] ):
            #i,j映射回原图形的坐标
            m = haarblock_width + i
            n = haarblock_height + i
            haar_all = integimg[m][n] - integimg[m-haarblock_width][n] - integimg[m][n-haarblock_height] + integimg[m-haarblock_width][n-haarblock_height]
            haar_black = integimg[m][n- int( haarblock_height/2 )] - integimg[m-haarblock_width][n-int( haarblock_height/2 )]- integimg[m][n-haarblock_height] + integimg[m-haarblock_width][n-haarblock_height]
            #1*all - 2*black = white - black
            haarimg[i][j] = 1 * haar_all - 2 * haar_black
            haar_feature_onescale.append( haarimg[i][j] )
    # plt.imshow( haarimg )
    # plt.show()
    print( ' 当前尺度下的Haar特征维度为： {}'.format( len( haar_feature_onescale ) ) )
    
    return haar_feature_onescale

#获取全尺度下的Haar特征
def harr( haarblock_width, haarblock_height, Scale_num ):
    feature = []
    haar_num = 0
    for i in range( Scale_num):
        haarblock_width = i*haarblock_width + 24
        haarblock_height = i*haarblock_height + 24
        print( '    当前 Haarblock 尺度为: ( {}, {} )'.format( haarblock_height, haarblock_width ) ) 
        haar_feature_onescale = haar_onescale( img, integimg, haarblock_width, haarblock_height )
        haar_num += len( haar_feature_onescale ) 
        feature.append( haar_feature_onescale )
        haarblock_width = 24
        haarblock_height = 24
    #计算总的Haar特征维度
    print( '[INFO] 计算Haar特征维数' )
    print( '    Haar特征总的维度为： {}'.format( haar_num ) )
    return feature
    
if __name__ == '__main__':
    #以灰度图的方式读取图像
    img = cv2.imread( 'figures/lena.jpg', cv2.IMREAD_GRAYSCALE )
    if ( img is None ):
        print( 'Not read img.' )
    #确定Haarblock的大小
    haarblock_width = 24
    haarblock_height = 24
    width_limt = int( img.shape[0] / haarblock_width )
    height_limt = int( img.shape[1] / haarblock_height )
    print( '--行方向尺度个数为: {}， 列方向尺度个数为： {}'.format( width_limt, height_limt ) )
    #可获取的尺度数量
    Scale_num = min( height_limt, width_limt )
    print( '--可用尺度个数为： {}'.format( Scale_num ) )
    #获取积分图像
    print( '[INFO] 计算积分图像' )
    integimg = integral( img )

    print( '[INFO] 提取图像Haar特征' )
    haar_feature = harr( haarblock_width, haarblock_height, Scale_num  )
    
       

