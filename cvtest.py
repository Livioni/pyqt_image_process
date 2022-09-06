import cv2
# 读一个图片并进行显示(图片路径需自己指定)
lena=cv2.imread("logo_01.png")
cv2.imshow("image",lena)
cv2.waitKey(0)
