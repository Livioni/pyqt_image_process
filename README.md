# pyqt_image_process

Non opencv library implementation. Easy to code in python but super slow.

## Enviroment and Dependency

- macOS Monterey 12.6 Apple M1 Max
- Opencv-python 4.5.5 (Rosetta2)
- Pyqt 6.3.1
- matplotlib  3.5.3 (For ploting histogram)
- scipy 1.9.1 (For side window filter, extremly slow if dont use it.)

![截屏2022-09-27 20.05.58](README.assets/%E6%88%AA%E5%B1%8F2022-09-27%2020.05.58.png)

<img src="README.assets/%E6%88%AA%E5%B1%8F2022-10-16%2020.23.07.png" alt="截屏2022-10-16 20.23.07" style="zoom:50%;" />

## Functions

1. [X] image intensification

    1. [X] gray scale
    2. [X] gray histogram
    3. [X] histogram equalization
    4. [X] gradient sharpen
    5. [X] Laplace sharpen
2. [X] edge detection

    1. [X] Roberts
    2. [X] Sobel
    3. [X] Laplace
    4. [X] Krisch
    5. [X] Canny
3. [X] Filter

    1. [X] average filter $3\times 3$.
    2. [X] median filter $3\times 3$.self.ui.pushButton_26.clicked.connect(self.button_26_clicked)
    3. [X] side window filter (average side window filter, also $3\times3$ ).
    4. [X] morphological filter : This function is for homework exclusively, please select the exlusive image.
    5. [X] Diy gaussian filter :  Please input gaussain template:  like $[[1,2,1],[2,8,2],[1,2,1]]$ or $[[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]$.
4. [X] Camera Calibration
5. [X] Bi-Camera Calibration
6. [X] Detections

    1. [X] Threshold
    2. [X] OTSU
    3. [X] Kittler
    4. [X] Single Gaussian Model
    5. [X] Mixed Gaussian Model
    6. [X] Hsv Histogram Matching
    7. [X] Template Matching
    8. [X] Local Binary Pattern
    9. [X] SIFT

7. Machine Learning

   [X] SVM

   [X] SVM+HOG
   
   [X] Haar features
   
   [X] Camshift

## Links

C++ Version: https://github.com/QiTianyu-0403/OpenCVImage

Canny: https://github.com/StefanPitur/Edge-detection---Canny-detector/blob/master/canny.py

Side window filter python implementation: https://github.com/Beta-y/Side_Window_Filtering

Camera Calibration: https://blog.csdn.net/qq_41035283/article/details/123778452

Bi-Camera Calibration: https://blog.csdn.net/qq_36076137/article/details/118383472

OTSU: https://blog.csdn.net/laonafahaodange/article/details/123746067

Mixed Gaussian background modeling : https://zhuanlan.zhihu.com/p/90103849

Local Binary Pattern: https://zhuanlan.zhihu.com/p/91768977

SIFT : https://github.com/rmislam/PythonSIFT/blob/master/pysift.py

Yolo_v5: https://github.com/ultralytics/yolov5
