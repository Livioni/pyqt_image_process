# pyqt_image_process

Non opencv library implementation. Easy to code in python but super slow.

## Enviroment and Dependency

- macOS Monterey 12.6 Apple M1 Max
- Opencv-python 4.5.5 (Rosetta2)
- Pyqt 6.3.1
- matplotlib  3.5.3 (For ploting histogram)
- scipy 1.9.1 (For side window filter, extremly slow if dont use it.)

![截屏2022-09-16 23.45.43](README.assets/%E6%88%AA%E5%B1%8F2022-09-16%2023.45.43.png)

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

    1. [X] average filter $3\times 3$
    
    2. [X] median filter $3\times 3$
    
    3. [X] side window filter (average side window filter, also $3\times3$ )
    
    4. [X] morphological filter : This function is for homework exclusively, please select the exlusive image.
    
    5. [X] Diy gaussian filter :  Please input gaussain template:  like $[[1,2,1],[2,8,2],[1,2,1]]$ or $[[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]$
    
       
    
       

## Links

C++ Version: https://github.com/QiTianyu-0403/OpenCVImage

Canny: https://github.com/StefanPitur/Edge-detection---Canny-detector/blob/master/canny.py

Side window filter python implementation: https://github.com/Beta-y/Side_Window_Filtering
