# This Python file uses the following encoding: utf-8
import sys
import cv2
from PySide6.QtWidgets import QApplication, QWidget, QGraphicsScene, QFileDialog,QMessageBox,QInputDialog
from PySide6.QtGui import QPixmap, QImage
# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_Widget
from utility import rgb2gray,gray_histogram,histogram_equalize,gradient_sharpening,laplace_sharpening,\
                    roberts,sobel,laplace,krisch,canny,impluse_noise,gaussian_noise,\
                    mean_filter,median_filter,s_meanfilter,morphological_filter,diy_gaussian_filter,\
                    affine_transform,perspective_transform

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)
        self.setFixedSize(1664,737)

        #buttom
        self.ui.ClearButton.clicked.connect(self.clear_windows)
        self.ui.pushButton.clicked.connect(self.button_clicked)
        self.ui.pushButton_2.clicked.connect(self.button_2_clicked)
        self.ui.pushButton_3.clicked.connect(self.button_3_clicked)
        self.ui.pushButton_4.clicked.connect(self.button_4_clicked)
        self.ui.pushButton_5.clicked.connect(self.button_5_clicked)
        self.ui.pushButton_6.clicked.connect(self.button_6_clicked)
        self.ui.pushButton_7.clicked.connect(self.button_7_clicked)
        self.ui.pushButton_8.clicked.connect(self.button_8_clicked)
        self.ui.pushButton_9.clicked.connect(self.button_9_clicked)
        self.ui.pushButton_10.clicked.connect(self.button_10_clicked)
        self.ui.pushButton_11.clicked.connect(self.button_11_clicked)
        self.ui.pushButton_12.clicked.connect(self.button_12_clicked)
        self.ui.pushButton_13.clicked.connect(self.button_13_clicked)
        self.ui.pushButton_14.clicked.connect(self.button_14_clicked)
        self.ui.pushButton_15.clicked.connect(self.button_15_clicked)
        self.ui.pushButton_16.clicked.connect(self.button_16_clicked)
        self.ui.pushButton_17.clicked.connect(self.button_17_clicked)
        self.ui.pushButton_18.clicked.connect(self.button_18_clicked)
        self.ui.pushButton_19.clicked.connect(self.button_19_clicked)
        self.ui.pushButton_20.clicked.connect(self.button_20_clicked)
        self.ui.pushButton_21.clicked.connect(self.button_21_clicked)

        #graph views
        self.scene,self.scene_2,self.scene_3 = QGraphicsScene(),QGraphicsScene(),QGraphicsScene()
        self.ui.graphicsView.setScene(self.scene)
        self.ui.graphicsView.show()
        self.ui.graphicsView_2.setScene(self.scene_2)
        self.ui.graphicsView_2.show()
        self.ui.graphicsView_3.setScene(self.scene_3)
        self.ui.graphicsView_3.show()

        #variables 
        self.image_path = None
        self.image_matrix_buffer,self.image_matrix_buffer_2 = None,None
        self.image_pix_buffer,self.image_pix_buffer_2,self.image_pix_buffer_3 = None,None,None  

    def clear_windows(self):
        self.scene_2.clear()
        self.scene_3.clear()
        self.image_matrix_buffer_2 = None
        self.image_pix_buffer_2,self.image_pix_buffer_3 = None,None
        return

    def button_clicked(self):
        # read and show
        self.image_path = self.__load_image_path()
        if self.image_path:
            #clear all the buffer
            self.image_matrix_buffer = None
            self.image_pix_buffer_2,self.image_pix_buffer_3 = None,None
            self.scene_2.clear()
            self.scene_3.clear()
            img = cv2.imread(str(self.image_path))
            self.image_matrix_buffer = img
            self.image_pix_buffer = self.__conver2pixmap(img)
            self.__show_image(self.image_pix_buffer,self.scene) 
        return 

    def button_2_clicked(self):
        if self.image_pix_buffer_3 is not None:
            fileName_save = QFileDialog.getSaveFileName(self, "Save file",self.image_path,"Image file (*.jpg *.jpeg *.bmp *.png)")
            if fileName_save != ('',''):
                self.image_pix_buffer_3.save(fileName_save[0])
        return

    def button_3_clicked(self):
        if self.image_matrix_buffer is not None:
            new_img = rgb2gray(self.image_matrix_buffer)
            pixmap = self.__conver2pixmap(new_img)
            self.image_pix_buffer_3 = pixmap
            self.__show_image(pixmap,self.scene_3)
        return

    def button_4_clicked(self):
        if self.image_matrix_buffer is not None:
            gray_histogram(self.image_matrix_buffer)
        return

    def button_5_clicked(self):
        if self.image_matrix_buffer is not None:
            new_img = histogram_equalize(self.image_matrix_buffer)
            pixmap = self.__conver2pixmap(new_img)
            self.image_pix_buffer_3 = pixmap
            self.__show_image(pixmap,self.scene_3)
        return

    def button_6_clicked(self):
        if self.image_matrix_buffer is not None:
            new_img,gradmap = gradient_sharpening(self.image_matrix_buffer)
            pixmap = self.__conver2pixmap(new_img)
            gradpix = self.__conver2pixmap(gradmap)
            self.image_pix_buffer_2 = gradpix
            self.image_pix_buffer_3 = pixmap
            self.__show_image(gradpix,self.scene_2)
            self.__show_image(pixmap,self.scene_3)
        return

    def button_7_clicked(self):
        if self.image_matrix_buffer is not None:
            new_img,gradmap = laplace_sharpening(self.image_matrix_buffer)
            pixmap = self.__conver2pixmap(new_img)
            gradpix = self.__conver2pixmap(gradmap)
            self.image_pix_buffer_2 = gradpix
            self.image_pix_buffer_3 = pixmap
            self.__show_image(gradpix,self.scene_2)
            self.__show_image(pixmap,self.scene_3)
        return

    def button_8_clicked(self):
        if self.image_matrix_buffer is not None:
            new_img = roberts(self.image_matrix_buffer)
            pixmap = self.__conver2pixmap(new_img)
            self.image_pix_buffer_3 = pixmap
            self.__show_image(pixmap,self.scene_3)
        return
    
    def button_9_clicked(self):
        if self.image_matrix_buffer is not None:
            new_img = sobel(self.image_matrix_buffer)
            pixmap = self.__conver2pixmap(new_img)
            self.image_pix_buffer_3 = pixmap
            self.__show_image(pixmap,self.scene_3)
        return

    def button_10_clicked(self):
        if self.image_matrix_buffer is not None:
            new_img = laplace(self.image_matrix_buffer)
            pixmap = self.__conver2pixmap(new_img)
            self.image_pix_buffer_3 = pixmap
            self.__show_image(pixmap,self.scene_3)
        return

    def button_11_clicked(self):
        if self.image_matrix_buffer is not None:
            new_img = krisch(self.image_matrix_buffer)
            pixmap = self.__conver2pixmap(new_img)
            self.image_pix_buffer_3 = pixmap
            self.__show_image(pixmap,self.scene_3)
        return

    def button_12_clicked(self):
        if self.image_matrix_buffer is not None:
            new_img = canny(self.image_matrix_buffer)
            pixmap = self.__conver2pixmap(new_img)
            self.image_pix_buffer_3 = pixmap
            self.__show_image(pixmap,self.scene_3)
        return

    def button_13_clicked(self):
        if self.image_matrix_buffer_2 is not None:
            process_img = self.image_matrix_buffer_2
        elif self.image_matrix_buffer is not None: 
            process_img = self.image_matrix_buffer
        else:
            return
        new_img = mean_filter(process_img)
        pixmap = self.__conver2pixmap(new_img)
        self.image_pix_buffer_3 = pixmap
        self.__show_image(pixmap,self.scene_3)
        return
    
    def button_14_clicked(self):
        if self.image_matrix_buffer_2 is not None:
            process_img = self.image_matrix_buffer_2
        elif self.image_matrix_buffer is not None: 
            process_img = self.image_matrix_buffer
        else:
            return
        new_img = median_filter(process_img)
        pixmap = self.__conver2pixmap(new_img)
        self.image_pix_buffer_3 = pixmap
        self.__show_image(pixmap,self.scene_3)
        return

    def button_15_clicked(self):
        if self.image_matrix_buffer_2 is not None:
            process_img = self.image_matrix_buffer_2
        elif self.image_matrix_buffer is not None: 
            process_img = self.image_matrix_buffer
        else:
            return
        new_img = s_meanfilter(process_img,radius=1)
        pixmap = self.__conver2pixmap(new_img)
        self.image_pix_buffer_3 = pixmap
        self.__show_image(pixmap,self.scene_3)
        return

    def button_16_clicked(self):
        self.clear_windows()
        QMessageBox.information(self,"Warning","This function is for homework exclusively, please select the exlusive image.")
        img_path, _ = QFileDialog.getOpenFileName(self, 'Open file', filter="Image files (*.jpg *.jpeg *.bmp *.png) ") #pylint: disable=line-too-long
        if img_path == '':
            QMessageBox.information(self,"Warning","No file selected.")
            return
        else:  
            self.image_matrix_buffer = None
            img = cv2.imread(str(img_path))
            self.image_matrix_buffer = img
            self.image_pix_buffer = self.__conver2pixmap(img)
            self.__show_image(self.image_pix_buffer,self.scene) 
            self.image_matrix_buffer_2,self.image_matrix_buffer_3 = morphological_filter(self.image_matrix_buffer)

            pixmap1 = self.__conver2pixmap(self.image_matrix_buffer_2)
            pixmap2 = self.__conver2pixmap(self.image_matrix_buffer_3)
            self.image_pix_buffer_2 = pixmap1
            self.image_pix_buffer_3 = pixmap2
            self.__show_image(pixmap1,self.scene_2)
            self.__show_image(pixmap2,self.scene_3)
        return

    def button_17_clicked(self):
        if self.image_matrix_buffer is not None:
            text, ok = QInputDialog.getText(self, 'Gaussian', 'Please input gaussain template: \n like\"[[1,2,1],[2,8,2],[1,2,1]]\"')
            if ok:
                filtered_img = diy_gaussian_filter(self.image_matrix_buffer,template=text)
                pixmap = self.__conver2pixmap(filtered_img)
                self.image_pix_buffer_3 = pixmap
                self.__show_image(pixmap,self.scene_3)
        return

    def button_18_clicked(self):
        if self.image_matrix_buffer is not None:
            self.image_matrix_buffer_2 = impluse_noise(self.image_matrix_buffer)
            pixmap = self.__conver2pixmap(self.image_matrix_buffer_2)
            self.image_pix_buffer_2 = pixmap
            self.__show_image(pixmap,self.scene_2)
        return

    def button_19_clicked(self):
        if self.image_matrix_buffer is not None:
            self.image_matrix_buffer_2 = gaussian_noise(self.image_matrix_buffer)
            pixmap = self.__conver2pixmap(self.image_matrix_buffer_2)
            self.image_pix_buffer_2 = pixmap
            self.__show_image(pixmap,self.scene_2)
        return

    def button_20_clicked(self):
        if self.image_matrix_buffer is not None:
            self.image_matrix_buffer_3 = affine_transform(self.image_matrix_buffer)
            pixmap = self.__conver2pixmap(self.image_matrix_buffer_3)
            self.image_pix_buffer_3 = pixmap
            self.__show_image(pixmap,self.scene_3)
        return

    def button_21_clicked(self):
        if self.image_matrix_buffer is not None:
            self.image_matrix_buffer_3 = perspective_transform(self.image_matrix_buffer)
            pixmap = self.__conver2pixmap(self.image_matrix_buffer_3)
            self.image_pix_buffer_3 = pixmap
            self.__show_image(pixmap,self.scene_3)
        return        

    def __conver2pixmap(self,img):
        if len(img.shape) > 2:
            cvimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = QImage(cvimg, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
        else:
            frame = QImage(img, img.shape[1], img.shape[0], img.shape[1]*1, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(frame)
        return pixmap

    def __show_image(self,image_pixmap,position):
        # position: self.scene left box; self.scene_2 right box;
        if image_pixmap == None:
            return
        else:
            # image_pixmap = image_pixmap.scaledToHeight(469)
            position.clear()  #先清空上次的残留
            position.addPixmap(image_pixmap)
        return 

    def __load_image_path(self):
        img_path, _ = QFileDialog.getOpenFileName(self, 'Open file', filter="Image files (*.jpg *.jpeg *.bmp *.png) ") #pylint: disable=line-too-long
        if img_path == '':
            QMessageBox.information(self,"Warning","No file selected.")
            return
        else:  
            return img_path


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.setWindowTitle("Image Process Application")
    widget.show()
    sys.exit(app.exec())
