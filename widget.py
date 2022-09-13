# This Python file uses the following encoding: utf-8
import sys
import cv2
from PySide6.QtWidgets import QApplication, QWidget, QGraphicsScene, QFileDialog,QMessageBox
from PySide6.QtGui import QPixmap, QImage
# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_Widget
from utility import rgb2gray,gray_histogram,histogram_equalize

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)
        self.setFixedSize(1664,800)

        #buttom
        self.ui.pushButton.clicked.connect(self .button_clicked)
        self.ui.pushButton_2.clicked.connect(self.button_2_clicked)
        self.ui.pushButton_3.clicked.connect(self.button_3_clicked)
        self.ui.pushButton_4.clicked.connect(self.button_4_clicked)
        self.ui.pushButton_5.clicked.connect(self.button_5_clicked)

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
        self.image_matrix_buffer = None
        self.image_pix_buffer,self.image_pix_buffer_2,self.image_pix_buffer_3 = None,None,None  


    def button_clicked(self):
        # read and show
        #clear all the buffer
        self.image_path = None
        self.image_matrix_buffer = None
        self.image_pix_buffer,self.image_pix_buffer_2,self.image_pix_buffer_3 = None,None,None  
        self.image_path = self.__load_image_path()
        if self.image_path:
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
            self.image_pix_buffer_2 = pixmap
            self.__show_image(pixmap,self.scene_2)
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
