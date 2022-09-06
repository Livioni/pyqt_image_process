# This Python file uses the following encoding: utf-8
import sys
import cv2
from PySide6.QtWidgets import QApplication, QWidget, QGraphicsScene, QFileDialog, QPushButton
from PySide6.QtGui import QPixmap, QImage
# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_Widget

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self._show_image)
        self.ui.pushButton_2.clicked.connect(self._load_image)
        self.scene,self.scene_2 = QGraphicsScene(),QGraphicsScene()
        self.ui.graphicsView.setScene(self.scene)
        self.ui.graphicsView.show()
        self.ui.graphicsView_2.setScene(self.scene_2)
        self.ui.graphicsView_2.show()
        self.image_buffer = None

    def __resize_image(self,img, new_height=300):
        height = img.shape[0]
        width = img.shape[1]
        ratio = float(height / width)
        new_width =  int(300/ratio)
        new_img = cv2.resize(img, (new_width, new_height))
        return new_img

    def _show_image(self):

        if self.image_buffer == None:
            print("No file selected!")
            return
        else:
            img = cv2.imread(str(self.image_buffer))
            img = self.__resize_image(img)
            cvimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            y, x = img.shape[:-1]
            frame = QImage(cvimg, x, y, x*3,QImage.Format_RGB888)
            self.scene.clear()  #先清空上次的残留
            self.pix = QPixmap.fromImage(frame)
            self.scene.addPixmap(self.pix)
            self.scene_2.addPixmap(self.pix)
            return

    
    def _load_image(self):
        img_path, _ = QFileDialog.getOpenFileName(self, 'Open file', filter="Image files (*.jpg *.jpeg *.bmp *.png) ") #pylint: disable=line-too-long
        if img_path == '':
            return
        else:  
            self.image_buffer = img_path
            return 


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.setWindowTitle("Image Process Application")
    widget.show()
    sys.exit(app.exec())
