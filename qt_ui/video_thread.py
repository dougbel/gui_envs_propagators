import os
import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from pydev_ipython.qt import QtGui


class SequenceImagesWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.display_width = 224
        self.display_height = 224
        self.image_label = QLabel(self)
        # grey = QPixmap(self.display_width, self.display_height)
        # grey.fill(QColor('white'))
        # self.image_label.setPixmap(grey)
        self.q_thread = VideoThread()
        # connect its signal to the update_image slot
        self.q_thread.change_pixmap_signal.connect(self.update_image)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height)#, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    npz_file = None
    frames_nums = None
    jpg_frames_dir = None

    def __init__(self):
        super().__init__()
        self._run_flag = False

    def set_sequences_files(self, npz_file, frames_nums, jpg_frames_dir):
        self.npz_file = npz_file
        self.frames_nums = frames_nums
        self.jpg_frames_dir = jpg_frames_dir

    # def run(self):
    #     # capture from web cam
    #     cap = cv2.VideoCapture(0)
    #     while self._run_flag:
    #         ret, cv_img = cap.read()
    #         if ret:
    #             self.change_pixmap_signal.emit(cv_img)
    #         self.msleep(0.3)
    #     # shut down capture system
    #     cap.release()


    def run(self):
        self._run_flag = True
        i = 0
        while self._run_flag and self.npz_file is not None:
            jpg_frame_file = os.path.join(self.jpg_frames_dir, "image_frame_" + str(self.frames_nums[i]) + "_input.jpg")
            cv_frame = cv2.imread(jpg_frame_file)
            self.msleep(40)
            self.change_pixmap_signal.emit(cv_frame)
            i += 1
            if i >= len(self.frames_nums):
                i = 0

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
