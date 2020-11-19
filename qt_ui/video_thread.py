import os
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
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height)  # , Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = False
        self.frames_npz_data = None
        self.frames_nums = None
        self.jpg_frames_dir = None
        self.current_pos = 0

    def set_sequences_files(self, frames_npz_data, frames_nums, jpg_frames_dir):
        self.frames_npz_data = frames_npz_data
        self.frames_nums = frames_nums
        self.jpg_frames_dir = jpg_frames_dir
        self.current_pos = 0

    def run(self):
        self._run_flag = True
        while self._run_flag and self.frames_npz_data is not None:
            str_pos = str(self.frames_nums[self.current_pos])
            jpg_frame_file = os.path.join(self.jpg_frames_dir, "image_frame_" + str_pos + "_input.jpg")
            cv_frame = cv2.imread(jpg_frame_file)
            h, w, c = cv_frame.shape
            npy_frame_data = self.frames_npz_data["image_frame_" + str_pos + "_scores_1.npy"]
            npy_frame_data = npy_frame_data * 255
            img_scores_255 = npy_frame_data.astype(np.uint8)
            red_mask = np.zeros((h, w, 3), dtype=np.uint8)
            red_mask[:, :, 1] = img_scores_255
            alpha = 0.55
            beta = (1.0 - alpha)
            img_bgr_masked = cv2.addWeighted(cv_frame, alpha, red_mask, beta, 0.0)
            self.change_pixmap_signal.emit(img_bgr_masked)
            self.msleep(40)
            self.current_pos += 1
            if self.current_pos >= len(self.frames_nums):
                self.current_pos = 0

    def pause(self):
        self._run_flag = False

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
        self.frames_npz_data = None
        self.frames_nums = None
        self.jpg_frames_dir = None
