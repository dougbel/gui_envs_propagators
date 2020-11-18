import os
import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, jpg_frames_dir, frames_npz, frames_nums):
        super().__init__()
        self._run_flag = True
        self.frames_npz = frames_npz
        self.frames_nums = frames_nums
        self.jpg_frames_dir = jpg_frames_dir

    def run(self):
        i = 0
        while self._run_flag:
            frame_n = self.frames_nums[i]
            jpg_frame_file = os.path.join(self.jpg_frames_dir, "image_frame_" + str(frame_n) + "_input.jpg")
            cv_frame = cv2.imread(jpg_frame_file)
            self.change_pixmap_signal.emit(cv_frame)
            i += 1
            if i > len(self.frames_nums):
                i = 0
            time.sleep(0.3)

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()