import gc
import json
import os
import pickle
import sys
import re

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import vtk

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QFileDialog

from PyQt5.QtGui import QPixmap

from vedo import Plotter, load, Points, Lines, Spheres

from qt_ui.Ui_propagators_loader import Ui_MainWindow
from qt_ui.video_thread import VideoThread, SequenceImagesWidget
from si.scannet.datascannet import DataScanNet
from thirdparty.QJsonModel.qjsonmodel import QJsonModel


class CtrlPropagatorVisualizer:

    def __init__(self):
        self.scannet_data = None
        self.vtk_env = None
        self.vtk_pc_tested = None
        self.vtk_samples = None
        self.interactions = []
        self.affordances = []
        self.BATCH_PROPAGATION = 10000

        self.frames_npz_data = None
        self.frames_nums = None
        self.jpg_frames_dir = None
        self.thread = None

        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(MainWindow)
        self.lbl_results = SequenceImagesWidget(self.ui.centralwidget)
        #self.lbl_results.setGeometry(QtCore.QRect(870, 180, 224, 224))
        self.lbl_results.setAutoFillBackground(False)
        self.ui.vtk_widget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.ui.vtk_interaction.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self.vp = Plotter(qtWidget=self.ui.vtk_widget, bg="white")
        self.vp.show([], axes=0)
        vp = Plotter(qtWidget=self.ui.vtk_interaction, bg="white")
        vp.show([], axes=0)
        self.ui.lbl_results.setHidden(True)

        # ### BUTTON SIGNALS
        self.ui.btn_dataset.clicked.connect(
            lambda: self.click_set_dir(MainWindow, self.ui.line_dataset, "dataset"))
        self.ui.btn_results.clicked.connect(
            lambda: self.click_set_dir(MainWindow, self.ui.line_results, "results"))
        self.ui.btn_configurations.clicked.connect(
            lambda: self.click_set_dir(MainWindow, self.ui.line_configurations, "configurations"))
        # self.ui.btn_view_interaction.clicked.connect(self.click_view_interaction)
        self.ui.btn_add_sample.clicked.connect(self.click_add_sample_on_environment)
        self.ui.btn_show_samples.clicked.connect(self.click_show_samples_on_environment)
        self.ui.btn_play.clicked.connect(self.click_btn_play)

        # ### check box signal
        self.ui.chk_on_gray_env.stateChanged.connect(self.changed_chk_on_gray_env)
        self.ui.chk_on_tested_points.stateChanged.connect(self.changed_chk_tested_points)

        # ### INFO LOADERS
        # DATASET
        self.ui.line_dataset.textChanged.connect(lambda: self.update_list_environments(self.ui.line_dataset.text()))
        self.ui.line_configurations.textChanged.connect(lambda: self.update_list_interactions())
        # ITEM SELECTION
        self.ui.l_env.itemSelectionChanged.connect(self.update_visualized_environment)
        self.ui.l_interactions.itemSelectionChanged.connect(self.update_visualized_interaction)

        # ### WORKING INDEXES
        self.idx_env = None
        self.idx_iter = None

        # ### VARIABLES FOR WORKING GLOBALLY ON SELECTIONS
        self.train_data = None
        self.propagation_data = None
        self.np_pc_tested = None
        self.np_scores = None
        self.sampler = None

        self.defaults()

        MainWindow.show()
        sys.exit(app.exec_())

    def defaults(self):
        default_dataset = './data/scans'
        if os.path.exists(default_dataset):
            self.ui.line_dataset.setText(os.path.abspath(default_dataset))
        default_results = './data/train'
        if os.path.exists(default_results):
            self.ui.line_results.setText(os.path.abspath(default_results))
        default_configurations = './data/configs_exe'
        if os.path.exists(default_configurations):
            self.ui.line_configurations.setText(os.path.abspath(default_configurations))

    def click_set_dir(self, window, line, element):
        file_name = str(QFileDialog.getExistingDirectory(window, "Select " + element + " directory"))
        if file_name:
            line.setText(file_name)

    def update_vtk(self, vtk_env, vtk_points=None, vtk_samples=None, camera=None, resetcam=False):
        self.vp = Plotter(qtWidget=self.ui.vtk_widget, bg="white")
        if camera is not None:
            self.vp.camera = camera

        actors = [vtk_env]
        if self.ui.chk_on_tested_points.isChecked() and vtk_points is not None:
            actors.append(vtk_points)
        if vtk_samples is not None:
            [actors.append(sample) for sample in vtk_samples]

        self.vp.show(*actors, axes=1, resetcam=resetcam)
        gc.collect()

    def __load_env_from_hdd(self):
        self.vtk_env = load(self.scannet_data.env_files_decimated[self.idx_env])
        if self.ui.chk_on_gray_env.isChecked():
            self.vtk_env.c((.4, 0.4, 0.4))

    def changed_chk_on_gray_env(self):
        if len(self.ui.l_env.selectedIndexes()) > 0:
            old_camera = self.vp.camera
            self.__load_env_from_hdd()
            if self.idx_iter is None:
                self.update_vtk(vtk_env=self.vtk_env, camera=old_camera)
            else:
                self.update_vtk(vtk_env=self.vtk_env, vtk_points=self.vtk_pc_tested, camera=old_camera)

    def changed_chk_tested_points(self):
        old_camera = self.vp.camera
        self.update_vtk(vtk_env=self.vtk_env,
                        vtk_points=self.vtk_pc_tested,
                        vtk_samples=self.vtk_samples,
                        camera=old_camera)

    def update_visualized_environment(self):
        resetcam = self.idx_env is None

        if len(self.ui.l_env.selectedIndexes()) > 0:
            self.idx_env = self.ui.l_env.selectedIndexes()[0].row()

            self.__load_env_from_hdd()

            scan = self.scannet_data.scans[self.idx_env]
            self.color_interaction_propagated_in_env(scan)

            if self.idx_iter is None:
                self.update_vtk(vtk_env=self.vtk_env, resetcam=True)
            else:
                # self.update_vtk(vtk_env=self.vtk_env, resetcam=resetcam)
                self.update_visualized_interaction(resetcam=resetcam)

    def color_interaction_propagated_in_env(self, scan):
        if scan is not None:
            path_results = self.ui.line_results.text()
            tested = []
            dir_tested = os.path.join(path_results, "env_test", scan)
            if os.path.exists(dir_tested):
                tested = os.listdir(dir_tested)

            calc_prop = []
            dir_calc_prop = os.path.join(path_results, "propagators", scan)
            if os.path.exists(dir_calc_prop):
                calc_prop = os.listdir(dir_calc_prop)

            propagated = []
            dir_propagated = os.path.join(path_results, "frame_propagation", scan)
            if os.path.exists(dir_propagated):
                propagated = os.listdir(dir_propagated)
                propagated = [re.sub('_img_segmentation_w[0-9]+_x_h[0-9]+$', '', i) for i in propagated]

            for i in range(self.ui.l_interactions.count()):
                affordance = self.affordances[i]
                if affordance in tested and affordance in calc_prop and affordance in propagated:
                    self.ui.l_interactions.item(i).setForeground(Qt.black)
                else:
                    self.ui.l_interactions.item(i).setForeground(Qt.red)

    def update_list_environments(self, path_dataset):
        self.ui.l_env.clear()
        self.scannet_data = DataScanNet(path_dataset)

        for scan in self.scannet_data.scans:
            self.ui.l_env.addItem(scan)

    def update_visualized_interaction(self, resetcam=False):
        old_camera = self.vp.camera
        self.idx_iter = None

        if len(self.ui.l_interactions.selectedIndexes()) > 0:
            self.idx_iter = self.ui.l_interactions.selectedIndexes()[0].row()

            if self.idx_env is not None:
                self.train_data = None
                self.propagation_data = None
                self.np_pc_tested = None
                self.np_scores = None
                self.sampler = None
                self.vtk_pc_tested = None

                self.ui.chk_on_tested_points.setChecked(True)
                self.update_vtk(vtk_env=self.vtk_env, camera=old_camera, resetcam=resetcam)

                npy_samples_scores_file = os.path.join(self.ui.line_results.text(), "samples_propagation",
                                                       self.scannet_data.scans[self.idx_env],
                                                       self.affordances[self.idx_iter],
                                                       "samples_propagated_scores_1.npy")
                npy_samples_scores = np.load(npy_samples_scores_file)

                self.np_pc_tested = npy_samples_scores[:, 0:3]
                self.vtk_pc_tested = Points(self.np_pc_tested, r=3, alpha=1, c='blue')
                self.np_scores = npy_samples_scores[:, 3]
                self.vtk_pc_tested.cellColors(self.np_scores, cmap='jet_r', vmin=0, vmax=1)
                self.vtk_pc_tested.addScalarBar(c='jet_r', nlabels=5, pos=(0.8, 0.25))

                self.update_vtk(vtk_env=self.vtk_env, vtk_points=self.vtk_pc_tested, camera=old_camera)

                self.update_data()
                self.update_segmentation_video()

    def update_list_interactions(self, scan=None):
        self.interactions.clear()
        path_configurations = self.ui.line_configurations.text()
        path_descriptors = os.path.join(path_configurations, "descriptor_repository")
        if os.path.exists(path_descriptors):
            self.interactions = os.listdir(path_descriptors)
            self.interactions.sort()
            for inter in self.interactions:
                self.ui.l_interactions.addItem(inter)
                self.affordances.append([Path(aff).stem for aff in os.listdir(os.path.join(path_descriptors, inter)) if
                                         aff.endswith('.json')][0])

    def update_data(self):
        train_model = QJsonModel()
        self.ui.tree_train.setModel(train_model)

        if self.idx_iter is not None:
            with open(self.json_training_file()) as f:
                self.train_data = json.load(f)
            train_model.load(self.train_data)

            self.ui.tree_train.header().resizeSection(0, 200)
            self.ui.tree_train.expandAll()

            propagation_model = QJsonModel()
            self.ui.tree_propagation.setModel(propagation_model)

            with open(self.json_propagation_file()) as f:
                self.propagation_data = json.load(f)
            propagation_model.load(self.propagation_data)
            self.ui.tree_propagation.header().resizeSection(0, 200)

            self.update_vtk_interaction()

    def update_segmentation_video(self):
        self.ui.btn_pause.setEnabled(True)
        self.ui.btn_play.setEnabled(True)
        self.ui.btn_stop.setEnabled(True)
        img_width = 224
        img_height = 224
        stride = 10
        npz_file = os.path.join(self.ui.line_results.text(), "frame_propagation", self.scannet_data.scans[self.idx_env],
                                self.affordances[self.idx_iter] + "_img_segmentation_w224_x_h224", "scores_1.npz")
        self.frames_npz_data = np.load(npz_file)
        self.frames_nums = [int(i[12:i.find("_scores_1.npy")]) for i in self.frames_npz_data.files]

        self.jpg_frames_dir = os.path.join(self.ui.line_results.text(), "frame_img_samplings",
                                           self.scannet_data.scans[self.idx_env],
                                           "w" + str(img_width) + "h" + str(img_height) + "s" + str(stride))

        jpg_frame_file = os.path.join(self.jpg_frames_dir, "image_frame_" + str(self.frames_nums[0]) + "_input.jpg")
        cv_frame = cv2.imread(jpg_frame_file)

        # self.ui.lbl_results.setPixmap(CtrlPropagatorVisualizer.convert_cv_qt(cv_frame, img_width, img_height))
        self.ui.lbl_results.show()

    def click_btn_play(self):
        print("not implemented")
        # self.thread = VideoThread(self.jpg_frames_dir, self.frames_npz_data, self.frames_nums)
        # self.thread.change_pixmap_signal.connect(self.update_image)
        # self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img, 224, 224)
        self.ui.lbl_results.setPixmap(qt_img)

    @staticmethod
    def convert_cv_qt(cv_img, display_width, display_height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(display_width, display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def __iter_meshes_files(self):
        training_path = self.training_path()

        env_file = os.path.join(training_path,
                                self.train_data['affordance_name'] + '_' + self.train_data[
                                    'obj_name'] + '_environment.ply')
        obj_file = os.path.join(training_path,
                                self.train_data['affordance_name'] + '_' + self.train_data[
                                    'obj_name'] + '_object.ply')
        ibs_file = os.path.join(training_path,
                                self.train_data['affordance_name'] + '_' + self.train_data[
                                    'obj_name'] + '_ibs_mesh_segmented.ply')

        return env_file, obj_file, ibs_file

    def update_vtk_interaction(self):
        env_file, obj_file, ibs_file = self.__iter_meshes_files()
        vp = Plotter(qtWidget=self.ui.vtk_interaction, bg="white")
        vp.load(env_file).c((.7, .7, .7)).alpha(.6)
        vp.load(obj_file).c((0, 1, 0)).alpha(.78)
        vp.load(ibs_file).c((0, 0, 1)).alpha(.39)
        training_path = self.training_path()
        aff = self.train_data['affordance_name']
        obj_name = self.train_data['obj_name']
        # provenance vectors
        pv_sample_size = self.train_data["trainer"]["sampler"]["sample_size"]
        pv_pnt_file = os.path.join(training_path, 'UNew_' + aff + '_' + obj_name + '_descriptor_8_points.pcd')
        pv_vec_file = os.path.join(training_path, 'UNew_' + aff + '_' + obj_name + '_descriptor_8_vectors.pcd')
        pv_points = np.asarray(o3d.io.read_point_cloud(pv_pnt_file).points)[0:pv_sample_size]
        pv_vectors = np.asarray(o3d.io.read_point_cloud(pv_vec_file).points)[0:pv_sample_size]
        provenance_vectors = Lines(pv_points, pv_points + pv_vectors, c='red', alpha=1).lighting("plastic")
        vp += provenance_vectors
        # clearance vectors
        cv_sample_size = self.train_data["trainer"]["cv_sampler"]["sample_clearance_size"]
        cv_pnt_file = os.path.join(training_path, 'UNew_' + aff + '_' + obj_name + '_descriptor_8_clearance_points.pcd')
        cv_vct_file = os.path.join(training_path,
                                   'UNew_' + aff + '_' + obj_name + '_descriptor_8_clearance_vectors.pcd')
        cv_points = np.asarray(o3d.io.read_point_cloud(cv_pnt_file).points)[0:cv_sample_size]
        cv_vectors = np.asarray(o3d.io.read_point_cloud(cv_vct_file).points)[0:cv_sample_size]
        clearance_vectors = Lines(cv_points, cv_points + cv_vectors, c='yellow', alpha=1).lighting("plastic")
        cv_from = Spheres(cv_points, r=.003, c="yellow", alpha=1).lighting("plastic")
        vp += clearance_vectors
        vp += cv_from

        vp.show(axes=1)

        self.ui.btn_add_sample.setEnabled(True)
        self.ui.btn_show_samples.setEnabled(False)

    def click_add_sample_on_environment(self):
        old_camera = self.vp.camera

        if self.idx_env is not None and self.idx_iter is not None:
            path_prop = os.path.join(self.ui.line_results.text(),
                                     "env_test",
                                     self.scannet_data.scans[self.idx_env],
                                     self.affordances[self.idx_iter])
            csv_scores_file = os.path.join(path_prop, "test_scores.csv")

            json_propagation_file = self.json_propagation_file()

            if self.sampler is None:
                __, obj_file, __ = self.__iter_meshes_files()
                self.sampler = CtrlPropagatorSampler(self.np_pc_tested, self.np_scores,
                                                     csv_scores_file, json_propagation_file, obj_file)

            obj_vtk = [self.sampler.get_sample()]

            self.update_vtk(vtk_env=self.vtk_env, vtk_points=self.vtk_pc_tested, vtk_samples=obj_vtk, camera=old_camera)

            self.ui.btn_show_samples.setEnabled(True)

    def json_propagation_file(self):
        conf_prop_dir = os.path.join(self.ui.line_configurations.text(), "json_propagators",
                                     self.affordances[self.idx_iter])
        return os.path.join(conf_prop_dir, "propagation_data.json")

    def json_training_file(self):
        return os.path.join(self.training_path(), self.affordances[self.idx_iter] + ".json")

    def training_path(self):
        return os.path.join(self.ui.line_configurations.text(), "descriptor_repository",
                            self.interactions[self.idx_iter])

    def propagator_file(self):
        return os.path.join(self.ui.line_results.text(), "propagators", self.scannet_data.scans[self.idx_env],
                            self.affordances[self.idx_iter], "propagation_rbf.pkl")

    def samples_file(self):
        path_samples = os.path.join(self.ui.line_results.text(), "samples", self.scannet_data.scans[self.idx_env])
        npy_samples_file = os.path.join(path_samples, "sample_points.npy")
        return npy_samples_file

    def click_show_samples_on_environment(self):
        old_camera = self.vp.camera

        self.update_vtk(vtk_env=self.vtk_env, vtk_points=self.vtk_pc_tested, vtk_samples=self.sampler.vtk_samples,
                        camera=old_camera)


class CtrlPropagatorSampler:

    def __init__(self, np_pc_tested, np_scores, csv_file_scores, json_propagation_file, ply_obj_file):
        self.np_pc_tested = np_pc_tested
        self.np_scores = np_scores
        self.csv_file_scores = csv_file_scores
        self.json_file_propagation = json_propagation_file
        self.ply_obj_file = ply_obj_file
        self.votes = None
        self.pd_best_scores = None
        self.idx_votes = -1
        self.vtk_samples = []

    def filter_data_scores(self):
        with open(self.json_file_propagation) as f:
            propagation_data = json.load(f)

        max_score = propagation_data['max_limit_score']
        max_missing = propagation_data['max_limit_missing']
        max_collided = propagation_data['max_limit_cv_collided']

        pd_scores = pd.read_csv(self.csv_file_scores)

        filtered_df = pd_scores.loc[(pd_scores.score.notnull()) &  # avoiding null scores (bar normal environment)
                                    (pd_scores.missings <= max_missing) &  # avoiding scores with more than max missing
                                    (pd_scores.score <= max_score) &
                                    (pd_scores.cv_collided <= max_collided),
                                    pd_scores.columns != 'interaction']  # returning all columns but interaction name

        return filtered_df.loc[filtered_df.groupby(['point_x', 'point_y', 'point_z'])['score'].idxmin()]

    def get_sample(self):
        if self.votes is None:
            self.pd_best_scores = self.filter_data_scores()
            self.votes = self.generate_votes()

        while True:
            self.idx_votes += 1
            idx_sample = self.votes[self.idx_votes][0]
            point_sample = self.np_pc_tested[idx_sample]
            angle_sample = self.angle_with_best_score(x=point_sample[0], y=point_sample[1], z=point_sample[2])
            if angle_sample != -1:
                vtk_object = load(self.ply_obj_file)
                vtk_object.rotate(angle_sample, axis=(0, 0, 1), rad=True)
                print(point_sample, " at ", angle_sample)
                vtk_object.pos(x=point_sample[0], y=point_sample[1], z=point_sample[2])
                self.vtk_samples.append(vtk_object)
                break
        return vtk_object

    def generate_votes(self):
        sum_mapped_norms = sum(self.np_scores)
        probabilities = [float(score) / sum_mapped_norms for score in self.np_scores]
        n_rolls = 10 * self.np_scores.shape[0]
        rolls = np.random.choice(self.np_scores.shape[0], n_rolls, p=probabilities)
        return Counter(rolls).most_common()

    def angle_with_best_score(self, x, y, z):
        angles = self.pd_best_scores[(self.pd_best_scores['point_x'].round(decimals=5) == round(x, 5)) &
                                     (self.pd_best_scores['point_y'].round(decimals=5) == round(y, 5)) &
                                     (self.pd_best_scores['point_z'].round(decimals=5) == round(z, 5))].angle

        return angles.array[0] if (angles.shape[0] == 1) else -1
