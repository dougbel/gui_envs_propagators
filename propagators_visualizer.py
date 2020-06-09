import glob
import os
import sys

import trimesh
import vtk
import pickle
import json
import numpy as np
import open3d as o3d

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from vtkplotter import Plotter, load, show, Points, Cone

from qt_ui.Ui_propagators_loader import Ui_MainWindow
from si.scannet.datascannet import DataScanNet
from thirdparty.QJsonModel.qjsonmodel import QJsonModel


class View:

    def __init__(self):
        self.scannet_data = None
        self.vtk_env = None
        self.vtk_pc_tested = None
        self.train_data = None
        self.interactions = []
        self.BATCH_PROPAGATION = 10000

        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(MainWindow)
        self.ui.vtk_widget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.ui.vtk_interaction.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self.vp = Plotter(qtWidget=self.ui.vtk_widget, bg="white")
        self.vp.show([], axes=0)

        vp = Plotter(qtWidget=self.ui.vtk_interaction, bg="white")
        vp.show([], axes=0)

        # ### BUTTON SIGNALS
        self.ui.btn_dataset.clicked.connect(
            lambda: self.click_set_dir(MainWindow, self.ui.line_dataset, "dataset"))
        self.ui.btn_propagator.clicked.connect(
            lambda: self.click_set_dir(MainWindow, self.ui.line_propagators, "propagators"))
        self.ui.btn_descriptors.clicked.connect(
            lambda: self.click_set_dir(MainWindow, self.ui.line_descriptors, "descriptors"))
        self.ui.btn_view_interaction.clicked.connect(self.click_view_interaction)

        # ### check box signal
        self.ui.chk_on_gray.stateChanged.connect(self.update_color_env)

        # ### INFO LOADERS
        # DATASET
        self.ui.line_dataset.textChanged.connect(lambda: self.update_list_environments(self.ui.line_dataset.text()))
        self.ui.l_env.itemSelectionChanged.connect(self.update_visualized_environment)
        self.ui.line_propagators.textChanged.connect(
            lambda: self.update_list_interactions(self.ui.line_propagators.text()))
        self.ui.l_interactions.itemSelectionChanged.connect(self.update_visualized_interaction)

        # ### WORKING INDEXES
        self.idx_env = None
        self.idx_iter = None

        self.defaults()

        MainWindow.show()
        sys.exit(app.exec_())

    def defaults(self):
        default_dataset = './data/dataset'
        if os.path.exists(default_dataset):
            self.ui.line_dataset.setText(os.path.abspath(default_dataset))
        default_propagators = './data/propagators'
        if os.path.exists(default_propagators):
            self.ui.line_propagators.setText(os.path.abspath(default_propagators))
        default_descriptors = './data/descriptors_repository'
        if os.path.exists(default_descriptors):
            self.ui.line_descriptors.setText(os.path.abspath(default_descriptors))

    def click_set_dir(self, window, line, element):
        file_name = str(QFileDialog.getExistingDirectory(window, "Select " + element + " directory"))
        if file_name:
            line.setText(file_name)

    def update_vtk(self, *actors, camera=None, resetcam=False):
        self.vp = Plotter(qtWidget=self.ui.vtk_widget, bg="white")
        if camera is not None:
            self.vp.camera = camera
        self.vp.show(*actors, axes=1, resetcam=resetcam)

    def __load_env_from_hdd(self):
        if len(self.ui.l_env.selectedIndexes()) > 0:
            self.idx_env = self.ui.l_env.selectedIndexes()[0].row()
            if self.ui.chk_on_gray.isChecked():
                self.vtk_env = load(self.scannet_data.env_files_decimated[self.idx_env], alpha=1, c=(.4, 0.4, 0.4))
            else:
                self.vtk_env = load(self.scannet_data.env_files_decimated[self.idx_env], alpha=1)

    def update_color_env(self):
        if len(self.ui.l_env.selectedIndexes()) > 0:
            old_camera = self.vp.camera
            self.__load_env_from_hdd()
            if self.idx_iter is None:
                self.update_vtk([self.vtk_env], camera=old_camera)
            else:
                self.update_vtk([self.vtk_env, self.vtk_pc_tested], camera=old_camera)

    def update_visualized_environment(self):
        self.idx_env = None
        if len(self.ui.l_env.selectedIndexes()) > 0:
            self.__load_env_from_hdd()
            if self.idx_iter is None:
                self.update_vtk([self.vtk_env], resetcam=True)
            else:
                self.update_visualized_interaction()

    def update_list_environments(self, path_dataset):
        self.ui.l_env.clear()
        self.scannet_data = DataScanNet(path_dataset)

        for scan in self.scannet_data.scans:
            self.ui.l_env.addItem(scan)

    def update_visualized_interaction(self):
        oldcamera = self.vp.camera
        self.idx_iter = None

        if len(self.ui.l_interactions.selectedIndexes()) > 0:
            self.idx_iter = self.ui.l_interactions.selectedIndexes()[0].row()
            if self.idx_env is not None:
                self.update_vtk([self.vtk_env], camera=oldcamera)

                path_prop = os.path.join(self.ui.line_propagators.text(),
                                         self.interactions[self.idx_iter],
                                         self.scannet_data.scans[self.idx_env])

                propagator_file = os.path.join(path_prop, "propagation_rbf.pkl")
                pc_tested_file = os.path.join(path_prop, "test_tested_points.pcd")

                with open(propagator_file, 'rb') as rbf_file:
                    propagator = pickle.load(rbf_file)

                pc_tested_array = np.asarray(o3d.io.read_point_cloud(pc_tested_file).points)
                self.vtk_pc_tested = Points(pc_tested_array, r=3, alpha=1, c='blue')

                np_scores = np.array([])
                for j in range(0, pc_tested_array.shape[0], self.BATCH_PROPAGATION):
                    batch = pc_tested_array[j:j + self.BATCH_PROPAGATION]
                    temp = propagator(batch[:, 0], batch[:, 1], batch[:, 2])
                    np_scores = np.concatenate((np_scores, temp), axis=0)

                self.vtk_pc_tested.cellColors(np_scores, cmap='jet_r', vmin=0, vmax=1)

                self.update_vtk([self.vtk_env, self.vtk_pc_tested], camera=oldcamera)

                self.update_data()

    def update_list_interactions(self, path_propagators):
        self.interactions.clear()
        self.interactions = os.listdir(path_propagators)
        self.interactions.sort()
        for inter in self.interactions:
            self.ui.l_interactions.addItem(inter)

    def update_data(self):
        train_model = QJsonModel()
        self.ui.tree_train.setModel(train_model)

        if self.idx_iter is not None:
            path_prop = os.path.join(self.ui.line_descriptors.text(), self.interactions[self.idx_iter])

            files = glob.glob(os.path.join(path_prop, '*.json'))
            for json_file in files:
                if json_file != os.path.join(path_prop, "propagation_data.json"):
                    with open(json_file) as f:
                        self.train_data = json.load(f)
                    train_model.load(self.train_data)
                    break
            self.ui.tree_train.header().resizeSection(0, 200)
            self.ui.tree_train.expandAll()

            propagation_model = QJsonModel()
            self.ui.tree_propagation.setModel(propagation_model)
            propagation_data_json = os.path.join(path_prop, "propagation_data.json")
            with open(propagation_data_json) as f:
                propagation_model.load(json.load(f))
            self.ui.tree_propagation.header().resizeSection(0, 200)

            self.update_vtk_interaction()

    def __iter_meshes_files(self):
        path_prop = os.path.join(self.ui.line_descriptors.text(), self.interactions[self.idx_iter])

        ibs_file = os.path.join(path_prop,
                                self.train_data['affordance_name'] + '_' + self.train_data[
                                    'obj_name'] + '_ibs_mesh_segmented.ply')
        env_file = os.path.join(path_prop,
                                self.train_data['affordance_name'] + '_' + self.train_data[
                                    'obj_name'] + '_environment.ply')
        obj_file = os.path.join(path_prop,
                                self.train_data['affordance_name'] + '_' + self.train_data[
                                    'obj_name'] + '_object.ply')
        return env_file, obj_file, ibs_file

    def update_vtk_interaction(self):
        env_file, obj_file, ibs_file = self.__iter_meshes_files()
        vp = Plotter(qtWidget=self.ui.vtk_interaction, bg="white")
        vp.load(env_file, c=(.7, .7, .7), alpha=.6)
        vp.load(obj_file, c=(0, 1, 0), alpha=.78)
        vp.load(ibs_file, c=(0, 0, 1), alpha=.39)
        vp.show(axes=1)

        self.ui.btn_view_interaction.setEnabled(True)

    def click_view_interaction(self):
        env_file, obj_file, ibs_file = self.__iter_meshes_files()
        path_prop = os.path.join(self.ui.line_descriptors.text(), self.interactions[self.idx_iter])
        pv_begin_file = os.path.join(path_prop,
                                     'UNew_' + self.train_data['affordance_name'] + '_' + self.train_data[
                                         'obj_name'] + '_descriptor_8_points.pcd')
        pv_env_file = os.path.join(path_prop,
                                   'UNew_' + self.train_data['affordance_name'] + '_' + self.train_data[
                                       'obj_name'] + '_descriptor_8_vectors.pcd')

        tri_mesh_env = trimesh.load_mesh(env_file)
        tri_mesh_obj = trimesh.load_mesh(obj_file)
        tri_mesh_ibs = trimesh.load_mesh(ibs_file)
        pv_begin = np.asarray(o3d.io.read_point_cloud(pv_begin_file).points)[0:self.train_data['sample_size']]
        pv_end = np.asarray(o3d.io.read_point_cloud(pv_env_file).points)[0:self.train_data['sample_size']]
        pv = trimesh.load_path(np.hstack((pv_begin, pv_begin + pv_end)).reshape(-1, 2, 3))

        tri_mesh_env.visual.face_colors = [200, 200, 200, 150]
        tri_mesh_obj.visual.face_colors = [0, 255, 0, 200]
        tri_mesh_ibs.visual.face_colors = [0, 0, 255, 100]

        scene = trimesh.Scene([tri_mesh_obj, tri_mesh_env, tri_mesh_ibs, pv])
        scene.show(flags={'cull': False, 'wireframe': False, 'axis': False},
                   caption=self.train_data['affordance_name'] + ' ' + self.train_data['obj_name'])


if __name__ == "__main__":
    v = View()
    v.show()
