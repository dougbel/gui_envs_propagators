import os
import sys
from pathlib import Path

import vtk

import numpy as np
from PyQt5 import QtWidgets
from PyQt5 import Qt
from qtconsole.qt import QtGui, QtCore
from vedo import Plotter, Points
from vedo.colors import colors

from qt_ui.Ui_pointcloud_mesh_loader import Ui_viewerWindow


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


class CtrlPcMeshVisualizer:

    available_colours = ["blue", "green", "red", "orange", "brown", "black", "indigo"]

    def __init__(self):
        app = Qt.QApplication(sys.argv)
        self.ui = Ui_viewerWindow(self)
        self.ui.qvtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.vp = Plotter(qtWidget=self.ui.qvtkWidget, bg="white")
        self.vp.show([], axes=0)

        self.items_vtk = []
        self.items_file_path = []
        self.items_file_name = []
        self.items_file_ext = []

        # self.ui.table_cloud.clicked.connect(self.open_menu)
        # self.ui.table_cloud.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

        self.ui.table_cloud.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.context_import = QtWidgets.QAction("Delete")
        self.context_import.triggered.connect(self.delete_item)
        self.ui.table_cloud.addAction(self.context_import)

        app.aboutToQuit.connect(self.ui.onClose)  # <-- connect the onClose event
        app.exec_()

    def delete_item(self):
        row = self.ui.table_cloud.currentRow()
        del self.items_vtk[row]
        del self.items_file_path[row]
        del self.items_file_name[row]
        del self.items_file_ext[row]
        self.ui.table_cloud.removeRow(row)

        self.update_vtk()

    def add_visualization(self, file_path):
        print(file_path)
        ext = os.path.splitext(file_path)[1]

        if ext == ".npy":
            np_pc = np.load(file_path)
            file_name = Path(file_path).stem
            colour_name = self.available_colours[len(self.items_vtk) % len(self.available_colours)]
            colour_hex = colors[colour_name]
            colour_rgb = hex_to_rgb(colour_hex)
            vtk_pc = Points(np_pc, r=3, alpha=1, c=colour_name)
            self.items_vtk.append(vtk_pc)
            self.items_file_path.append(file_path)
            self.items_file_name.append(file_name)
            self.items_file_ext.append(ext)
            row_position = self.ui.table_cloud.rowCount()
            self.ui.table_cloud.insertRow(row_position)
            self.ui.table_cloud.setItem(row_position, 0, QtGui.QTableWidgetItem(""))
            self.ui.table_cloud.item(row_position, 0).setBackground(QtGui.QColor(colour_rgb[0], colour_rgb[1], colour_rgb[2]))
            self.ui.table_cloud.setItem(row_position, 1, QtGui.QTableWidgetItem("..."+file_path[-50:]))
        self.update_vtk()

    def update_vtk(self):
        old_camera = self.vp.camera
        self.vp = Plotter(qtWidget=self.ui.qvtkWidget, bg="white")
        self.vp.camera = old_camera
        self.vp.show(self.items_vtk, axes=4)
        print("updated")
