import glob
import os
import sys
import gc

import trimesh
import vtk
import pickle
import json
import numpy as np
import open3d as o3d

from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QFileDialog
from vedo import Plotter, load, Points

from ctrl.pc_mesh_visualizer import CtrlPcMeshVisualizer
from qt_ui.Ui_pointcloud_mesh_loader import Ui_viewerWindow
from thirdparty.QJsonModel.qjsonmodel import QJsonModel

if __name__ == "__main__":
    v = CtrlPcMeshVisualizer()
    sys.exit()
