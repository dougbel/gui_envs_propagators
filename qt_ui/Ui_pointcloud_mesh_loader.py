# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pointcloud_mesh_loader.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QMainWindow
from PyQt5 import Qt


class Ui_viewerWindow(QMainWindow):
    def __init__(self, controller, parent=None):
        # ###### No update on changing the layout
        Qt.QMainWindow.__init__(self, parent)
        self.setObjectName("viewerWindow")
        self.ctrl = controller
        self.setAcceptDrops(True)
        # ######

        self.resize(1389, 831)
        self.centralWidget = QtWidgets.QWidget(self)
        self.centralWidget.setObjectName("centralWidget")
        self.qvtkWidget = QVTKRenderWindowInteractor(self.centralWidget)
        self.qvtkWidget.setGeometry(QtCore.QRect(520, 10, 861, 741))
        self.qvtkWidget.setAutoFillBackground(False)
        self.qvtkWidget.setObjectName("qvtkWidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 680, 121, 101))
        self.groupBox.setObjectName("groupBox")
        self.btn_getCamera = QtWidgets.QPushButton(self.groupBox)
        self.btn_getCamera.setGeometry(QtCore.QRect(10, 30, 91, 22))
        self.btn_getCamera.setObjectName("btn_getCamera")
        self.btn_resetCamera = QtWidgets.QPushButton(self.groupBox)
        self.btn_resetCamera.setGeometry(QtCore.QRect(10, 60, 91, 22))
        self.btn_resetCamera.setObjectName("btn_resetCamera")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 10, 481, 651))
        self.groupBox_2.setObjectName("groupBox_2")
        self.table_cloud = QtWidgets.QTableWidget(self.groupBox_2)
        self.table_cloud.setGeometry(QtCore.QRect(10, 30, 471, 531))
        self.table_cloud.setMinimumSize(QtCore.QSize(0, 0))
        self.table_cloud.setBaseSize(QtCore.QSize(0, 0))
        self.table_cloud.setObjectName("table_cloud")
        self.table_cloud.setColumnCount(2)
        self.table_cloud.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.table_cloud.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_cloud.setHorizontalHeaderItem(1, item)
        self.table_cloud.horizontalHeader().setDefaultSectionSize(30)
        self.table_cloud.horizontalHeader().setStretchLastSection(True)
        self.btn_loadFile = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_loadFile.setGeometry(QtCore.QRect(390, 570, 91, 21))
        self.btn_loadFile.setObjectName("btn_loadFile")
        self.table_cloud_color = QtWidgets.QTableWidget(self.groupBox_2)
        self.table_cloud_color.setEnabled(True)
        self.table_cloud_color.setGeometry(QtCore.QRect(270, 610, 205, 31))
        self.table_cloud_color.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.table_cloud_color.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table_cloud_color.setColumnCount(8)
        self.table_cloud_color.setObjectName("table_cloud_color")
        self.table_cloud_color.setRowCount(0)
        self.table_cloud_color.horizontalHeader().setVisible(False)
        self.table_cloud_color.horizontalHeader().setDefaultSectionSize(25)
        self.table_cloud_color.horizontalHeader().setMinimumSectionSize(25)
        self.table_cloud_color.horizontalHeader().setStretchLastSection(False)
        self.table_cloud_color.verticalHeader().setVisible(False)
        self.label_cloud = QtWidgets.QLabel(self.groupBox_2)
        self.label_cloud.setGeometry(QtCore.QRect(270, 590, 201, 16))
        self.label_cloud.setObjectName("label_cloud")
        self.spinPointSize = QtWidgets.QSpinBox(self.groupBox_2)
        self.spinPointSize.setGeometry(QtCore.QRect(430, 0, 51, 21))
        self.spinPointSize.setProperty("value", 3)
        self.spinPointSize.setObjectName("spinPointSize")
        self.chkSeeCoordenate = QtWidgets.QCheckBox(self.centralWidget)
        self.chkSeeCoordenate.setGeometry(QtCore.QRect(1240, 755, 141, 20))
        self.chkSeeCoordenate.setObjectName("chkSeeCoordenate")
        self.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(self)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1389, 20))
        self.menuBar.setObjectName("menuBar")
        self.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(self)
        self.mainToolBar.setObjectName("mainToolBar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(self)
        self.statusBar.setObjectName("statusBar")
        self.setStatusBar(self.statusBar)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

        # ###### No update on changing the layout
        self.show()
        # ######

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("viewerWindow", "Point cloud/Mesh Viewer"))
        self.groupBox.setTitle(_translate("viewerWindow", "Camera"))
        self.btn_getCamera.setText(_translate("viewerWindow", "Get position"))
        self.btn_resetCamera.setText(_translate("viewerWindow", "Reset"))
        self.groupBox_2.setTitle(_translate("viewerWindow", "Point clouds"))
        item = self.table_cloud.horizontalHeaderItem(1)
        item.setText(_translate("viewerWindow", "item"))
        self.btn_loadFile.setText(_translate("viewerWindow", "Open file"))
        self.spinPointSize.setToolTip(_translate("viewerWindow", "Size point"))
        self.chkSeeCoordenate.setText(_translate("viewerWindow", "Coordenate system"))

    def dropEvent(self, event):
        # if event.mimeData().hasImage:
        # event.setDropAction(Qt.CopyAction)
        for file in event.mimeData().urls():
            file_path = file.toLocalFile()
            self.ctrl.add_visualization(file_path)
        event.accept()
        # else:
        #     event.ignore()

    def dragEnterEvent(self, event):
        event.accept()

    def onClose(self):
        # print("Disable the interactor before closing to prevent it from trying to act on a already deleted items_vtk")
        self.qvtkWidget.close()


from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

if __name__ == "__main__":
    import sys

    app = Qt.QApplication(sys.argv)
    window = Ui_viewerWindow(None)
    app.aboutToQuit.connect(window.onClose)  # <-- connect the onClose event
    app.exec_()
