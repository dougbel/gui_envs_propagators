# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'propagators_loader.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1150, 808)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.vtk_widget = QVTKRenderWindowInteractor(self.centralwidget)
        self.vtk_widget.setGeometry(QtCore.QRect(270, 90, 871, 681))
        self.vtk_widget.setObjectName("vtk_widget")
        self.l_env = QtWidgets.QListWidget(self.centralwidget)
        self.l_env.setGeometry(QtCore.QRect(10, 90, 251, 681))
        self.l_env.setObjectName("l_env")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 70, 101, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(270, 70, 71, 16))
        self.label_2.setObjectName("label_2")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(190, 10, 661, 52))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.line_dataset = QtWidgets.QLineEdit(self.layoutWidget)
        self.line_dataset.setObjectName("line_dataset")
        self.gridLayout.addWidget(self.line_dataset, 0, 1, 1, 1)
        self.btn_dataset = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_dataset.setObjectName("btn_dataset")
        self.gridLayout.addWidget(self.btn_dataset, 0, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 1, 0, 1, 1)
        self.line_propagator = QtWidgets.QLineEdit(self.layoutWidget)
        self.line_propagator.setObjectName("line_propagator")
        self.gridLayout.addWidget(self.line_propagator, 1, 1, 1, 1)
        self.btn_propagator = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_propagator.setObjectName("btn_propagator")
        self.gridLayout.addWidget(self.btn_propagator, 1, 2, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1150, 19))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Environment:"))
        self.label_2.setText(_translate("MainWindow", "Interaction:"))
        self.label_3.setText(_translate("MainWindow", "Dataset"))
        self.btn_dataset.setText(_translate("MainWindow", "..."))
        self.label_4.setText(_translate("MainWindow", "Propagators"))
        self.btn_propagator.setText(_translate("MainWindow", "..."))

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

