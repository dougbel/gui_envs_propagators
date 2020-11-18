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
        MainWindow.setEnabled(True)
        MainWindow.resize(1477, 922)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.vtk_widget = QVTKRenderWindowInteractor(self.centralwidget)
        self.vtk_widget.setGeometry(QtCore.QRect(220, 180, 871, 701))
        self.vtk_widget.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.vtk_widget.setObjectName("vtk_widget")
        self.l_env = QtWidgets.QListWidget(self.centralwidget)
        self.l_env.setGeometry(QtCore.QRect(10, 200, 201, 431))
        self.l_env.setObjectName("l_env")
        self.lbl_environment = QtWidgets.QLabel(self.centralwidget)
        self.lbl_environment.setGeometry(QtCore.QRect(10, 180, 101, 16))
        self.lbl_environment.setTextFormat(QtCore.Qt.RichText)
        self.lbl_environment.setObjectName("lbl_environment")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 0, 881, 154))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.line_dataset = QtWidgets.QLineEdit(self.layoutWidget)
        self.line_dataset.setObjectName("line_dataset")
        self.gridLayout.addWidget(self.line_dataset, 1, 0, 1, 1)
        self.line_configurations = QtWidgets.QLineEdit(self.layoutWidget)
        self.line_configurations.setText("")
        self.line_configurations.setObjectName("line_configurations")
        self.gridLayout.addWidget(self.line_configurations, 3, 0, 1, 1)
        self.line_results = QtWidgets.QLineEdit(self.layoutWidget)
        self.line_results.setObjectName("line_results")
        self.gridLayout.addWidget(self.line_results, 5, 0, 1, 1)
        self.btn_dataset = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_dataset.setObjectName("btn_dataset")
        self.gridLayout.addWidget(self.btn_dataset, 1, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        self.label_4.setTextFormat(QtCore.Qt.RichText)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 4, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        self.label_3.setTextFormat(QtCore.Qt.RichText)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.btn_results = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_results.setObjectName("btn_results")
        self.gridLayout.addWidget(self.btn_results, 5, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.layoutWidget)
        self.label_5.setTextFormat(QtCore.Qt.RichText)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 2, 0, 1, 1)
        self.btn_configurations = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_configurations.setObjectName("btn_configurations")
        self.gridLayout.addWidget(self.btn_configurations, 3, 1, 1, 1)
        self.lbl_interactions = QtWidgets.QLabel(self.centralwidget)
        self.lbl_interactions.setGeometry(QtCore.QRect(10, 650, 101, 16))
        self.lbl_interactions.setTextFormat(QtCore.Qt.RichText)
        self.lbl_interactions.setObjectName("lbl_interactions")
        self.l_interactions = QtWidgets.QListWidget(self.centralwidget)
        self.l_interactions.setGeometry(QtCore.QRect(10, 670, 201, 211))
        self.l_interactions.setObjectName("l_interactions")
        self.tree_train = QtWidgets.QTreeView(self.centralwidget)
        self.tree_train.setGeometry(QtCore.QRect(1110, 300, 351, 461))
        self.tree_train.setAlternatingRowColors(True)
        self.tree_train.setAutoExpandDelay(-1)
        self.tree_train.setIndentation(10)
        self.tree_train.setItemsExpandable(True)
        self.tree_train.setObjectName("tree_train")
        self.tree_propagation = QtWidgets.QTreeView(self.centralwidget)
        self.tree_propagation.setGeometry(QtCore.QRect(1110, 770, 351, 111))
        self.tree_propagation.setAlternatingRowColors(True)
        self.tree_propagation.setAutoExpandDelay(-1)
        self.tree_propagation.setIndentation(10)
        self.tree_propagation.setItemsExpandable(True)
        self.tree_propagation.setObjectName("tree_propagation")
        self.chk_on_gray_env = QtWidgets.QCheckBox(self.centralwidget)
        self.chk_on_gray_env.setGeometry(QtCore.QRect(920, 10, 181, 20))
        self.chk_on_gray_env.setObjectName("chk_on_gray_env")
        self.vtk_interaction = QVTKRenderWindowInteractor(self.centralwidget)
        self.vtk_interaction.setEnabled(True)
        self.vtk_interaction.setGeometry(QtCore.QRect(1110, 10, 351, 281))
        self.vtk_interaction.setObjectName("vtk_interaction")
        self.btn_add_sample = QtWidgets.QPushButton(self.centralwidget)
        self.btn_add_sample.setEnabled(False)
        self.btn_add_sample.setGeometry(QtCore.QRect(990, 820, 91, 22))
        self.btn_add_sample.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.btn_add_sample.setObjectName("btn_add_sample")
        self.btn_show_samples = QtWidgets.QPushButton(self.centralwidget)
        self.btn_show_samples.setEnabled(False)
        self.btn_show_samples.setGeometry(QtCore.QRect(990, 850, 91, 23))
        self.btn_show_samples.setObjectName("btn_show_samples")
        self.chk_on_tested_points = QtWidgets.QCheckBox(self.centralwidget)
        self.chk_on_tested_points.setGeometry(QtCore.QRect(920, 40, 181, 20))
        self.chk_on_tested_points.setIconSize(QtCore.QSize(16, 27))
        self.chk_on_tested_points.setChecked(True)
        self.chk_on_tested_points.setAutoRepeatDelay(300)
        self.chk_on_tested_points.setObjectName("chk_on_tested_points")
        self.lbl_results = QtWidgets.QLabel(self.centralwidget)
        self.lbl_results.setGeometry(QtCore.QRect(870, 180, 224, 224))
        self.lbl_results.setAutoFillBackground(False)
        self.lbl_results.setStyleSheet("")
        self.lbl_results.setText("")
        self.lbl_results.setObjectName("lbl_results")
        self.btn_play = QtWidgets.QPushButton(self.centralwidget)
        self.btn_play.setEnabled(False)
        self.btn_play.setGeometry(QtCore.QRect(1030, 150, 31, 23))
        self.btn_play.setObjectName("btn_play")
        self.btn_stop = QtWidgets.QPushButton(self.centralwidget)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setGeometry(QtCore.QRect(1000, 150, 31, 23))
        self.btn_stop.setObjectName("btn_stop")
        self.btn_pause = QtWidgets.QPushButton(self.centralwidget)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setGeometry(QtCore.QRect(1060, 150, 31, 23))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.btn_pause.setFont(font)
        self.btn_pause.setObjectName("btn_pause")
        self.vtk_interaction.raise_()
        self.vtk_widget.raise_()
        self.l_env.raise_()
        self.lbl_environment.raise_()
        self.layoutWidget.raise_()
        self.lbl_interactions.raise_()
        self.l_interactions.raise_()
        self.tree_train.raise_()
        self.tree_propagation.raise_()
        self.btn_show_samples.raise_()
        self.btn_add_sample.raise_()
        self.chk_on_tested_points.raise_()
        self.chk_on_gray_env.raise_()
        self.btn_play.raise_()
        self.btn_stop.raise_()
        self.btn_pause.raise_()
        self.lbl_results.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1477, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Vizualizing propagation"))
        self.lbl_environment.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Environment:</span></p></body></html>"))
        self.btn_dataset.setText(_translate("MainWindow", "..."))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Propagation results</span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Dataset</span></p></body></html>"))
        self.btn_results.setText(_translate("MainWindow", "..."))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Propagation configurations</span></p></body></html>"))
        self.btn_configurations.setText(_translate("MainWindow", "..."))
        self.lbl_interactions.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Interactions:</span></p></body></html>"))
        self.chk_on_gray_env.setText(_translate("MainWindow", "environment in gray color"))
        self.btn_add_sample.setText(_translate("MainWindow", "Add sample"))
        self.btn_show_samples.setText(_translate("MainWindow", "Show samples"))
        self.chk_on_tested_points.setText(_translate("MainWindow", "tested points"))
        self.btn_play.setText(_translate("MainWindow", "►"))
        self.btn_stop.setText(_translate("MainWindow", "■"))
        self.btn_pause.setText(_translate("MainWindow", "||"))

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

