import vtk
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from vtkplotter import Plotter, Cone

from gui import Ui_MainWindow


def click_set_dir_dataset(window, ui):
    fileName = str(QFileDialog.getExistingDirectory(window,  "Select Directory"))
    if fileName:
        ui.line_dataset.setText(fileName)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.vtk_widget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    # BUTTON SIGNALS
    ui.btn_dataset.clicked.connect(lambda: click_set_dir_dataset(MainWindow, ui))


    vp = Plotter(qtWidget=ui.vtk_widget)
    vp += Cone()
    vp.show()

    MainWindow.show()
    sys.exit(app.exec_())

