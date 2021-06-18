from PyQt5 import QtWidgets as widgets


def not_implemented(msg: str):
    dialog = widgets.QMessageBox()
    dialog.setIcon(widgets.QMessageBox.Critical)
    dialog.setText("This function is not implemented yet!")
    dialog.setInformativeText("Mesage: {}".format(msg))
    dialog.setWindowTitle("Error")
    dialog.exec()
