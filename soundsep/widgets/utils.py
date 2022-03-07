from PyQt6 import QtWidgets as widgets
from PyQt6.QtCore import Qt, pyqtSignal


class QClickableLineEdit(widgets.QLineEdit):
    clicked = pyqtSignal() # signal when the text entry is left clicked

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.clicked.emit()
        else: super().mousePressEvent(event)


def not_implemented(msg: str):
    dialog = widgets.QMessageBox()
    dialog.setIcon(widgets.QMessageBox.Critical)
    dialog.setText("This function is not implemented yet!")
    dialog.setInformativeText("Mesage: {}".format(msg))
    dialog.setWindowTitle("Error")
    dialog.exec()
