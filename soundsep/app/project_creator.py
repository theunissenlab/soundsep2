from pathlib import Path

from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import pyqtSignal


class ProjectCreator(widgets.QWidget):
    """Project creation application
    """

    createdProject = pyqtSignal(Path)
    createProjectCanceled = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.connect_events()

    def init_ui(self):
        layout = widgets.QVBoxLayout()

        layout.addWidget(widgets.QLabel("1. Choose the directory to create project in"))
        layout.addWidget(widgets.QLabel("2. Click 'Initialize' to create project folders and configuration file"))
        layout.addWidget(widgets.QLabel("3. Copy WAV files into the audio/ directory of your project"))

        self.button = widgets.QPushButton("Go")
        layout.addWidget(self.button)
        self.setLayout(layout)

    def connect_events(self):
        self.button.clicked.connect(self.createProjectCanceled.emit)
