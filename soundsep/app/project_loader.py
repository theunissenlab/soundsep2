from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5 import QtWidgets as widgets

from soundsep.config.paths import ProjectPathFinder


class ProjectLoader(QObject):
    """A loader for already configured project folders
    """

    openProject = pyqtSignal(Path)
    openProjectCanceled = pyqtSignal()

    def show(self):
        options = widgets.QFileDialog.Options()
        project_dir = widgets.QFileDialog.getExistingDirectory(
            None, "Load project", ".", options=options)

        if not project_dir:
            self.openProjectCanceled.emit()
            return

        project_dir = Path(project_dir)
        config_file = ProjectPathFinder(project_dir).config
        self.openProject.emit(Path(project_dir))

    def close(self):
        pass
