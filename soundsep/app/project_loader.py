import os
from pathlib import Path

from PyQt6.QtCore import QObject, QSettings, pyqtSignal
from PyQt6 import QtWidgets as widgets

from soundsep.config.paths import ProjectPathFinder
from soundsep.settings import QSETTINGS_APP, QSETTINGS_ORG, SETTINGS_VARIABLES


class ProjectLoader(QObject):
    """A loader for already configured project folders
    """

    openProject = pyqtSignal(Path)
    openProjectCanceled = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qsettings = QSettings(
            QSETTINGS_ORG,
            QSETTINGS_APP,
        )

    def show(self):
        # options = widgets.QFileDialog.options(self)
        reopen_path = "."
        if self.qsettings.contains(SETTINGS_VARIABLES["REOPEN_PROJECT_PATH"]):
            path = str(self.qsettings.value(SETTINGS_VARIABLES["REOPEN_PROJECT_PATH"]))
            if os.path.exists(path):
                reopen_path = path

        # project_dir = widgets.QFileDialog.getExistingDirectory(
        #    None, "Load project", reopen_path, options=options)

        project_dir = widgets.QFileDialog.getExistingDirectory(
            None, "Load project", reopen_path)
        if not project_dir:
            self.openProjectCanceled.emit()
            return

        project_dir = Path(project_dir)
        # TODO: use config file to validate if the project is openable
        # alternate TODO: open a project by config file rather than folder
        config_file = ProjectPathFinder(project_dir).config
        self.openProject.emit(Path(project_dir))

    def close(self):
        pass
