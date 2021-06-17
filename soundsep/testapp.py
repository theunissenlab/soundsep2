from pathlib import Path
import asyncio

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5 import QtWidgets as widgets

from qasync import QEventLoop

from soundsep.app.main import run_app


class ProjectLoader(widgets.QWidget):

    loadDirectory = pyqtSignal(Path)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        button = widgets.QPushButton("Load", self)
        layout = widgets.QVBoxLayout()
        layout.addWidget(button)
        self.setLayout(layout)

        button.clicked.connect(self.on_load)

    def on_load(self):
        self.loadDirectory.emit(Path("asdF"))


class SoundsepGui(widgets.QWidget):


    def __init__(self, api, parent=None):
        super().__init__(parent=parent)
        self.api = api

        test = widgets.QLabel("Project", self)
        layout = widgets.QVBoxLayout()
        layout.addWidget(test)
        layout.addWidget(reset)
        self.setLayout(layout)


class CoreApp(QObject):
    pass


class CoreApiFunctions(QObject):

    loadProject = pyqtSignal(Path)

    def __init__(self, app: CoreApp):
        self._app = app

    def load_project(self, directory: Path):
        """Load a new project from a directory"""
        loadProject.emit(directory)


class SoundsepApplication(QObject):
    def __init__(self, core_app, gui_app=None):
        super().__init__()
        self.app = core_app
        self.gui = gui_app
        self.init_ui()

    def init_ui(self):
        self.main_window = widgets.QMainWindow()
        self.central_widget = widgets.QStackedWidget()
        self.main_window.setCentralWidget(self.central_widget)

        self.loader_view = ProjectLoader()
        self.loader_view.loadDirectory.connect(self.on_load_directory)

        self.central_widget.addWidget(self.loader_view)
        if self.gui is not None:
            self.central_widget.addWidget(self.gui)
            self.central_widget.setCurrentWidget(self.gui)
        else:
            self.central_widget.setCurrentWidget(self.loader_view)

    def replace_gui(self, new_gui: SoundsepGui):
        if self.gui is not None:
            self.central_widget.removeWidget(self.gui)
            self.gui.deleteLater()

        self.gui = new_gui
        self.central_widget.addWidget(self.gui)
        self.central_widget.setCurrentWidget(self.gui)

    def show(self):
        self.main_window.show()

    def on_load_directory(self, directory: Path):
        self.replace_gui(SoundsepGui(CoreApiFunctions(self.app)))


if __name__ == "__main__":
    import sys
    app = widgets.QApplication(sys.argv)

    core_app = None
    gui_app = None

    soundsep = SoundsepApplication(core_app, gui_app)
    soundsep.show()

    # mainWindow.destroyed.connect(self.on_destroy)
    # sys.exit(app.exec())
    # run_app(Application=TestWindow)

    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    '''
    if MainWindow is None:
        mainWindow = widgets.QMainWindow(*args, **kwargs)
    else:
        mainWindow = MainWindow(*args, **kwargs)

    mainWindow.show()
    '''

    with loop:
        loop.run_forever()

