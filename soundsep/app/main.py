import asyncio
import logging
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5 import QtWidgets as widgets
from qasync import QEventLoop

from soundsep.api import SoundsepControllerApi, SoundsepGuiApi
from soundsep.app.gui import SoundsepGui
from soundsep.app.app import SoundsepController


logger = logging.getLogger(__name__)


def run_app(*args, MainWindow=None, **kwargs):
    """Run an app using asyncio event loop
    """
    import sys

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)

    rootLogger = logging.getLogger()
    rootLogger.addHandler(console)
    rootLogger.setLevel(level=logging.DEBUG)

    app = widgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    if MainWindow is None:
        mainWindow = widgets.QMainWindow(*args, **kwargs)
    else:
        mainWindow = MainWindow(*args, **kwargs)

    mainWindow.show()

    with loop:
        loop.run_forever()


# TODO: move this into gui components?
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
        self.loadDirectory.emit(Path("data"))



class SoundsepApp(QObject):
    """An object that links together the core functions of the app to the gui

    Manages a basic splash screen to load a project, and switches
    to the gui when a new project is loaded.

    Attributes
    ----------
    app : soundsep.app.controller.SoundsepController
    gui : Optional[soundsep.app.gui.SoundsepGui]
        The SoundsepGui object representing the main display area. When it is None
        the basic landing page will be displayed instead.
    """
    def __init__(self, core_app: SoundsepController, gui_app=None):
        super().__init__()
        self.app = core_app
        self.api = SoundsepControllerApi(self.app)
        self.gui = gui_app
        self.init_ui()
        self.connect_events()

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

    def connect_events(self):
        self.api.projectClosed.connect(self.on_project_closed)
        self.api.projectLoaded.connect(self.on_project_loaded)

    def on_project_closed(self):
        self.remove_gui()

    def on_project_loaded(self):
        self.replace_gui(SoundsepGui(self.api))

        # Have the app reinstantiate all plugins with the
        # api object (so all plugins have access to the SoundsepControllerApi)
        # and access to modifying the GUI elements
        self.app.reload_plugins(self.api, self.gui)

    def remove_gui(self):
        if self.gui is not None:
            self.central_widget.removeWidget(self.gui)
            self.gui.deleteLater()

        self.gui = None
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
        gui = SoundsepGui(self.api)
        self.replace_gui(gui)
        self.api.load_project(directory)


if __name__ == "__main__":
    run_app(SoundsepController(), None, MainWindow=SoundsepApp)
