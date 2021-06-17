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


class Splash(widgets.QWidget):
    """Splash screen displaying initial options"""

    loadDirectory = pyqtSignal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = widgets.QVBoxLayout(self)
        self.new_project_button = widgets.QPushButton("Create new project", self)
        self.open_button = widgets.QPushButton("Open Directory", self)
        self.recent_button = widgets.QPushButton("Open Last", self)
        self.import_button = widgets.QPushButton("Import audio files", self)
        self.quit_button = widgets.QPushButton("Quit", self)
        layout.addWidget(self.new_project_button)
        layout.addWidget(self.open_button)
        layout.addWidget(self.recent_button)
        layout.addWidget(self.import_button)
        layout.addWidget(self.quit_button)
        self.setLayout(layout)
        self.setMinimumSize(500, 300)

        self.new_project_button.clicked.connect(self.create_new_project_directory)
        self.open_button.clicked.connect(self.run_directory_loader)
        self.recent_button.clicked.connect(self.open_most_recent)
        self.quit_button.clicked.connect(self.close)

    def create_new_project_directory(self):
        """Basically the same as opening one except it does some checks for you"""
        options = widgets.QFileDialog.Options()
        selected_file = widgets.QFileDialog.getExistingDirectory(
            self,
            "Initialize a new project folder",
            ".",
            options=options
        )
        if selected_file:
            selected_file = Path(selected_file)
            if len(list(selected_file.glob("*"))):
                pass  # Warn user that they might be overwriting an existnig folder
            # TODO: see if there isa  project yaml in the folder already...
            # TODO: call the import wav file function here...
            self.loadDirectory.emit(selected_file)

    def run_directory_loader(self):
        """Dialog to read in a directory of wav files and intervals """

        # TODO: this was copied directly from gui.py. can we put this in one spot?
        options = widgets.QFileDialog.Options()
        selected_file = widgets.QFileDialog.getExistingDirectory(
            self,
            "Load directory",
            ".",
            options=options
        )

        if selected_file:
            self.loadDirectory.emit(Path(selected_file))

    def open_most_recent(self):
        # TODO: check if most recent exists (using qsettings?, otherwise grey out button)
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
        self.splash = Splash()
        self.splash.loadDirectory.connect(self.on_load_directory)
        self._center_on(self.splash)

    def _center_on(self, w):
        rect = w.frameGeometry()
        geom = widgets.QDesktopWidget().availableGeometry()
        center_on = geom.center()
        center_on.setY(center_on.y() - geom.height() / 8)
        rect.moveCenter(center_on)
        w.move(rect.topLeft())

    def connect_events(self):
        self.api.projectClosed.connect(self.on_project_closed)
        self.api.projectLoaded.connect(self.on_project_loaded)

    def on_project_closed(self):
        self.remove_gui()

    def on_project_loaded(self):
        self.replace_gui(SoundsepGui(self.api))
        # Have the app reinstantiate all plugins
        self.app.reload_plugins(self.api, self.gui)
        self.api.projectReady.emit()

    def remove_gui(self):
        if self.gui is not None:
            self.gui.close()
            self.gui = None

        self.splash.show()

    def replace_gui(self, new_gui: SoundsepGui):
        self.remove_gui()
        self.splash.hide()
        self.gui = new_gui
        self.gui.showMaximized()

    def show(self):
        self.splash.show()

    def on_load_directory(self, directory: Path):
        self.api.load_project(directory)


if __name__ == "__main__":
    run_app(SoundsepController(), None, MainWindow=SoundsepApp)
