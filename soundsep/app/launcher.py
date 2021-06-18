from enum import Enum
from functools import partial

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5 import QtWidgets as widgets

from soundsep.app.app import SoundsepApp
from soundsep.app.main_window import SoundsepMainWindow
from soundsep.app.project_creator import ProjectCreator
from soundsep.app.project_loader import ProjectLoader
from soundsep.widgets.utils import not_implemented


class Splash(widgets.QWidget):

    class Option(Enum):
        CREATE_PROJECT = "Create Project"
        OPEN_PROJECT = "Open Project"
        DEBUG_PROJECT = "Debug Project"
        QUIT = "Quit"

    choiceMade = pyqtSignal(Option)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.connect_events()

    def init_ui(self):
        layout = widgets.QVBoxLayout(self)
        self.new_project_button = widgets.QPushButton("Create new project", self)
        self.open_button = widgets.QPushButton("Open Directory", self)
        self.quit_button = widgets.QPushButton("Quit", self)
        layout.addWidget(self.new_project_button)
        layout.addWidget(self.open_button)
        layout.addWidget(self.quit_button)
        self.setLayout(layout)
        self.setMinimumSize(800, 600)

    def connect_events(self):
        self.new_project_button.clicked.connect(partial(self.choiceMade.emit, Splash.Option.CREATE_PROJECT))
        self.open_button.clicked.connect(partial(self.choiceMade.emit, Splash.Option.OPEN_PROJECT))
        self.quit_button.clicked.connect(partial(self.choiceMade.emit, Splash.Option.QUIT))


class Launcher(QObject):
    """An thin layer that starts and stops apps independently of a project

    Starts with a basic splash screen and helps coordinate transitions
    between that, the main SoundSep window, and project utilities such as
    the project creator, debugger, or importer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_window = None

    def _center_on(self, w):
        rect = w.frameGeometry()
        geom = widgets.QDesktopWidget().availableGeometry()
        center_on = geom.center()
        center_on.setY(center_on.y() - geom.height() / 8)
        rect.moveCenter(center_on)
        w.move(rect.topLeft())

    def show(self):
        self.show_splash()

    def show_splash(self):
        if self.current_window:
            self.current_window.close()

        self.splash = Splash()
        self.splash.choiceMade.connect(self.on_splash_choice)
        self._center_on(self.splash)
        self.splash.show()

    def on_splash_choice(self, choice: 'Splash.Option'):
        if choice == Splash.Option.CREATE_PROJECT:
            self.current_window = ProjectCreator()
            self.current_window.createdProject.connect(self.open_project_directory)
            self.current_window.createProjectCanceled.connect(self.show_splash)
            self._center_on(self.current_window)
            self.current_window.show()
        elif choice == Splash.Option.OPEN_PROJECT:
            self.current_window = ProjectLoader()
            self.current_window.openProject.connect(self.open_project_directory)
            self.current_window.openProjectCanceled.connect(self.show_splash)
            self.current_window.show()
        elif choice == Splash.Option.DEBUG_PROJECT:
            not_implemented("Debugger not implemented")
            self.show_splash()
        elif choice == Splash.Option.QUIT:
            self.splash.close()

    def open_project_directory(self, project_dir: 'pathlib.Path'):
        if self.current_window:
            self.current_window.close()
            self.splash.close()

        # TODO: We should catch every possible error here we can think of...
        app = SoundsepApp(project_dir)
        self.current_window = SoundsepMainWindow(app.api)
        app.instantiate_plugins(gui=self.current_window)
        app.setup()
        app.api.projectLoaded.emit()

        app.api._closeProject.connect(self.show_splash)
        app.api._switchProject.connect(self.open_project_directory)

        self.current_window.show()
