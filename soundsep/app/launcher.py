import logging
from enum import Enum
from functools import partial

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5 import QtWidgets as widgets

from soundsep.app.app import SoundsepApp
from soundsep.app.exceptions import BadConfigFormat, ConfigDoesNotExist
from soundsep.app.main_window import SoundsepMainWindow
from soundsep.app.project_creator import ProjectCreator
from soundsep.app.project_loader import ProjectLoader
from soundsep.ui.splash import Ui_SplashPage
from soundsep.widgets.utils import not_implemented


logger = logging.getLogger(__name__)


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
        self.ui = Ui_SplashPage()
        self.ui.setupUi(self)

        # TODO: maybe this is too much
        # self.setWindowFlags(Qt.FramelessWindowHint)
        self.setFixedSize(self.sizeHint())

        self.ui.debugProjectButton.setVisible(False)

    def connect_events(self):
        self.ui.createProjectButton.clicked.connect(partial(self.choiceMade.emit, Splash.Option.CREATE_PROJECT))
        self.ui.openProjectButton.clicked.connect(partial(self.choiceMade.emit, Splash.Option.OPEN_PROJECT))
        self.ui.debugProjectButton.clicked.connect(partial(self.choiceMade.emit, Splash.Option.DEBUG_PROJECT))
        self.ui.exitButton.clicked.connect(partial(self.choiceMade.emit, Splash.Option.QUIT))


class Launcher(QObject):
    """An thin layer that starts and stops apps independently of a project

    Starts with a basic splash screen and helps coordinate transitions
    between that, the main SoundSep window, and project utilities such as
    the project creator, debugger, or importer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_window = None
        self.splash = None

    def _center_on(self, w):
        rect = w.frameGeometry()
        geom = widgets.QDesktopWidget().availableGeometry()
        center_on = geom.center()
        center_on.setY(center_on.y())
        rect.moveCenter(center_on)
        w.move(rect.topLeft())

    def show(self):
        self.show_splash()

    def show_splash(self):
        if self.current_window:
            self.current_window.close()
        if self.splash:
            self.splash.close()

        self.splash = Splash()
        self.splash.choiceMade.connect(self.on_splash_choice)
        self._center_on(self.splash)
        self.splash.show()

    def on_splash_choice(self, choice: 'Splash.Option'):
        if choice == Splash.Option.CREATE_PROJECT:
            self.splash.close()
            self.current_window = ProjectCreator()
            self.current_window.createConfigCanceled.connect(self.show_splash)
            self.current_window.openProject.connect(self.open_project_directory)
            self._center_on(self.current_window)
            self.current_window.show()
        elif choice == Splash.Option.OPEN_PROJECT:
            self.current_window = ProjectLoader()
            self.current_window.openProject.connect(self.open_project_directory)
            self.current_window.openProjectCanceled.connect(self.show_splash)
            self.current_window.show()
        elif choice == Splash.Option.DEBUG_PROJECT:
            self.splash.close()
            self.current_window = ProjectCreator()
            self.current_window.createConfigCanceled.connect(self.show_splash)
            self.current_window.openProject.connect(self.open_project_directory)
            self._center_on(self.current_window)
            self.current_window.show()
        elif choice == Splash.Option.QUIT:
            self.splash.close()

    def open_project_directory(self, project_dir: 'pathlib.Path'):
        if self.current_window:
            self.current_window.close()
            self.splash.close()

        # TODO: We should catch every possible error here we can think of...
        try:
            app = SoundsepApp(project_dir)
        except ConfigDoesNotExist:
            self.show_splash()
            widgets.QMessageBox.critical(
                self.splash,
                "Config not found",
                "soundsep.yaml in {} not found.".format(project_dir),
            )
            return
        except BadConfigFormat:
            self.show_splash()
            widgets.QMessageBox.critical(
                self.splash,
                "Config not readable",
                "Config file {} could not be read. Check syntax.".format(project_dir),
            )
            self.show_splash()
            return
        except Exception:
            self.show_splash()
            logger.exception("Error loading project")
            widgets.QMessageBox.critical(
                self.splash,
                "Error",
                "An unexpected error occured loading {}. See logs.".format(project_dir),
            )
            self.show_splash()
            return

        self.current_window = SoundsepMainWindow(app.api)
        app.instantiate_plugins(gui=self.current_window)
        app.setup()
        app.api.projectLoaded.emit()

        app.api._closeProject.connect(self.show_splash)
        app.api._switchProject.connect(self.open_project_directory)

        # By default lets open to 3/4 screen size and center
        screen = widgets.QApplication.primaryScreen()
        size = screen.size()
        rect = screen.availableGeometry()
        self.current_window.resize(rect.width() * 0.75, rect.height() * 0.75)
        self._center_on(self.current_window)

        self.current_window.show()
