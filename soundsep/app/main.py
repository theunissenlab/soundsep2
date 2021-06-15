import asyncio
import importlib
import logging
import pkgutil
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
    to the gui when
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
        self.gui_api = SoundsepGuiApi(self.gui)
        self.init_ui()
        self.connect_events()

        self.active_plugins = {}

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

    def remove_gui(self):
        if self.gui is not None:
            self.central_widget.removeWidget(self.gui)
            self.gui.deleteLater()

        self.gui = None
        self.gui_api._set_gui(None)
        self.central_widget.setCurrentWidget(self.loader_view)

    def replace_gui(self, new_gui: SoundsepGui):
        if self.gui is not None:
            self.central_widget.removeWidget(self.gui)
            self.gui.deleteLater()

        self.gui = new_gui
        self.gui_api._set_gui(new_gui)
        self.central_widget.addWidget(self.gui)
        self.central_widget.setCurrentWidget(self.gui)

        # Setup plugins
        self.active_plugins = {
            Plugin.__name__: Plugin(self.api, self.gui_api)
            for Plugin in self.load_plugins()
        }

        logger.info("Loaded Plugins {}".format(self.active_plugins))

        for name, plugin in self.active_plugins.items():
            plugin.setup_plugin_shortcuts()
            for w in plugin.plugin_toolbar_items():
                self.gui.toolbar.addWidget(w)

            panel = plugin.plugin_panel_widget()
            if panel:
                self.gui.ui.pluginPanelToolbox.addItem(
                    panel,
                    name
                )

            menu = plugin.plugin_menu(self.gui.ui.menuPlugins)
            # if menu:
            #     # self.gui.ui.menuPlugins.addSeparator()
            #     print("we here")
            #     self.gui.ui.menuPlugins.addMenu()
            #     print(menu)
                        # self.toolbar.addWidget(self.add_source_button)
                        # self.add_source_button.clicked.connect(self.on_add_source)


        # Tell all plugins to set up on the new gui?

    def show(self):
        self.main_window.show()

    # TODO: look for more plugins when a new project is loaded?
    def on_load_directory(self, directory: Path):
        gui = SoundsepGui(self.api)
        self.replace_gui(gui)

        self.api.load_project(directory)
    #
    # def _find_plugins(self):
    #     return glob.glob(os.path.join(self.paths.plugin_dir, "soundsep_*.py"))

    def load_plugins(self):
        """Load plugins from three possible locations

        (1) soundsep.plugins, and (2) self.paths.plugin_dir
        """
        import soundsep.plugins
        from soundsep.core.base_plugin import BasePlugin

        def iter_namespace(ns_pkg):
            return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

        # Search for builtin plugins in soundsep/plugins
        plugin_modules = [
            importlib.import_module(name)
            for finder, name, ispkg
            in iter_namespace(soundsep.plugins)
        ]
        #
        # # Search for plugins in the plugin_dir that are prefixed with "soundsep_"
        # for plugin_file in self._find_plugins():
        #     name = os.path.splitext(os.path.basename(plugin_file))[0]
        #     spec = importlib.util.spec_from_file_location("plugin.{}".format(name), plugin_file)
        #     plugin_module = importlib.import_module(importlib.util.module_from_spec(spec))
        #     spec.loader.exec_module(plugin_module)
        #     plugin_modules.append(plugin_module)

        plugins = []
        # for mod in plugin_modules:
        #     try:
        #         self.plugins.append(getattr(mod, "ExportPlugin"))
        #     except:
        #         warnings.warn("Did not find an ExportPlugin class in potential plugin file {}".format(mod))

        for Plugin in BasePlugin.registry:
            plugins.append(Plugin)

        return plugins



if __name__ == "__main__":
    run_app(SoundsepController(), None, MainWindow=SoundsepApp)
