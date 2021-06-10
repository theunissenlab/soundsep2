from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from soundsep.core.app import Workspace


class BasePlugin(QObject):
    def __init__(self, api, gui):
        super().__init__()
        self.api = api
        self.gui = gui

    def plugin_toolbar_items(self):
        return []

    def plugin_menu(self):
        return None

    def plugin_panel_widget(self):
        return None

    def setup_plugin_shortcuts(self):
        return None

