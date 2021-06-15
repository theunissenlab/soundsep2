from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class BasePlugin(QObject):

    registry = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry.append(cls)

    def __init__(self, api, gui):
        super().__init__()
        self.api = api
        self.gui = gui

    def plugin_toolbar_items(self):
        return []

    def add_plugin_menu(self, menu):
        return None

    def plugin_panel_widget(self):
        return None

    def setup_plugin_shortcuts(self):
        return None
