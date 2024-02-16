from typing import List

from PyQt6.QtCore import QObject


class BasePlugin(QObject):

    registry = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry.append(cls)

    def __init__(self, api, gui):
        super().__init__()
        self.api = api
        self.gui = gui

    def plugin_toolbar_items(self) -> List['PyQt6.QtWidgets.QWidget']:
        """Reimplement this

        Returns
        -------
        widgets : list[QWidget]
            a list of widgets to place in the application toolbar
        """
        return []

    def add_plugin_menu(self, menu: 'PyQt6.QtWidgets.QMenu'):
        """Reimplement this

        Example
        -------:
        ..code-block:: python
        def add_plugin_menu(self, menu):
            submenu = menu.addMenu("&PluginName")
            submenu.addAction(self.foo)

        Arguments
        ---------
        menu : QMenu
            The &Plugin menu in the main window menu bar. This function can
            add actions to the menu
        """
        return None

    def plugin_panel_widget(self) -> list['PyQt6.QtWidgets.QWidget']:
        """Reimplement this

        Returns
        -------
        widgets : QWidget
            the main panel for this plugin to show in the plugin panel
        """
        return []

    def setup_plugin_shortcuts(self):
        return None

    def needs_saving(self) -> bool:
        """Reimplement this

        Returns
        -------
        needs_saving : bool
            return True from this function if the plugin should ask the user for
            confirmation before closing the program to save.
        """
        return False

    def save(self):
        """Reimplement this

        This function is called when the Save action is taken in the main program
        """
        pass
