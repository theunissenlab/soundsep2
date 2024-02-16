"""A template for a new plugin

Any methods here can be deleted and will use the default implementations
found in soundsep.core.base_plugin
"""
from PyQt6 import QtWidgets as widgets
from soundsep.core.base_plugin import BasePlugin
from soundsep.widgets.block_view import AudioFileView



class BlockViewer(BasePlugin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.view = AudioFileView()
        self.view.show_columns(["Name", "Rate", "Ch", "Dur"])
        self.view.set_blocks(self.api.project.blocks)

    def plugin_panel_widget(self) -> list['PyQt6.QtWidgets.QWidget']:
        return [self.view]
