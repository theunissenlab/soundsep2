import logging
import os

import PyQt6.QtWidgets as widgets
import pandas as pd
from PyQt6 import QtGui

from soundsep.core.base_plugin import BasePlugin
from .export_window import ExportWizard


logger = logging.getLogger(__name__)


class ExportPlugin(BasePlugin):
    """Export segments, depends on segments plugin"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_actions()
        self.window = None

    def setup_actions(self):
        self.export_csv_action = QtGui.QAction("Export to &CSV")
        self.export_csv_action.triggered.connect(self.export_to_csv)

    def export_to_csv(self):
        """Exports to csv each Segments data

        source_name
        channel
        t_start
        t_stop

        original_file
        original_file_channel
        original_file_t_start
        original_file_t_stop
        """
        datastore = self.api.get_mut_datastore()
        self.window = ExportWizard(datastore, self.api)
        self.window.exportReady.connect(self.on_export_ready)
        self.window.exportCanceled.connect(self.on_export_canceled)
        self.window.show()

    def on_export_ready(self, df: 'pandas.DataFrame'):
        self.window.hide()
        save_target, _ = widgets.QFileDialog.getSaveFileName(
            None,
            "Export csv",
            str(self.api.paths.export_dir / "segment_export.csv"),
            "*.csv",
            options=options)

        if not save_target:
            logger.info("Aborted export attempt; no file chosen")
            self.window.show()
            return

        df.to_csv(save_target, index=False)
        logger.info("Exported csv to {}".format(save_target))
        self.window.close()

    def on_export_canceled(self):
        logger.debug("Canceled export")
        if self.window:
            self.window.close()

    def add_plugin_menu(self, menu_parent):
        menu = menu_parent.addMenu("&Export")
        menu.addAction(self.export_csv_action)
        return menu
