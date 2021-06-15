import logging

import PyQt5.QtWidgets as widgets
import pandas as pd

from soundsep.core.base_plugin import BasePlugin


logger = logging.getLogger(__name__)


class ExportPlugin(BasePlugin):
    """Export segments, depends on segments plugin"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setup_actions()

    def setup_actions(self):
        self.export_csv_action = widgets.QAction("Export to &CSV")
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
        if "segments" not in datastore:
            logger.error("Attempted to but segments data not found. Is the SegmentPlugin loaded?")
            msg = widgets.QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText(e)
            msg.setWindowTitle("Error")
            return

        project = self.api.get_current_project()

        segment_dicts = []
        for segment in datastore["segments"]:
            block_start = project.to_block_index(segment.start)
            block_stop = project.to_block_index(segment.stop)
            original_file, original_channel = block_start.block.get_channel_info(segment.source.channel)

            segment_dicts.append({
                "source_name": segment.source.name,
                "channel": segment.source.channel,
                "t_start": segment.start.to_timestamp(),
                "t_stop": segment.stop.to_timestamp(),

                "original_file": original_file,
                "original_file_channel": original_channel,
                "original_file_t_start": block_start.to_file_timestamp(),
                "original_file_t_stop": block_stop.to_file_timestamp(),
            })

        df = pd.DataFrame(segment_dicts)
        
        # TODO: where to save this?
        logger.info("Saving exported segment data to {}".format("TBD"))
        print(df)

    def add_plugin_menu(self, menu_parent):
        menu = menu_parent.addMenu("&Export")
        menu.addAction(self.export_csv_action)
        return menu
