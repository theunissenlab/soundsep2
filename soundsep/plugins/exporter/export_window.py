import functools
import logging
from collections import namedtuple

import PyQt5.QtWidgets as widgets
import pandas as pd
from PyQt5.QtCore import Qt, pyqtSignal


logger = logging.getLogger(__name__)


ExportSetting = namedtuple("ExportSetting", ["config_name", "readable_name", "help"])


SETTINGS = (
    ExportSetting("source.name", "Name", "source name"),
    ExportSetting("source.channel", "Channel",
        "the source channel (may not correspond to channel in file, see file.channel)"),
    ExportSetting("project.start_index", "Project Start (int, samples)",
        "the start index of segments relative to the entire project"),
    ExportSetting("project.stop_index", "Project Stop (int, samples)",
        "the stop index of segments relative to the entire project"),
    ExportSetting("project.t_start", "Project Start (float, seconds)",
        "the start time of segments relative to the project"),
    ExportSetting("project.t_stop", "Project Start (float, seconds)",
        "the stop time of segments relative to the project"),
    ExportSetting("file.name", "Filename",
        "the original file the segment is from"),
    ExportSetting("file.channel", "File channel",
        "the original channel the segment is from"),
    ExportSetting("file.start_index", "File Start (int, samples)",
        "the start index of segments relative to their original file"),
    ExportSetting("file.stop_index", "File Start (int, samples)",
        "the stop index of segments relative to their original file"),
    ExportSetting("file.t_start", "File Start (float, seconds)",
        "the start time of segments relative to their original file"),
    ExportSetting("file.t_stop", "File Stop (float, seconds)",
        "the stop time of segments relative to their original file"),
    ExportSetting("tags", "Tags (string)",
        "comma separated tag string values"),
)


def segment_to_dict(segment):
    source = segment.source
    project = source.project
    block_start = project.to_block_index(segment.start)
    block_stop = project.to_block_index(segment.stop)
    block = block_start.block
    original_file, original_channel = block.get_channel_info(source.channel)

    return {
        "source.name": source.name,
        "source.channel": source.channel,
        "project.start_index": int(segment.start),
        "project.stop_index": int(segment.stop),
        "project.t_start": segment.start.to_timestamp(),
        "project.t_stop": segment.stop.to_timestamp(),
        "file.name": original_file,
        "file.channel": original_channel,
        "file.start_index": int(block_start),
        "file.stop_index": int(block_stop),
        "file.t_start": block_start.to_file_timestamp(),
        "file.t_stop": block_stop.to_file_timestamp(),
        "tags": ",".join(segment.data.get("tags", [])),
    }


class ExportWizard(widgets.QWidget):
    """Window for customizing an export
    """

    exportReady = pyqtSignal(object)
    exportCanceled = pyqtSignal()

    def __init__(self, datastore):
        super().__init__()
        logger.info("Starting export wizard")
        self.datastore = datastore
        self.init_ui()
        self.connect_events()

    def init_ui(self):
        layout = widgets.QVBoxLayout(self)
        
        form_layout = widgets.QGridLayout()
        form_layout.addWidget(widgets.QLabel("column"), 0, 0)
        form_layout.addWidget(widgets.QLabel("include?"), 0, 1)
        form_layout.addWidget(widgets.QLabel("as name"), 0, 2)
        self.fields = {}
        self.mapped_names = {}
        for i, setting in enumerate(SETTINGS):
            self.fields[setting.config_name] = widgets.QCheckBox()
            self.fields[setting.config_name].setCheckState(Qt.Checked)
            self.mapped_names[setting.config_name] = widgets.QLineEdit()
            self.mapped_names[setting.config_name].setPlaceholderText(setting.config_name)
            form_layout.addWidget(widgets.QLabel(setting.readable_name), i + 1, 0)
            form_layout.addWidget(self.fields[setting.config_name], i + 1, 1)
            form_layout.addWidget(self.mapped_names[setting.config_name], i + 1, 2)

        buttons_layout = widgets.QHBoxLayout()
        self.submit_button = widgets.QPushButton("&Export")
        self.cancel_button = widgets.QPushButton("&Cancel")
        buttons_layout.addWidget(self.submit_button)
        buttons_layout.addWidget(self.cancel_button)

        layout.addLayout(form_layout)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)

    def connect_events(self):
        self.submit_button.clicked.connect(self.on_submit)
        self.cancel_button.clicked.connect(self.on_cancel)

    def get_form_keys(self):
        return set([k for k, checkbox in self.fields.items() if checkbox.checkState() == Qt.Checked])

    def get_name_mappings(self):
        return {
            k: lineedit.text() or lineedit.placeholderText()
            for k, lineedit in self.mapped_names.items()
        }

    def on_submit(self):
        name_map = self.get_name_mappings()
        include_keys = self.get_form_keys()

        logger.info("Exporting csv with columns {}".format(list(include_keys)))

        segment_dicts = []
        for segment in self.datastore["segments"]:
            segment_row = {
                name_map.get(k, k): v for k, v in segment_to_dict(segment).items()
                if k in include_keys
            }
            segment_dicts.append(segment_row)

        df = pd.DataFrame(segment_dicts)
        self.exportReady.emit(df)

    def on_cancel(self):
        self.exportCanceled.emit()

