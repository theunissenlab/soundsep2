import os
from typing import Callable, List, Optional

from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import pyqtSignal

from soundsep.core.models import AudioFile, Block
from soundsep.core.utils import hhmmss


class AudioFileView(widgets.QTreeWidget):
    """A view that can switch between a Block tree view and a linear Audio File view
    """
    column_map = {
            "Name": 0,
            "Id": 1,
            "Rate": 2,
            "Ch": 3,
            "Len": 4,
            "Dur": 5,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setHeaderLabels(["Name", "Id", "Rate", "Ch", "Len", "Dur"])
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, widgets.QHeaderView.Interactive)
        self.base_dir = None

    def show_columns(self, column_names):
        for k, v in self.column_map.items():
            self.setColumnHidden(v, k not in column_names)

    def format_path(self, path: 'Path'):
        """If base_dir is set, removes it from the path
        """
        if self.base_dir:
            return str(os.path.relpath(path, self.base_dir))
        else:
            return str(path)

    def audio_file_to_widget(
            self,
            audio_file: AudioFile,
            path_to_channel_keys_fn: Optional[Callable] = None
        ) -> widgets.QTreeWidgetItem:
        """Convert an AudioFile into a QTreeWidgetItem

        Arguments
        ---------
        audio_file : AudioFile
        path_to_channel_keys_fn : Optional[Callable]
            A function that maps a filename or path to a list of ids parsed from
            the filename. For example, a function that takes the path "/home/data/subject0_ch1.wav"
            and return a list ["ch1"]. The output of this column is filled into the column "Id".
            This would typically be used to fill in channel_keys
        """
        widget = widgets.QTreeWidgetItem()
        widget.setText(0, self.format_path(audio_file._path))
        if path_to_channel_keys_fn is not None:
            ids = path_to_channel_keys_fn(self.format_path(audio_file._path))
            widget.setText(1, ";".join([str(id_) for id_ in ids]))
        widget.setText(2, str(audio_file.sampling_rate))
        if audio_file.channels == 1:
            widget.setText(3, "0")
        elif audio_file.channels == 2:
            widget.setText(3, "0,1")
        else:
            widget.setText(3, "0-{}".format(audio_file.channels - 1))
        widget.setText(4, str(audio_file.frames))
        widget.setText(5, hhmmss(audio_file.frames / audio_file.sampling_rate, dec=0))
        widget.setToolTip(0, str(audio_file._path))

        return widget

    def block_to_widget(
            self,
            block: Block,
            path_to_block_keys_fn: Optional[Callable] = None,
            path_to_channel_keys_fn: Optional[Callable] = None
        ) -> widgets.QTreeWidgetItem:
        """Convert a Block into a QTreeWidgetItem

        Arguments
        ---------
        audio_file : AudioFile
        path_to_block_keys_fn : Optional[Callable]
            A function that maps a filename or path to a list of ids parsed from
            the filename. For example, a function that takes the path "/home/data/subject0_ch1.wav"
            and return a list ["subject0"]. The output of this column is filled into the column "Id".
            This would typically be used to fill in channel_keys
        path_to_channel_keys_fn : Optional[Callable]
            A function that maps a filename or path to a list of ids parsed from
            the filename. For example, a function that takes the path "/home/data/subject0_ch1.wav"
            and return a list ["ch1"]. The output of this column is filled into the column "Id"
            for each AudioFile in the block. 
        """
        widget = widgets.QTreeWidgetItem()
        widget.setText(0, str(block))
        if path_to_block_keys_fn is not None:
            ids = path_to_block_keys_fn(self.format_path(block._files[0]._path))
            widget.setText(1, ";".join([str(id_) for id_ in ids]))
        widget.setText(2, str(block.sampling_rate))
        if block.channels == 1:
            widget.setText(3, "0")
        elif block.channels == 2:
            widget.setText(3, "0,1")
        else:
            widget.setText(3, "0-{}".format(block.channels))
        widget.setText(4, str(block.frames))
        widget.setText(5, hhmmss(block.frames / block.sampling_rate, dec=0))

        # Create the audio file child widgets with updated channel columns
        ch = 0
        for audio_file in block._files:
            subwidget = self.audio_file_to_widget(audio_file, path_to_channel_keys_fn)
            if audio_file.channels == 1:
                subwidget.setText(3, "{}".format(ch))
            elif audio_file.channels == 2:
                subwidget.setText(3, "{},{}".format(ch, ch + 1))
            else:
                subwidget.setText(3, "{}-{}".format(ch, ch + audio_file.channels - 1))
            ch += audio_file.channels
            widget.addChild(subwidget)

        return widget

    def set_audio_files(
            self,
            audio_files: List[AudioFile],
            path_to_channel_keys_fn : Optional[Callable] = None,
            ):
        self.clear()
        self.insertTopLevelItems(0, [self.audio_file_to_widget(a, path_to_channel_keys_fn) for a in audio_files])
        for i in range(len(self.column_map)):
            self.resizeColumnToContents(i)
        self.expandAll()

    def set_blocks(
            self,
            blocks: List[Block],
            path_to_block_keys_fn : Optional[Callable] = None,
            path_to_channel_keys_fn : Optional[Callable] = None,
            ):
        t = 0.0
        widgets = []
        for b in blocks:
            widget = self.block_to_widget(b, path_to_block_keys_fn, path_to_channel_keys_fn)
            widget.setText(0, "{} at {}".format(b, hhmmss(t)))
            widgets.append(widget)
            t += b.frames / b.sampling_rate

        self.clear()
        self.insertTopLevelItems(0, widgets)
        for i in range(len(self.column_map)):
            self.resizeColumnToContents(i)
        self.expandAll()

    def set_base_dir(self, base):
        self.base_dir = base


class KeysSelector(widgets.QWidget):

    keysChanged = pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var_to_buttons = {}
        self._last_result = {}  # A dict with "block_keys" and "channel_keys"

    def set_variables(self, variables):
        for i in reversed(range(self.layout().count())):
            self.layout().itemAt(i).widget().deleteLater()

        self.var_to_buttons = {}
        for var in variables:
            var_widget = widgets.QGroupBox()
            var_layout = widgets.QHBoxLayout()
            var_layout.addWidget(widgets.QLabel(var))
            self.var_to_buttons[var] = {
                "block": widgets.QRadioButton("Block Key"),
                "channel": widgets.QRadioButton("Channel Key"),
                "neither": widgets.QRadioButton("Neither"),
            }
            var_layout.addWidget(self.var_to_buttons[var]["block"])
            var_layout.addWidget(self.var_to_buttons[var]["channel"])
            var_layout.addWidget(self.var_to_buttons[var]["neither"])
            self.var_to_buttons[var]["neither"].setChecked(True)

            for btn in self.var_to_buttons[var].values():
                btn.clicked.connect(self.check_for_changes)

            var_widget.setLayout(var_layout)
            self.layout().addWidget(var_widget)

    def get_keys(self):
        result = {"block_keys": [], "channel_keys": []}
        for var, btns in self.var_to_buttons.items():
            if btns["block"].isChecked():
                result["block_keys"].append(var)
            elif btns["channel"].isChecked():
                result["channel_keys"].append(var)
        return result

    def check_for_changes(self):
        result = self.get_keys()
        if result != self._last_result:
            self._last_result = result
            self.keysChanged.emit(result)
