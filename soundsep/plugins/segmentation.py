import PyQt5.QtWidgets as widgets
from PyQt5 import QtGui

import numpy as np

from soundsep.core.app import Workspace
from soundsep.core.base_plugin import BasePlugin
from soundsep.core.models import Source, ProjectIndex
from soundsep.core.segments import Segment


class _Panel(widgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        self.table = widgets.QTableWidget(0, 3)
        header = self.table.horizontalHeader()
        self.table.setHorizontalHeaderLabels([
            "SourceName",
            "Start",
            "Stop"
        ])
        header.setSectionResizeMode(0, widgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, widgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, widgets.QHeaderView.Stretch)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def set_data(self, segments):
        self.table.setRowCount(len(segments))
        for row, segment in enumerate(segments):
            self.table.setItem(row, 0, widgets.QTableWidgetItem(segment.source.name))
            self.table.setItem(row, 1, widgets.QTableWidgetItem(
                "{:.3f}s".format(segment.start / segment.project.sampling_rate)
            ))
            self.table.setItem(row, 2, widgets.QTableWidgetItem(
                "{:.3f}s".format(segment.start / segment.project.sampling_rate)
            ))


class Segmentation(BasePlugin):
    """Plugin for allowing users to 
    """

    def __init__(self, api, gui):
        super().__init__(api, gui)

        self._datastore = self.api.get_datastore()
        self._datastore["segmentation"] = {"segments": []}

        self.panel = _Panel()
        self.createSegmentButton = widgets.QPushButton("+Segment")
        self.createSegmentButton.clicked.connect(self.on_create_segment)

        self.setup_shortcuts()

        self.api.xrangeChanged.connect(self.on_xrange_changed)
        self.api.selectionChanged.connect(self.on_selection_changed)

    def setup_shortcuts(self):
        self.create_shortcut = widgets.QShortcut(QtGui.QKeySequence("F"), self.gui)
        self.create_shortcut.activated.connect(self.on_create_segment)

    def on_xrange_changed(self, i0, i1):
        """Keeps the table pointed at a visible segment"""
        start_times = [int(s.start) for s in self._datastore["segmentation"]["segments"]]
        segment_idx = np.searchsorted(start_times, i0)
        index = self.panel.table.model().index(segment_idx, 0)
        self.panel.table.scrollTo(index)

    def on_selection_changed(self):
        """Keeps the table pointed at a visible segment"""
        selection = self.api.get_active_selection()
        if selection:
            xbounds, ybounds, source = selection
            start_times = [int(s.start) for s in self._datastore["segmentation"]["segments"]]
            segment_idx = np.searchsorted(start_times, int(xbounds[0]))
            index = self.panel.table.model().index(segment_idx, 0)
            self.panel.table.scrollTo(index)

    def on_create_segment(self):
        selection = self.api.get_active_selection()
        if selection:
            xbounds, ybounds, source = selection
            self.create_segment(xbounds[0], xbounds[1], source)

    def create_segment(self, i0: ProjectIndex, i1: ProjectIndex, source: Source):
        new_segment = Segment(i0, i1, source)
        self._datastore["segmentation"]["segments"].append(new_segment)
        self._datastore["segmentation"]["segments"].sort()
        self.panel.set_data(self._datastore["segmentation"]["segments"])
        self.gui.show_status("Created segment on {}".format(source.name))

    def plugin_toolbar_items(self):
        return [self.createSegmentButton]

    def plugin_menu(self):
        return None

    def plugin_panel_widget(self):
        return self.panel

        # First, set up a table in the datastore

ExportPlugin = Segmentation
