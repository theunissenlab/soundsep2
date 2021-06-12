import PyQt5.QtWidgets as widgets
from PyQt5 import QtGui

import pyqtgraph as pg

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
                "{:.3f}s".format(segment.stop / segment.project.sampling_rate)
            ))


def index_conversion(idxA, scaleA, scaleB):
    return idxA * (scaleB / scaleA)


class SegmentVisualizer(QtGui.QGraphicsRectItem):
    def __init__(self, x0, y0, width, height, segment, spectrogram, gui):
        super().__init__(
            x0, y0, width, height, parent=spectrogram.image
        )
        self.gui = gui
        self.x0 = x0
        self.segment = segment
        self.x1 = x0 + width
        self.spectrogram = spectrogram
        self.setPen(pg.mkPen(None))
        color = "#8fcfd1"
        self.setBrush(pg.mkBrush(color))
        self.setAcceptHoverEvents(True)

        self.lines = []

    def hoverEnterEvent(self, event):
        """Draw vertical lines as boundaries"""
        self.lines.append(pg.InfiniteLine(pos=self.x0))
        self.lines.append(pg.InfiniteLine(pos=self.x1))
        for line in self.lines:
            try:
                self.spectrogram.addItem(line)
            except:
                pass
        self.gui.show_status("{} to {}".format(self.segment.start, self.segment.stop), 10000)

    def hoverLeaveEvent(self, event):
        """Destroy lines"""
        for line in self.lines:
            try:
                self.spectrogram.removeItem(line)
            except:
                pass
        self.lines = []


class Segmentation(BasePlugin):
    """Plugin for allowing users to 
    """

    NAME = "Segmentation"

    def __init__(self, api, gui):
        super().__init__(api, gui)

        self.panel = _Panel()
        self._annotations = []
        self.createSegmentButton = widgets.QPushButton("+Segment")
        self.createSegmentButton.clicked.connect(self.on_create_segment_activated)

        self.setup_shortcuts()

        self.api.xrangeChanged.connect(self.on_xrange_changed)
        self.api.selectionChanged.connect(self.on_selection_changed)
        #  self.api.projectChanged.connect(self.on_project_changed)

    @property
    def _datastore(self):
        return self.api.get_datastore()

    @property
    def _segmentation_datastore(self):
        datastore = self._datastore
        if "segmentation" in datastore:
            return datastore["segmentation"]
        else:
            datastore["segmentation"] = {"segments": []}
            return datastore["segmentation"]

    def setup_shortcuts(self):
        self.create_shortcut = widgets.QShortcut(QtGui.QKeySequence("F"), self.gui)
        self.create_shortcut.activated.connect(self.on_create_segment_activated)

    def on_xrange_changed(self, i0, i1):
        self.refresh()

    def refresh(self):
        """Keeps the table pointed at a visible segment"""
        i0, i1 = self.api.get_xrange()
        # Also highlight all points visible
        for parent, annotation in self._annotations:
            parent.removeItem(annotation)
        self._annotations = []

        start_times = [int(s.start) for s in self._segmentation_datastore["segments"]]
        first_segment_idx = np.searchsorted(start_times, i0)
        last_segment_idx = np.searchsorted(start_times, i1)

        index = self.panel.table.model().index(first_segment_idx, 0)
        self.panel.table.scrollTo(index)

        for segment in self._segmentation_datastore["segments"][first_segment_idx:last_segment_idx]:
            # rect = SegmentIcon(segment.start, segment.stop)
            source_view = self.gui._source_views[segment.source.index]
            rect = SegmentVisualizer(
                index_conversion(segment.start - i0, i1 - i0, source_view.spectrogram.image.width()),
                180,
                index_conversion(segment.stop - i0, i1 - i0, source_view.spectrogram.image.width() - 1) - index_conversion(segment.start - i0, i1 - i0, source_view.spectrogram.image.width() - 1),
                10,
                segment,
                source_view.spectrogram,
                self.gui
            )
            source_view.spectrogram.addItem(rect)
            self._annotations.append((source_view.spectrogram, rect))

    def on_selection_changed(self):
        """Keeps the table pointed at a visible segment"""
        selection = self.api.get_active_selection()
        if selection:
            xbounds, ybounds, source = selection
            start_times = [int(s.start) for s in self._segmentation_datastore["segments"]]
            segment_idx = np.searchsorted(start_times, int(xbounds[0]))
            index = self.panel.table.model().index(segment_idx, 0)
            self.panel.table.scrollTo(index)

    def on_create_segment_activated(self):
        selection = self.api.get_active_selection()
        if selection:
            xbounds, ybounds, source = selection
            self.create_segment(xbounds[0], xbounds[1], source)

    def create_segment(self, i0: ProjectIndex, i1: ProjectIndex, source: Source):
        new_segment = Segment(i0, i1, source)
        self._segmentation_datastore["segments"].append(new_segment)
        self._segmentation_datastore["segments"].sort()
        self.panel.set_data(self._segmentation_datastore["segments"])
        self.gui.show_status("Created segment on {}".format(source.name))
        self.refresh()

    def plugin_toolbar_items(self):
        return [self.createSegmentButton]

    def plugin_menu(self):
        return None

    def plugin_panel_widget(self):
        return self.panel

        # First, set up a table in the datastore

ExportPlugin = Segmentation
