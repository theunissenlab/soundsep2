import logging

import PyQt5.QtWidgets as widgets
import pyqtgraph as pg
import numpy as np
from PyQt5 import QtGui

from soundsep.core.base_plugin import BasePlugin
from soundsep.core.models import Source, ProjectIndex, StftIndex
from soundsep.core.segments import Segment


logger = logging.getLogger(__name__)


class SegmentPanel(widgets.QWidget):

    # TODO add a filtering dropdown / text box
    # TODO jump to time with click events
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


class SegmentVisualizer(QtGui.QGraphicsRectItem):
    def __init__(self, segment, spectrogram, plugin):
        super().__init__(
            segment.start,
            spectrogram.viewRange()[1][0],
            segment.stop - segment.start,
            spectrogram.viewRange()[1][1],
            spectrogram.image
        )
        self.segment_plugin = plugin
        self.segment = segment
        self.spectrogram = spectrogram

        self.setPen(pg.mkPen("#00ff00", width=2))
        self.setOpacity(0.3)
        self.setBrush(pg.mkBrush(None))
        self.setAcceptHoverEvents(True)

        self.setToolTip("{}\n{:.2f}s to {:.2f}s".format(
            self.segment.source.name,
            self.segment.start.to_timestamp(),
            self.segment.stop.to_timestamp(),
        ))

    def mouseClickEvent(self, event):
        pass
        # TODO: Calling set_selection is hazardous here because it doesnt update
        # the roi in the main GUI. maybe an alternative would be call gui.draw_selection?
        # _, _, freqs = self.segment_plugin.api.get_workspace_stft()
        # self.segment_plugin.api.set_selection(
        #     self.segment.start,
        #     self.segment.stop,
        #     0,
        #     np.max(freqs),
        #     self.segment.source,
        # )

    def hoverEnterEvent(self, event):
        """Draw vertical lines as boundaries"""
        self.setOpacity(1.0)
        self.segment_plugin.gui.show_status(
            "Segment from {} to {} on {}".format(self.segment.start, self.segment.stop, self.segment.source)
        )

    def hoverLeaveEvent(self, event):
        self.setOpacity(0.3)


class SegmentPlugin(BasePlugin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.panel = SegmentPanel()

        self.init_actions()
        self.connect_events()

        self._annotations = []

    def init_actions(self):
        self.create_segment_action = widgets.QAction("Create segment from selection", self)
        self.create_segment_action.triggered.connect(self.on_create_segment_activated)

        self.delete_selection_action = widgets.QAction("Delete segments in selection", self)
        self.delete_selection_action.triggered.connect(self.on_delete_segment_activated)

    def connect_events(self):
        self.button = widgets.QPushButton("+Segment")
        self.button.clicked.connect(self.on_create_segment_activated)

        self.delete_button = widgets.QPushButton("-Segments")
        self.delete_button.clicked.connect(self.on_delete_segment_activated)

        self.api.workspaceChanged.connect(self.on_workspace_changed)

    @property
    def _datastore(self):
        return self.api.get_mut_datastore()

    @property
    def _segmentation_datastore(self):
        datastore = self._datastore
        if "segments" in datastore:
            return datastore["segments"]
        else:
            datastore["segments"] = []
            return datastore["segments"]

    def on_workspace_changed(self, x: StftIndex, y: StftIndex):
        self.refresh()

    def refresh(self):
        """Keeps the table pointed at a visible segment

        Refresh the rectangles drawn on spectrogram views
        """
        ws0, ws1 = self.api.workspace_get_lim()
        ws0 = ws0.to_project_index()
        ws1 = ws1.to_project_index()

        # Also highlight all points visible
        for parent, annotation in self._annotations:
            try:
                parent.removeItem(annotation)
            except RuntimeError:
                # This can happen if the parent was destroyed prior to this fn called
                pass
        self._annotations = []

        # Find the row in the table of the first visible segment
        start_times = [int(s.start) for s in self._segmentation_datastore]
        first_segment_idx = np.searchsorted(start_times, ws0)
        last_segment_idx = np.searchsorted(start_times, ws1)

        index = self.panel.table.model().index(first_segment_idx, 0)
        self.panel.table.scrollTo(index)

        for segment in self._segmentation_datastore[first_segment_idx:last_segment_idx]:
            source_view = self.gui.source_views[segment.source.index]
            rect = SegmentVisualizer(segment, source_view.spectrogram, self)
            source_view.spectrogram.addItem(rect)
            self._annotations.append((source_view.spectrogram, rect))

    def on_delete_segment_activated(self):
        selection = self.api.get_selection()
        if selection:
            self.delete_segments(selection.x0, selection.x1, selection.source)

    def on_create_segment_activated(self):
        selection = self.api.get_selection()
        if selection:
            self.create_segment(
                selection.x0,
                selection.x1,
                selection.source
            )

    def create_segment(self, start: ProjectIndex, stop: ProjectIndex, source: Source):
        self.delete_segments(start, stop, source)
        new_segment = Segment(start, stop, source)
        self._segmentation_datastore.append(new_segment)
        self._segmentation_datastore.sort()
        self.panel.set_data(self._segmentation_datastore)
        self.gui.show_status("Created segment {} to {}".format(start, stop))
        logger.debug("Created segment {} to {}".format(start, stop))
        self.refresh()

    def delete_segments(self, start: ProjectIndex, stop: ProjectIndex, source: Source):
        filtered_segments = [
            segment for segment in self._segmentation_datastore
            if (
                (segment.start <= start and segment.stop <= start) or
                (segment.start >= stop and segment.stop >= stop)
            ) or source != segment.source
        ]
        self._segmentation_datastore.clear()
        self._segmentation_datastore.extend(filtered_segments)
        self.refresh()

    def plugin_toolbar_items(self):
        return [self.button]

    def add_plugin_menu(self, menu_parent):
        menu = menu_parent.addMenu("&Segments")
        menu.addAction(self.create_segment_action)
        menu.addAction(self.delete_selection_action)
        return menu

    def plugin_panel_widget(self):
        return self.panel

    def setup_plugin_shortcuts(self):
        self.create_segment_action.setShortcut(QtGui.QKeySequence("F"))
        self.delete_selection_action.setShortcut(QtGui.QKeySequence("X"))
