import logging
import json
from functools import partial
from typing import List, Tuple

import PyQt5.QtWidgets as widgets
import pyqtgraph as pg
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5 import QtGui

from soundsep.core.base_plugin import BasePlugin
from soundsep.core.models import Source, ProjectIndex
from soundsep.core.segments import Segment
from soundsep.core.utils import hhmmss


logger = logging.getLogger(__name__)


class SegmentPanel(widgets.QWidget):

    contextMenuRequested = pyqtSignal(QPoint, object)
    segmentSelectionChanged = pyqtSignal(object)

    # TODO add a filtering dropdown / text box
    # TODO jump to time with click events
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.init_actions()

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        self.table = widgets.QTableWidget(0, 4)
        self.table.setEditTriggers(widgets.QTableWidget.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.DefaultContextMenu)
        header = self.table.horizontalHeader()
        self.table.setHorizontalHeaderLabels([
            "SourceName",
            "Start",
            "Stop",
            "Tags",
        ])
        header.setSectionResizeMode(0, widgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, widgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, widgets.QHeaderView.Stretch)
        header.setSectionResizeMode(3, widgets.QHeaderView.Stretch)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def init_actions(self):
        self.table.itemSelectionChanged.connect(self.on_click)
        # self.table.customContextMenuRequested.connect(self.on_context_menu)

    def contextMenuEvent(self, event):
        pos = event.globalPos()
        self.contextMenuRequested.emit(pos, self.get_selection())

    def on_click(self):
        selection = self.get_selection()
        self.segmentSelectionChanged.emit(selection)

    def get_selection(self):
        selection = []
        ranges = self.table.selectedRanges()
        for selection_range in ranges:
            selection += list(range(selection_range.topRow(), selection_range.bottomRow() + 1))
        return sorted(selection)

    def set_data(self, segments):
        # TODO: this is extremely slow - we need a better way to update the table.
        self.table.setRowCount(len(segments))
        for row, segment in enumerate(segments):
            self.table.setItem(row, 0, widgets.QTableWidgetItem(segment.source.name))
            self.table.setItem(row, 1, widgets.QTableWidgetItem(
                hhmmss(segment.start / segment.project.sampling_rate, dec=3)
            ))
            self.table.setItem(row, 2, widgets.QTableWidgetItem(
                hhmmss(segment.stop / segment.project.sampling_rate, dec=3)
            ))
            self.table.setItem(row, 3, widgets.QTableWidgetItem(
                ",".join(segment.data.get("tags", []))
            ))


class SegmentVisualizer(widgets.QGraphicsRectItem):
    def __init__(
            self,
            segment,
            parent_plot: pg.PlotWidget,
            color,
            opacity,
            draw_fractions: Tuple[float, float],
            plugin):
        y0, y1 = parent_plot.viewRange()[1]
        dy = y1 - y0
        super().__init__(
            segment.start,
            y0 + draw_fractions[0] * dy,
            segment.stop - segment.start,
            (draw_fractions[1] - draw_fractions[0]) * dy,
            parent_plot.plotItem
        )
        self.segment_plugin = plugin
        self.segment = segment
        self.opacity = opacity
        self.color = color
        self.draw_fractions = draw_fractions

        self.setPen(pg.mkPen(self.color, width=2))
        self.setOpacity(self.opacity)
        self.setBrush(pg.mkBrush(None))
        self.setAcceptHoverEvents(True)

        self.setToolTip("{}\n{:.2f}s to {:.2f}s\nDuration: {:.1f} ms\nTags: {}".format(
            self.segment.source.name,
            self.segment.start.to_timestamp(),
            self.segment.stop.to_timestamp(),
            (self.segment.stop.to_timestamp() - self.segment.start.to_timestamp()) * 1000,
            ",".join([t for t in self.segment.data.get("tags", [])]),
        ))

        parent_plot.sigYRangeChanged.connect(self.adjust_ylims)

    def adjust_ylims(self, _, yrange):
        y0, y1 = yrange
        dy = y1 - y0
        self.setRect(
            self.segment.start,
            y0 + self.draw_fractions[0] * dy,
            self.segment.stop - self.segment.start,
            (self.draw_fractions[1] - self.draw_fractions[0]) * dy
        )

    # def mouseClickEvent(self, event):
        # pass
        # event.ignore()
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
        self.setPen(pg.mkPen(self.color, width=4))
        self.segment_plugin.gui.show_status(
            "Segment from {} to {} on {}".format(self.segment.start, self.segment.stop, self.segment.source)
        )

    def hoverLeaveEvent(self, event):
        self.setPen(pg.mkPen(self.color, width=2))
        self.setOpacity(self.opacity)


class SegmentPlugin(BasePlugin):

    SAVE_FILENAME = "segments.csv"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.panel = SegmentPanel()

        self.init_actions()
        self.connect_events()

        self._needs_saving = False
        self._annotations = []

    def init_actions(self):
        self.create_segment_action = widgets.QAction("&Create segment from selection", self)
        self.create_segment_action.triggered.connect(self.on_create_segment_activated)

        self.delete_selection_action = widgets.QAction("&Delete segments in selection", self)
        self.delete_selection_action.triggered.connect(self.on_delete_segment_activated)

        self.merge_selection_action = widgets.QAction("&Merge segments in selection", self)
        self.merge_selection_action.triggered.connect(self.on_merge_segments_activated)

    def connect_events(self):
        self.button = widgets.QPushButton("+Segment")
        self.button.clicked.connect(self.on_create_segment_activated)

        self.delete_button = widgets.QPushButton("-Segments")
        self.delete_button.clicked.connect(self.on_delete_segment_activated)

        self.merge_button = widgets.QPushButton("Merge")
        self.merge_button.clicked.connect(self.on_merge_segments_activated)

        self.panel.contextMenuRequested.connect(self.on_context_menu_requested)
        self.panel.segmentSelectionChanged.connect(self.on_segment_selection_changed)

        self.api.projectLoaded.connect(self.on_project_ready)
        self.api.projectDataLoaded.connect(self.on_project_data_loaded)
        self.api.workspaceChanged.connect(self.on_workspace_changed)
        self.api.selectionChanged.connect(self.on_selection_changed)
        self.api.sourcesChanged.connect(self.on_sources_changed)

    def on_context_menu_requested(self, pos, selection):
        self.tag_menu = widgets.QMenu()
        _, actions = self.api.plugins["TagPlugin"].get_tag_menu(self.tag_menu)
        for tag, action in actions.items():
            action.setCheckable(True)

            selected_tags = [(tag in self._segmentation_datastore[i].data["tags"]) for i in selection]
            if all(selected_tags):
                action.setChecked(True)
            else:
                action.setChecked(False)

            action.triggered.connect(partial(self.api.plugins["TagPlugin"].on_toggle_selection_tag, tag, selection))
        self.tag_menu.popup(pos)

    def on_segment_selection_changed(self, selection):
        # Change workspace to the end of the preceding segment if it exists and the start of the next one
        start = self._segmentation_datastore[selection[0]].start
        stop = self._segmentation_datastore[selection[-1]].stop

        start = self.api.convert_project_index_to_stft_index(start)
        stop = self.api.convert_project_index_to_stft_index(stop)

        # add some padding
        duration = stop - start
        start -= duration // 2
        stop += duration // 2

        self.api.clear_selection()
        self.api.workspace_set_position(start, stop)

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

    @_segmentation_datastore.setter
    def _segmentation_datastore(self, value):
        self._datastore["segments"] = value

    def on_project_ready(self):
        """Called once"""
        save_file = self.api.paths.save_dir / self.SAVE_FILENAME
        if not save_file.exists():
            return

        data = pd.read_csv(save_file, converters={"Tags": str})

        sources = []
        indices = []
        tags = []
        source_lookup = set([
            (source.name, source.channel) for source in self.api.get_sources()
        ])

        def _read(row):
            source_key = (row["SourceName"], row["SourceChannel"])
            if source_key not in source_lookup:
                source_lookup.add(source_key)
                self.api.create_source(source_key[0], source_key[1])
            sources.append(source_key)
            indices.append((row["StartIndex"], row["StopIndex"]))
            if "Tags" in row and row["Tags"]:
                tags.append(set([t for t in json.loads(row["Tags"])]))
            else:
                tags.append(set())
        data.apply(_read, axis=1)

        source_dict = {
            (source.name, source.channel): source
            for source in self.api.get_sources()
        }
        for source_key, (start, stop), segment_tags in zip(sources, indices, tags):
            self._segmentation_datastore.append(Segment(
                self.api.make_project_index(start),
                self.api.make_project_index(stop),
                source_dict[source_key],
                data={"tags": segment_tags}
            ))

        self._segmentation_datastore.sort()

    def on_project_data_loaded(self):
        self.panel.set_data(self._segmentation_datastore)
        self.refresh()

    def needs_saving(self):
        return self._needs_saving

    def save(self):
        """Save pointers within project"""
        # TODO: these pointers could get out of sync with a project if/when files are added.
        # Can we recover from this? or should we hash the project so we can at least
        # warn the user when things dont match up to when the file was saved?
        segment_dicts = []
        for segment in self._segmentation_datastore:
            segment_dicts.append({
                "SourceName": segment.source.name,
                "SourceChannel": segment.source.channel,
                "StartIndex": int(segment.start),
                "StopIndex": int(segment.stop),
                "Tags": json.dumps(list(segment.data["tags"]))
            })
        pd.DataFrame(segment_dicts).to_csv(self.api.paths.save_dir / self.SAVE_FILENAME)
        self._needs_saving = False

    def on_sources_changed(self):
        # We need to check if any sources have been deleted
        existing_sources = set(self.api.get_sources())
        self._segmentation_datastore = [
            s for s in self._segmentation_datastore if s.source in existing_sources
        ]
        self.refresh()

    def on_workspace_changed(self):
        self.refresh()

    def on_selection_changed(self):
        self.refresh()

    def jump_to_selection(self):
        selection = self.api.get_selection()
        if selection is not None:
            start_times = [int(s.start) for s in self._segmentation_datastore]
            first_selection_idx = np.searchsorted(start_times, selection.x0)
            index = self.panel.table.model().index(first_selection_idx, 0)
            self.panel.table.scrollTo(index, QtGui.QAbstractItemView.PositionAtTop)

    def refresh(self):
        """Keeps the table pointed at a selected region

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

        selection = self.api.get_fine_selection()

        # TODO: Its not that bad to draw every rectangle at once; but the annoying part is deleting them
        # and redrawing them when something changes
        for segment in self._segmentation_datastore[first_segment_idx:last_segment_idx]:
            source_view = self.gui.source_views[segment.source.index]
            rect = SegmentVisualizer(segment, source_view.spectrogram, "#00ff00", 0.3, (0.05, 0.95), self)
            source_view.spectrogram.addItem(rect)
            self._annotations.append((source_view.spectrogram, rect))

            if selection and segment.source == selection.source:
                rect = SegmentVisualizer(segment, self.gui.ui.previewPlot, "#00aa00", 0.3, (0.4, 0.6), self)
                self.gui.ui.previewPlot.addItem(rect)
                self._annotations.append((self.gui.ui.previewPlot, rect))

    def on_delete_segment_activated(self):
        selection = self.api.get_fine_selection()
        if selection:
            self.delete_segments(selection.x0, selection.x1, selection.source)

    def on_merge_segments_activated(self):
        selection = self.api.get_fine_selection()
        if selection:
            self.merge_segments(selection.x0, selection.x1, selection.source)

    def on_create_segment_activated(self):
        selection = self.api.get_fine_selection()
        if selection:
            self.create_segment(
                selection.x0,
                selection.x1,
                selection.source
            )

    def create_segments_batch(self, segment_data: List[Tuple[ProjectIndex, ProjectIndex, Source]]):
        """Create multiple segments, only updating the display one time at the end"""
        for start, stop, source in segment_data:
            self._segmentation_datastore = [
                segment for segment in self._segmentation_datastore
                if (
                    (segment.start <= start and segment.stop <= start) or
                    (segment.start >= stop and segment.stop >= stop)
                ) or source != segment.source
            ]
            new_segment = Segment(start, stop, source, data={"tags": set()})
            self._segmentation_datastore.append(new_segment)
        self._segmentation_datastore.sort()
        self.panel.set_data(self._segmentation_datastore)
        self.gui.show_status("Created {} segments".format(len(segment_data)))
        logger.debug("Created {} segments".format(len(segment_data)))
        self._needs_saving = True
        self.refresh()

    def create_segment(self, start: ProjectIndex, stop: ProjectIndex, source: Source):
        self.delete_segments(start, stop, source)
        new_segment = Segment(start, stop, source, data={"tags": set()})
        self._segmentation_datastore.append(new_segment)
        self._segmentation_datastore.sort()
        self.panel.set_data(self._segmentation_datastore)
        self.gui.show_status("Created segment {} to {}".format(start, stop))
        logger.debug("Created segment {} to {}".format(start, stop))
        self._needs_saving = True
        self.refresh()

    def delete_segments(self, start: ProjectIndex, stop: ProjectIndex, source: Source, refresh: bool = True):
        filtered_segments = [
            segment for segment in self._segmentation_datastore
            if (
                (segment.start <= start and segment.stop <= start) or
                (segment.start >= stop and segment.stop >= stop)
            ) or source != segment.source
        ]
        n_deleted = len(self._segmentation_datastore) - len(filtered_segments)
        self.gui.show_status("Deleting {} segments from {} to {}".format(n_deleted, start, stop))
        logger.debug("Deleting {} segments from {} to {}".format(n_deleted, start, stop))
        self._segmentation_datastore.clear()
        self._segmentation_datastore.extend(filtered_segments)
        self.panel.set_data(self._segmentation_datastore)
        self._needs_saving = True
        if refresh:
            self.refresh()

    def merge_segments(self, start: ProjectIndex, stop: ProjectIndex, source: Source):
        to_merge = [
            segment for segment in self._segmentation_datastore
            if (
                (segment.start <= start and segment.stop >= start) or
                (segment.start <= stop and segment.stop >= stop) or
                (segment.start >= start and segment.stop <= stop)
            ) and source == segment.source
        ]
        if not len(to_merge):
            return

        self.gui.show_status("Merging {} segments from {} to {}".format(len(to_merge), start, stop))
        logger.debug("Merging {} segments from {} to {}".format(len(to_merge), start, stop))

        new_tags = set.union(*[segment.data["tags"] for segment in to_merge])
        new_segment = Segment(to_merge[0].start, to_merge[-1].stop, source, data={"tags": new_tags})
        filtered_segments = [
            segment for segment in self._segmentation_datastore
            if (
                (segment.start <= start and segment.stop <= start) or
                (segment.start >= stop and segment.stop >= stop)
            ) or source != segment.source
        ]
        self._segmentation_datastore.clear()
        self._segmentation_datastore.extend(filtered_segments + [new_segment])
        self._segmentation_datastore.sort()
        self.panel.set_data(self._segmentation_datastore)
        self._needs_saving = True
        self.refresh()

    def plugin_toolbar_items(self):
        return [self.button, self.delete_button, self.merge_button]

    def add_plugin_menu(self, menu_parent):
        menu = menu_parent.addMenu("&Segments")
        menu.addAction(self.create_segment_action)
        menu.addAction(self.delete_selection_action)
        menu.addAction(self.merge_selection_action)
        return menu

    def plugin_panel_widget(self):
        return self.panel

    def setup_plugin_shortcuts(self):
        self.create_segment_action.setShortcut(QtGui.QKeySequence("F"))
        self.delete_selection_action.setShortcut(QtGui.QKeySequence("X"))
        self.merge_selection_action.setShortcut(QtGui.QKeySequence("Q"))
