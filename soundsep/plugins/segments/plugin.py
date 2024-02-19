import logging
import json
from functools import partial
from typing import List, Tuple

import PyQt6.QtWidgets as widgets
import pyqtgraph as pg
import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6 import QtGui

from soundsep.core.base_plugin import BasePlugin
from soundsep.core.models import Source, ProjectIndex
from soundsep.core.segments import Segment
from soundsep.core.utils import hhmmss


logger = logging.getLogger(__name__)

class UMAPVisPanel(widgets.QWidget):
    segmentSelectionChanged = pyqtSignal(object)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.init_actions()
        self.npoints = 0

    def init_ui(self):
        # setup a 2d plot
        self.plot = pg.plot()
        self.scatter = pg.ScatterPlotItem()
        self.plot.addItem(self.scatter)
        layout = widgets.QGridLayout()
        layout.addWidget(self.plot,0,0)
        self.setLayout(layout)
    
    def init_actions(self):
        self.scatter.sigClicked.connect(self.on_click)
        return
    
    def on_click(self, plot, points):
        if len(points) > 0:
            # TODO what to do for multiselect
            self.segmentSelectionChanged.emit([points[0].data()])
        return

    def on_selection_changed(self, selection):
        sizes = np.ones(self.npoints) * 10
        spot_inds = [spot['data'] for spot in self.scatter.data]
        sel_inds = [spot_inds.index(s) for s in selection]
        sizes[sel_inds] = 20
        self.scatter.setSize(sizes)
    
    def set_data(self, segments, func_get_color=None):
        # TODO: this is extremely slow - we need a better way to update the table.
        # make a scatter plot for the segments

        spots = []
        for ix,s_row in segments.iterrows():
            if func_get_color and len(s_row['Tags']) > 0:
                c = func_get_color(list(s_row['Tags'])[0])
            else:
                c = 'r'
            if len(s_row['Coords']) >= 2:
                spots.append(dict({
                    'pos': s_row['Coords'][:2],
                    'data': ix,
                    'brush': pg.mkBrush(c),
                    'size': 10
                }))
        
        self.npoints = len(spots)
        self.scatter.setData(
            spots=spots,
            hoverSize=20,
            hoverable=True
        )

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
        self.table.setEditTriggers(widgets.QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        header = self.table.horizontalHeader()
        self.table.setHorizontalHeaderLabels([
            "SourceName",
            "Start",
            "Stop",
            "Tags",
        ])
        header.setSectionResizeMode(0, widgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, widgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, widgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, widgets.QHeaderView.ResizeMode.Stretch)
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

    def on_selection_changed(self, selection):
        self.table.itemSelectionChanged.disconnect(self.on_click)
        self.set_selection(selection)
        self.table.itemSelectionChanged.connect(self.on_click)

    def set_selection(self, selection):
        self.table.clearSelection()
        for row in selection:
            self.table.selectRow(row)
    
    def get_selection(self):
        selection = []
        ranges = self.table.selectedRanges()
        for selection_range in ranges:
            selection += list(range(selection_range.topRow(), selection_range.bottomRow() + 1))
        return sorted(selection)

    def set_data(self, segments, project):
        # TODO Store indices
        # TODO: this is extremely slow - we need a better way to update the table.
        self.table.setRowCount(len(segments))
        for row, segment_row in segments.iterrows():
            self.table.setItem(row, 0, widgets.QTableWidgetItem(segment_row['Source'].name))
            self.table.setItem(row, 1, widgets.QTableWidgetItem(
                hhmmss(segment_row['StartIndex'] / project.sampling_rate, dec=3)
            ))
            self.table.setItem(row, 2, widgets.QTableWidgetItem(
                hhmmss(segment_row['StopIndex'] / project.sampling_rate, dec=3)
            ))
            self.table.setItem(row, 3, widgets.QTableWidgetItem(
                ",".join(segment_row["Tags"])
            ))

    def add_row(self, row_ind, segment, project):
        self.table.insertRow(row_ind)
        self.table.setItem(row_ind, 0, widgets.QTableWidgetItem(segment['Source'].name))
        self.table.setItem(row_ind, 1, widgets.QTableWidgetItem(
            hhmmss(segment['StartIndex'] / project.sampling_rate, dec=3)
        ))
        self.table.setItem(row_ind, 2, widgets.QTableWidgetItem(
            hhmmss(segment['StopIndex'] / project.sampling_rate, dec=3)
        ))
        self.table.setItem(row_ind, 3, widgets.QTableWidgetItem(
            ",".join(segment["Tags"])
        ))
    
    def remove_row(self, row_ind):
        self.table.removeRow(row_ind)

# class SegmentTableView(widgets.QTableView):
#     def __init__(self):
#         super(SegmentTableView, self).__init__()

#         self.model = DataModel()

#         self.view = QtWidgets.QTableView(self)
#         self.view.setAlternatingRowColors(True)
#         self.view.setStyleSheet('alternate-background-color: grey;background-color: white;')

#         self.proxy = QtCore.QSortFilterProxyModel(self)
#         self.proxy.setSourceModel(self.model)

#         self.view.setModel(self.proxy)
# class SegmentDataModel(QtCore.QAbstractTableModel):
#   data_updated = QtCore.pyqtSignal()
#   def __init__(self, data: pd.DataFrame = pd.DataFrame()):
#     super(SegmentDataModel, self).__init__()
#     self._data = data

#   def update_data(self, data: pd.DataFrame):
#     self.beginRestModel()
#     self._data = data
#     self.endResetModel()
#     self.data_updated.emit()

#   def data(self, index, role):
#     if role == QtCore.Qt.DisplayRole:
#       return str(self._data.iloc[index.row()][index.column()])

#   def rowCount(self, index = None) -> int:
#     return self._data.shape[0]
  
#   def columnCount(self, index = None) -> int:
#     return self._data.shape[1]

#   def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int ) -> Any:
#     # section is the index of the column/row.
#     if role == QtCore.Qt.DisplayRole:
#       if orientation == QtCore.Qt.Horizontal:
#         return str(self._data.columns[section])
#       if orientation == QtCore.Qt.Vertical:
#         return str(self._data.index[section])

class SegmentVisualizer(widgets.QGraphicsRectItem):
    def __init__(
            self,
            segment,
            parent_plot: pg.PlotWidget,
            color,
            width,
            opacity,
            draw_fractions: Tuple[float, float],
            plugin):
        y0, y1 = parent_plot.viewRange()[1]
        dy = y1 - y0
        super().__init__(
            segment.StartIndex,
            y0 + draw_fractions[0] * dy,
            segment.StopIndex - segment.StartIndex,
            (draw_fractions[1] - draw_fractions[0]) * dy,
            parent_plot.plotItem
        )
        self.segment_plugin = plugin
        self.segment = segment
        self.opacity = opacity
        self.color = color
        self.draw_fractions = draw_fractions

        self.setPen(pg.mkPen(self.color, width=width))
        self.setOpacity(self.opacity)
        self.setBrush(pg.mkBrush(None))
        self.setAcceptHoverEvents(True)

        self.setToolTip("{}\n{:.2f}s to {:.2f}s\nDuration: {:.1f} ms\nTags: {}".format(
            self.segment.Source.name,
            self.segment.StartIndex.to_timestamp(),
            self.segment.StopIndex.to_timestamp(),
            (self.segment.StopIndex.to_timestamp() - self.segment.StartIndex.to_timestamp()) * 1000,
            ",".join([t for t in self.segment.Tags]),
        ))

        parent_plot.sigYRangeChanged.connect(self.adjust_ylims)

    def adjust_ylims(self, _, yrange):
        y0, y1 = yrange
        dy = y1 - y0
        self.setRect(
            self.segment.StartIndex,
            y0 + self.draw_fractions[0] * dy,
            self.segment.StopIndex - self.segment.StartIndex,
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
            "Segment from {} to {} on {}".format(self.segment.StartIndex, self.segment.StopIndex, self.segment.Source)
        )

    def hoverLeaveEvent(self, event):
        self.setPen(pg.mkPen(self.color, width=2))
        self.setOpacity(self.opacity)


class SegmentPlugin(BasePlugin):

    SAVE_FILENAME = "segments.csv"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.panel = SegmentPanel()
        self.umap_panel = UMAPVisPanel()

        self.init_actions()
        self.connect_events()

        self._needs_saving = False
        self._annotations = []
        self._selected_segments = []

    def init_actions(self):
        self.create_segment_action = QtGui.QAction("&Create segment from selection", self)
        self.create_segment_action.triggered.connect(self.on_create_segment_activated)

        self.delete_selection_action = QtGui.QAction("&Delete segments in selection", self)
        self.delete_selection_action.triggered.connect(self.on_delete_segment_activated)

        self.merge_selection_action = QtGui.QAction("&Merge segments in selection", self)
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
        self.umap_panel.segmentSelectionChanged.connect(self.on_segment_selection_changed)
        # TODO hook both of them up to this signal too

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

            selected_tags = [(tag in self._segmentation_datastore.loc[i]["Tags"]) for i in selection]
            if all(selected_tags):
                action.setChecked(True)
            else:
                action.setChecked(False)

            action.triggered.connect(partial(self.api.plugins["TagPlugin"].on_toggle_selection_tag, tag, selection))
        self.tag_menu.popup(pos)

    def on_segment_selection_changed(self, selection):
        if self._selected_segments == selection:
            return
        # Change workspace to the end of the preceding segment if it exists and the start of the next one
        start = self._segmentation_datastore.loc[selection].StartIndex.min()
        stop = self._segmentation_datastore.loc[selection].StopIndex.max()

        # get bounds of current view, to determine final duration
        ws_start, ws_stop = self.api.workspace_get_lim()

        start = self.api.convert_project_index_to_stft_index(start)
        stop = self.api.convert_project_index_to_stft_index(stop)

        # # if start is within the current bounds, dont edit the start
        # if start > ws_start and start < ws_stop:
        #     start = ws_start
        # # if stop is within the current bounds, dont edit the stop
        # if stop > ws_start and stop < ws_stop:
        #     stop = ws_stop
        # add some padding
        # TODO BUG HERE: IF ALL SEGMENTS ARE SELECTED, then this errors
        duration = max(stop - start, ws_stop - ws_start)
        start.value = (start+stop) // 2
        stop.value = start.value
        start -= duration // 2
        stop += duration // 2

        self._selected_segments = selection
        # call the UI Selection changes
        self.panel.on_selection_changed(selection)
        self.umap_panel.on_selection_changed(selection)

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
            datastore["segments"] = pd.DataFrame(dict({
                "Source":[],
                "StartIndex": [],
                "StopIndex": [],
                "Tags": [],
                "Coords": []
            }))
            return datastore["segments"]

    @_segmentation_datastore.setter
    def _segmentation_datastore(self, value):
        # check that the value is a pandas dataframe
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Segmentation datastore must be a pandas dataframe")
        # check that it has the requisite columns
        if not all([c in value.columns for c in ["Source", "StartIndex", "StopIndex", "Tags", "Coords"]]):
            raise ValueError("Segmentation datastore must have columns SourceName, SourceChannel, StartIndex, StopIndex, Tags, Coords")
        self._datastore["segments"] = value

    def on_project_ready(self):
        """Called once"""
        save_file = self.api.paths.save_dir / self.SAVE_FILENAME
        if not save_file.exists():
            return

        data = pd.read_csv(save_file, converters={"Tags": str, "Coords": str})

        seg_df = dict({
                "Source": [],
                "StartIndex": [],
                "StopIndex": [],
                "Tags": [],
                "Coords": []
            })
        
        source_lookup = set([
            (source.name, source.channel) for source in self.api.get_sources()
        ])

        def _read(row):
            source_key = (row["SourceName"], row["SourceChannel"])
            if source_key not in source_lookup:
                source_lookup.add(source_key)
                self.api.create_source(source_key[0], source_key[1])
            source = self.api.get_source(source_key[0], source_key[1])
            seg_df['Source'].append(source)
            seg_df['StartIndex'].append(self.api.make_project_index(row["StartIndex"]))
            seg_df['StopIndex'].append(self.api.make_project_index(row["StopIndex"]))
            if "Tags" in row and row["Tags"]:
                seg_df['Tags'].append(set([t for t in json.loads(row["Tags"])]))
            else:
                seg_df['Tags'].append(set())
            if 'Coords' in row and row['Coords']:
                seg_df['Coords'].append(list([float(x) for x in json.loads(row['Coords'])]))
            else:
                # TODO figure out what we want to do in the case that coords is not there. Could sort by amplitude and duration or something
                seg_df['Coords'].append(list([row['StopIndex'] - row['StartIndex'], np.random.rand()])) 
        data.apply(_read, axis=1)
        for l in ['StartIndex', 'StopIndex']:
            seg_df[l] = pd.Series(seg_df[l],dtype=object)

        self._segmentation_datastore = pd.DataFrame(seg_df)
        #self._segmentation_datastore.sort()

    def on_project_data_loaded(self):
        self.panel.set_data(self._segmentation_datastore,self.api.project)
        self.umap_panel.set_data(self._segmentation_datastore, self.api.plugins["TagPlugin"].get_tag_color)
        self.refresh()

    def needs_saving(self):
        return self._needs_saving

    def save(self):
        """Save pointers within project"""
        # TODO: these pointers could get out of sync with a project if/when files are added.
        # Can we recover from this? or should we hash the project so we can at least
        # warn the user when things dont match up to when the file was saved?
        # LAT - Not sure if we are still saving pointers here
        
        self._segmentation_datastore.to_csv(self.api.paths.save_dir / self.SAVE_FILENAME)
        self._needs_saving = False

    def on_sources_changed(self):
        # We need to check if any sources have been deleted and remove their segments
        # TODO: maybe we just shouldnt show the segments in the panel
        self._segmentation_datastore = self._segmentation_datastore[
            self._segmentation_datastore['Source'].isin(self.api.get_sources())]
        self.refresh()

    def on_workspace_changed(self):
        self.refresh()

    def on_selection_changed(self):
        self.refresh()

    def jump_to_selection(self):
        selection = self.api.get_selection()
        if selection is not None:
            start_times = self._segmentation_datastore.StartIndex
            first_selection_idx = np.searchsorted(start_times, selection.x0)
            index = self.panel.table.model().index(first_selection_idx, 0)
            self.panel.table.scrollTo(index, QtGui.QAbstractItemView.PositionAtTop)
            # TODO highlight the scatter element in the UMAP plot

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

        # # Find the row in the table of the first visible segment
        # first_segment_idx = self._segmentation_datastore['StopIndex'].searchsorted(ws0)
        # last_segment_idx = self._segmentation_datastore['StartIndex'].searchsorted(ws1)
        # get all segments where the start index is less than ws1 and the stop index is greater than ws0
        segs_in_view = self._segmentation_datastore[ (self._segmentation_datastore['StartIndex'] < ws1) & (self._segmentation_datastore['StopIndex'] > ws0) ]
        selection = self.api.get_fine_selection()
        
        # Go through each source view and draw the segments
        for source in self.api.get_sources():
            source_view = self.gui.source_views[source.index]
            source_segs = segs_in_view[ segs_in_view['Source'] == source ]
            for idx, segment_row in source_segs.iterrows():
                # get the color of the first tag TODO maybe make this different than tags
                if len(segment_row["Tags"]) == 0:
                    c = "#00ff00"
                else:
                    t = list(segment_row["Tags"])[0]
                    c = self.api.plugins["TagPlugin"].get_tag_color(t, as_hex=True)
                # if this segment is selected in the Segment Table then color it differently
                if idx in self._selected_segments:
                    rect = SegmentVisualizer(segment_row, source_view.spectrogram, c, 4, 0.6, (0.1, 0.9), self)
                else:
                    rect = SegmentVisualizer(segment_row, source_view.spectrogram, c, 2, 0.6, (0.05, 0.95), self)
                source_view.spectrogram.addItem(rect)
                self._annotations.append((source_view.spectrogram, rect))

                if selection and source == selection.source:
                    rect = SegmentVisualizer(segment_row, self.gui.ui.previewPlot, "#00aa00", 2, 0.3, (0.4, 0.6), self)
                    self.gui.ui.previewPlot.addItem(rect)
                    self._annotations.append((self.gui.ui.previewPlot, rect))

    def on_delete_segment_activated(self):
        selection = self.api.get_fine_selection()
        if selection:
            self.delete_segments_between(selection.x0, selection.x1, selection.source)

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
        # for now lets do this one at a time and see if its still slow
        for start, stop, source in segment_data:
            self.create_segment(start, stop, source)
        # Remove all segments from seg datastore that are within the new segments
        # inds_to_remove = []
        # for start, stop, source in segment_data:
        #     inds_to_remove.append((self._segmentation_datastore['StartIndex'] <= start) 
        #                     & (self._segmentation_datastore['StopIndex'] >= stop) 
        #                     & (self._segmentation_datastore['SourceName'] == source.name) 
        #                     & (self._segmentation_datastore['SourceChannel'] == source.channel))
        #     # TODO may need to delete all the ones we need to delete, then add the new segment
        #     self._segmentation_datastore.append(new_segment)
        
        # self._segmentation_datastore.sort()
        # self.panel.set_data(self._segmentation_datastore)
        # # TODO kinda unnecessary here, as these tags are not UMAPPED, but just to keep the counts right
        # self.umap_panel.set_data(self._segmentation_datastore, self.api.plugins["TagPlugin"].get_tag_color)
        # self.gui.show_status("Created {} segments".format(len(segment_data)))
        # logger.debug("Created {} segments".format(len(segment_data)))
        # self._needs_saving = True
        # self.refresh()

    def create_segment(self, start: ProjectIndex, stop: ProjectIndex, source: Source, tags: set = set(), coords: list = list()):
        self.delete_segments_between(start, stop, source)
        new_segment = dict({
            "StartIndex": start,
            "StopIndex": stop,
            "Source": source,
            "Tags": tags,
            "Coords": coords
        })
        # Insert this into the correct place in the dataframe, relative to the start index
        ind = self._segmentation_datastore['StartIndex'].searchsorted(start)
        self._segmentation_datastore = pd.concat([self._segmentation_datastore.iloc[:ind], pd.Series(), self._segmentation_datastore.iloc[ind:]]).reset_index(drop=True)
        self._segmentation_datastore.loc[ind]['StartIndex'] = start
        self._segmentation_datastore.loc[ind]['StopIndex'] = stop
        self._segmentation_datastore.loc[ind]['Source'] = source
        self._segmentation_datastore.loc[ind]['Tags'] = tags
        self._segmentation_datastore.loc[ind]['Coords'] = coords
        # self._segmentation_datastore.loc[len(self._segmentation_datastore)] = pd.Series()
        # self._segmentation_datastore.loc[len(self._segmentation_datastore)]['StartIndex'] = start
        # self._segmentation_datastore.loc[len(self._segmentation_datastore)]['StopIndex'] = stop
        # self._segmentation_datastore.loc[len(self._segmentation_datastore)]['Source'] = source
        # self._segmentation_datastore.loc[len(self._segmentation_datastore)]['Tags'] = tags
        # self._segmentation_datastore.loc[len(self._segmentation_datastore)]['Coords'] = coords
        
        
        
        # TODO change panel to add a single row
        self.panel.add_row(ind, self._segmentation_datastore.loc[ind], self.api.project)
        #self.panel.set_data(self._segmentation_datastore, self.api.project)
        self.umap_panel.set_data(self._segmentation_datastore, self.api.plugins["TagPlugin"].get_tag_color)
        self.gui.show_status("Created segment {} to {}".format(start, stop))
        logger.debug("Created segment {} to {}".format(start, stop))
        self._needs_saving = True
        self.refresh()



    def delete_segments_between(self, start: ProjectIndex, stop: ProjectIndex, source: Source, refresh: bool = True):
        # Delete all segments from this source who have a start OR stop index within the range
        segs_to_delete = ((self._segmentation_datastore['StopIndex'].between(start,stop) |\
                                self._segmentation_datastore['StartIndex'].between(start,stop)) &\
                            (self._segmentation_datastore['Source'] == source))
        
        deleted_inds = self._segmentation_datastore[segs_to_delete].index
        if len(deleted_inds) == 0:
            return
        self.delete_segments(deleted_inds)

    def delete_segments(self, seg_indices, refresh: bool = True):
        # Delete all segments from this source who have a start OR stop index within the range
        n_deleted = len(seg_indices)
        self._segmentation_datastore.drop(seg_indices, inplace=True)
        self._segmentation_datastore.reset_index(inplace=True, drop=True)
        self.gui.show_status("Deleting {} segments".format(n_deleted))
        logger.debug("Deleting {} segments".format(n_deleted))
        # TODO change panel delete to take a list of indices
        for ind in seg_indices:
            self.panel.remove_row(ind)
        #self.panel.set_data(self._segmentation_datastore, self.api.project)
        self.umap_panel.set_data(self._segmentation_datastore, self.api.plugins["TagPlugin"].get_tag_color)
        self._needs_saving = True
        if refresh:
            self.refresh()

    def merge_segments(self, start: ProjectIndex, stop: ProjectIndex, source: Source):
        # Merge all segments from this source who have a start OR stop index within the range
        segs_to_merge = self._segmentation_datastore[((self._segmentation_datastore['StopIndex'].between(start, stop) |\
                            self._segmentation_datastore['StartIndex'].between(start, stop)) &\
                            (self._segmentation_datastore['Source'] == source))]

        if not len(segs_to_merge):
            return

        self.gui.show_status("Merging {} segments from {} to {}".format(len(segs_to_merge), start, stop))
        logger.debug("Merging {} segments from {} to {}".format(len(segs_to_merge), start, stop))
        new_tags = set.union(*list(segs_to_merge['Tags'].values))
        new_start = min(segs_to_merge['StartIndex'])
        new_stop = max(segs_to_merge['StopIndex'])
        # TODO, can maybe take coords too? or the mean
        self.delete_segments(segs_to_merge.index, refresh=False)
        self.create_segment(new_start, new_stop, source, new_tags)

    def plugin_toolbar_items(self):
        return [self.button, self.delete_button, self.merge_button]

    def add_plugin_menu(self, menu_parent):
        menu = menu_parent.addMenu("&Segments")
        menu.addAction(self.create_segment_action)
        menu.addAction(self.delete_selection_action)
        menu.addAction(self.merge_selection_action)
        return menu

    def plugin_panel_widget(self):
        return [self.panel,self.umap_panel]

    def setup_plugin_shortcuts(self):
        self.create_segment_action.setShortcut(QtGui.QKeySequence("F"))
        self.delete_selection_action.setShortcut(QtGui.QKeySequence("X"))
        self.merge_selection_action.setShortcut(QtGui.QKeySequence("Q"))
