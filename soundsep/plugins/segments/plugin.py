import logging
import json
from functools import partial
from typing import List, Tuple

import PyQt6.QtWidgets as widgets
import pyqtgraph as pg
import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QAbstractTableModel, QModelIndex, QItemSelectionModel
from PyQt6 import QtGui

from soundsep.core.base_plugin import BasePlugin
from soundsep.core.models import Source, ProjectIndex, StftIndex, NWBFile
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
        sel_inds = []
        for s in selection:
            if s in spot_inds:
                sel_inds.append(spot_inds.index(s))
        if len(sel_inds) > 0:
            sizes[sel_inds] = 20
        self.scatter.setSize(sizes)
    
    def set_data(self, segments, func_get_color=None):
        """Set scatter plot data from segments DataFrame.

        Optimized to extract data using vectorized column access rather than
        row-by-row iteration.
        """
        if len(segments) == 0:
            self.npoints = 0
            self.scatter.setData(spots=[], hoverSize=20, hoverable=True)
            return

        # Extract columns as lists for faster access
        seg_ids = segments.index.tolist()
        coords_list = segments['Coords'].tolist()
        tags_list = segments['Tags'].tolist()

        # Build spots list - only include segments with valid coords
        spots = []
        for ix, coords, tags in zip(seg_ids, coords_list, tags_list):
            if coords is not None and len(coords) >= 2:
                if func_get_color and len(tags) > 0:
                    c = func_get_color(list(tags)[0])
                else:
                    c = 'r'
                spots.append({
                    'pos': coords[:2],
                    'data': ix,
                    'brush': pg.mkBrush(c),
                    'size': 10
                })

        self.npoints = len(spots)
        self.scatter.setData(
            spots=spots,
            hoverSize=20,
            hoverable=True
        )
    
    def remove_spots(self, segIDs):
        visibilities = self.scatter.data['visible']
        spot_seg_IDs = [spot['data'] for spot in self.scatter.data]
        for segID in segIDs:
            if segID in spot_seg_IDs:
                visibilities[spot_seg_IDs.index(segID)] = False
                #spot_inds.append(seg_IDs.index(segID))
        self.scatter.setPointsVisible(visibilities)

    def add_spot(self, segment, func_get_color=None):
        if func_get_color and len(segment['Tags']) > 0:
            c = func_get_color(list(segment['Tags'])[0])
        else:
            c = 'r'
        if len(segment['Coords']) >= 2:
            self.scatter.addPoints(
                pos=[segment['Coords'][:2]],
                data=segment.name,
                brush=pg.mkBrush(c),
                size=10
            )
            self.npoints += 1

    def add_spots_batch(self, segments_df, func_get_color=None):
        """Add multiple spots at once, much faster than calling add_spot repeatedly."""
        spots = []
        # Avoid .iterrows() which triggers pandas type inference
        seg_ids = list(segments_df.index)
        for seg_id in seg_ids:
            tags = segments_df.at[seg_id, 'Tags']
            coords = segments_df.at[seg_id, 'Coords']
            if func_get_color and len(tags) > 0:
                c = func_get_color(list(tags)[0])
            else:
                c = 'r'
            if coords is not None and len(coords) >= 2:
                spots.append({
                    'pos': coords[:2],
                    'data': seg_id,
                    'brush': pg.mkBrush(c),
                    'size': 10
                })

        if spots:
            self.scatter.addPoints(spots)
            self.npoints += len(spots)

    def update_spots(self, segments, func_get_color=None):
        spot_seg_IDs = [spot['data'] for spot in self.scatter.data]
        spot_brushes = [spot['brush'] for spot in self.scatter.data]
        # first add all the spots that need to be added
        segs_to_add = []
        any_changed = False
        # Avoid .iterrows() which triggers pandas type inference
        seg_ids = list(segments.index)
        for ix in seg_ids:
            coords = segments.at[ix, 'Coords']
            tags = segments.at[ix, 'Tags']
            if coords is not None and len(coords) >= 2:
                if ix not in spot_seg_IDs:
                    segs_to_add.append(segments.loc[ix])
                else:
                    if func_get_color and len(tags) > 0:
                        c = func_get_color(list(tags)[0])
                    else:
                        c = 'r'
                    spot_brushes[spot_seg_IDs.index(ix)] = pg.mkBrush(c)
                    any_changed = True
        if any_changed:
            self.scatter.setBrush(spot_brushes)
        # now add the ones that were not present
        for s_row in segs_to_add:
            self.add_spot(s_row, func_get_color)
        
class SegmentTableModel(QAbstractTableModel):
    """High-performance table model for segments using model-view architecture.

    Instead of creating QTableWidgetItem objects for every cell, this model
    directly references the segments DataFrame and provides data on-demand.
    Only visible rows are rendered, making it efficient for large datasets.
    """

    COLUMNS = ["SegID", "SourceName", "Start", "Stop", "Duration", "Tags"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._segments_df = pd.DataFrame()
        self._project = None
        self._sorted_indices = []  # Sorted row indices into _segments_df
        self._sort_column = 2  # Default sort by Start time
        self._sort_order = Qt.SortOrder.AscendingOrder

    def set_data(self, segments_df, project):
        """Replace all data in the model.

        Note: We store a reference to the DataFrame rather than copying it.
        The model is read-only so this is safe and avoids expensive copy overhead.
        """
        self.beginResetModel()
        self._segments_df = segments_df if len(segments_df) > 0 else pd.DataFrame()
        self._project = project
        self._rebuild_sorted_indices()
        self.endResetModel()

    def add_rows(self, new_segments_df):
        """Add new rows to the model."""
        if len(new_segments_df) == 0:
            return

        # Append to internal dataframe
        start_row = len(self._sorted_indices)
        self._segments_df = pd.concat([self._segments_df, new_segments_df])

        # Re-sort to maintain order
        self.beginResetModel()
        self._rebuild_sorted_indices()
        self.endResetModel()

    def update_rows(self, updated_segments_df):
        """Update existing rows in the model."""
        if len(updated_segments_df) == 0:
            return

        for seg_id in updated_segments_df.index:
            if seg_id in self._segments_df.index:
                self._segments_df.loc[seg_id] = updated_segments_df.loc[seg_id]

        # Re-sort and refresh
        self.beginResetModel()
        self._rebuild_sorted_indices()
        self.endResetModel()

    def remove_row_by_id(self, seg_id):
        """Remove a row by segment ID."""
        if seg_id in self._segments_df.index:
            self.beginResetModel()
            self._segments_df = self._segments_df.drop(seg_id)
            self._rebuild_sorted_indices()
            self.endResetModel()
            return True
        return False

    def remove_rows_by_ids(self, seg_ids):
        """Remove multiple rows by segment IDs in a single batch operation.

        This is much faster than calling remove_row_by_id() in a loop because
        it only does one model reset and one sorted indices rebuild.
        """
        # Filter to only IDs that exist
        ids_to_remove = [sid for sid in seg_ids if sid in self._segments_df.index]
        if not ids_to_remove:
            return 0

        self.beginResetModel()
        self._segments_df = self._segments_df.drop(ids_to_remove)
        self._rebuild_sorted_indices()
        self.endResetModel()
        return len(ids_to_remove)

    def _rebuild_sorted_indices(self):
        """Rebuild the sorted index mapping."""
        if len(self._segments_df) == 0:
            self._sorted_indices = []
            return

        # Get sort values based on column
        if self._sort_column == 2:  # Start time
            sort_values = [int(idx) for idx in self._segments_df['StartIndex'].values]
        elif self._sort_column == 3:  # Stop time
            sort_values = [int(idx) for idx in self._segments_df['StopIndex'].values]
        elif self._sort_column == 4:  # Duration
            sort_values = [int(stop) - int(start) for start, stop in
                          zip(self._segments_df['StartIndex'].values,
                              self._segments_df['StopIndex'].values)]
        elif self._sort_column == 1:  # Source name
            sort_values = [s.name for s in self._segments_df['Source'].values]
        elif self._sort_column == 5:  # Tags
            sort_values = [",".join(t) for t in self._segments_df['Tags'].values]
        else:  # SegID (column 0)
            sort_values = list(self._segments_df.index)

        # Create (value, original_index) pairs and sort
        indexed_values = list(enumerate(sort_values))
        reverse = self._sort_order == Qt.SortOrder.DescendingOrder
        indexed_values.sort(key=lambda x: x[1], reverse=reverse)
        self._sorted_indices = [idx for idx, _ in indexed_values]

    def rowCount(self, parent=QModelIndex()):
        return len(self._sorted_indices)

    def columnCount(self, parent=QModelIndex()):
        return len(self.COLUMNS)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self._project is None:
            return None

        if role != Qt.ItemDataRole.DisplayRole:
            return None

        # Map view row to dataframe row using sorted indices
        df_row_idx = self._sorted_indices[index.row()]
        seg_id = self._segments_df.index[df_row_idx]
        row_data = self._segments_df.iloc[df_row_idx]
        col = index.column()

        sr = self._project.sampling_rate

        if col == 0:  # SegID
            return str(seg_id)
        elif col == 1:  # SourceName
            return row_data['Source'].name
        elif col == 2:  # Start
            return hhmmss(int(row_data['StartIndex']) / sr, dec=3)
        elif col == 3:  # Stop
            return hhmmss(int(row_data['StopIndex']) / sr, dec=3)
        elif col == 4:  # Duration
            duration = (int(row_data['StopIndex']) - int(row_data['StartIndex'])) / sr
            return hhmmss(duration, dec=3)
        elif col == 5:  # Tags
            return ",".join(row_data['Tags'])

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.COLUMNS[section]
        return str(section + 1)

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        self._sort_column = column
        self._sort_order = order
        self.beginResetModel()
        self._rebuild_sorted_indices()
        self.endResetModel()

    def get_seg_id_for_row(self, view_row):
        """Get the segment ID for a view row index."""
        if 0 <= view_row < len(self._sorted_indices):
            df_row_idx = self._sorted_indices[view_row]
            return self._segments_df.index[df_row_idx]
        return None

    def get_row_for_seg_id(self, seg_id):
        """Get the view row index for a segment ID."""
        if seg_id not in self._segments_df.index:
            return None
        df_row_idx = self._segments_df.index.get_loc(seg_id)
        try:
            return self._sorted_indices.index(df_row_idx)
        except ValueError:
            return None


class SegmentPanel(widgets.QWidget):
    """Segment table panel using model-view architecture for performance.

    Uses QTableView with SegmentTableModel instead of QTableWidget to handle
    large datasets efficiently. The model only renders visible rows and doesn't
    create QTableWidgetItem objects for every cell.
    """

    contextMenuRequested = pyqtSignal(QPoint, object)
    segmentSelectionChanged = pyqtSignal(object)

    # TODO add a filtering dropdown / text box
    # TODO jump to time with click events
    def __init__(self, parent=None):
        super().__init__(parent)
        self._project = None
        self.init_ui()
        self.init_actions()

    def init_ui(self):
        layout = widgets.QVBoxLayout()

        # Use QTableView with custom model instead of QTableWidget
        self.model = SegmentTableModel(self)
        self.table = widgets.QTableView()
        self.table.setModel(self.model)

        self.table.setEditTriggers(widgets.QTableView.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        self.table.setSelectionBehavior(widgets.QTableView.SelectionBehavior.SelectRows)
        self.table.setColumnHidden(0, True)  # Hide SegID column
        self.table.setSortingEnabled(True)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, widgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, widgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, widgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, widgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, widgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, widgets.QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.table)
        self.setLayout(layout)

    def init_actions(self):
        self.table.selectionModel().selectionChanged.connect(self.on_click)

    def contextMenuEvent(self, event):
        pos = event.globalPos()
        self.contextMenuRequested.emit(pos, self.get_selection())

    def on_click(self):
        selection = self.get_selection()
        self.segmentSelectionChanged.emit(selection)

    def on_selection_changed(self, selection):
        # Temporarily disconnect to avoid feedback loop
        self.table.selectionModel().selectionChanged.disconnect(self.on_click)
        self.set_selection(selection)
        self.table.selectionModel().selectionChanged.connect(self.on_click)

    def set_selection(self, selection):
        self.table.clearSelection()
        selection_model = self.table.selectionModel()

        for seg_id in selection:
            view_row = self.model.get_row_for_seg_id(seg_id)
            if view_row is not None:
                index = self.model.index(view_row, 0)
                selection_model.select(
                    index,
                    QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
                )
            else:
                raise ValueError("Cannot select Segment ID {}: not found in table".format(seg_id))

    def get_selection(self):
        """Get list of selected segment IDs."""
        selection_model = self.table.selectionModel()
        selected_rows = set()
        for index in selection_model.selectedIndexes():
            selected_rows.add(index.row())

        ids = []
        for row in selected_rows:
            seg_id = self.model.get_seg_id_for_row(row)
            if seg_id is not None:
                ids.append(seg_id)
        return sorted(ids)

    def set_data(self, segments, project):
        """Replace all data in the table."""
        self._project = project
        self.model.set_data(segments, project)
        # Sort by start time (column 2)
        self.model.sort(2, Qt.SortOrder.AscendingOrder)

    def add_row(self, segment, project):
        """Add a single row. For bulk additions, use add_rows_batch."""
        self._project = project
        # Convert single Series to DataFrame with proper index
        segment_df = segment.to_frame().T
        self.model.add_rows(segment_df)

    def add_rows_batch(self, segments_df, project):
        """Add multiple rows at once - now very fast with model-view architecture."""
        if len(segments_df) == 0:
            return
        self._project = project
        self.model.add_rows(segments_df)

    def update_rows(self, segments, project):
        """Update existing rows."""
        self._project = project
        self.model.update_rows(segments)

    def remove_row_by_segID(self, seg_id):
        """Remove a row by segment ID."""
        if seg_id in self.get_selection():
            self.table.clearSelection()
        if not self.model.remove_row_by_id(seg_id):
            raise ValueError("Cannot remove Segment ID {}: not found in table".format(seg_id))

    def remove_rows_by_segIDs(self, seg_ids):
        """Remove multiple rows by segment IDs in a single batch operation."""
        # Clear selection if any of the deleted segments are selected
        current_selection = self.get_selection()
        if any(sid in current_selection for sid in seg_ids):
            self.table.clearSelection()
        return self.model.remove_rows_by_ids(seg_ids)

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

        # StartIndex/StopIndex are now raw integers, compute timestamps using sampling rate
        sr = plugin.api.project.sampling_rate
        start_time = self.segment.StartIndex / sr
        stop_time = self.segment.StopIndex / sr
        self.setToolTip("{}\n{:.2f}s to {:.2f}s\nDuration: {:.1f} ms\nTags: {}".format(
            self.segment.Source.name,
            start_time,
            stop_time,
            (stop_time - start_time) * 1000,
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

    def mouseClickEvent(self, event):
        #TODO there is probably a more elegant way to do this
        self.segment_plugin.api.set_segment_selection([self.segment.name])

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

        self._next_seg_id = 0

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
        self.panel.segmentSelectionChanged.connect(self.api.set_segment_selection)
        self.umap_panel.segmentSelectionChanged.connect(self.api.set_segment_selection)
        
        # and connect api event to this
        self.api.segmentSelectionChanged.connect(self.on_segment_selection_changed)

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

    def on_segment_selection_changed(self):
        selection = self.api.get_segment_selection()
        if self._selected_segments == selection:
            return
        if selection != []:
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
            #start.value = (start+stop) // 2
            new_start = self.api.create_stftindex(( start + stop - duration )//2)
            new_stop = self.api.create_stftindex(( start + stop + duration )//2)
            start = new_start
            stop = new_stop

        self._selected_segments = selection
        # call the UI Selection changes
        # TODO move these to api listeners
        self.panel.on_selection_changed(selection)
        self.umap_panel.on_selection_changed(selection)

        self.api.clear_selection()
        if selection != []:
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
                "Coords": [],
                "SegmentID": []
            }))
            return datastore["segments"]

    @_segmentation_datastore.setter
    def _segmentation_datastore(self, value):
        # check that the value is a pandas dataframe
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Segmentation datastore must be a pandas dataframe")
        # check that it has the requisite columns
        if not all([c in value.columns for c in ["Source", "StartIndex", "StopIndex", "Tags", "Coords", "SegmentID"]]):
            raise ValueError("Segmentation datastore must have columns SourceName, SourceChannel, StartIndex, StopIndex, Tags, Coords")
        self._datastore["segments"] = value

    def on_project_ready(self):
        """Called once"""
        # Check if we should load from NWB file
        if self.api.is_nwb_mode and self.api.nwb_path:
            self._load_from_nwb()
            return

        save_file = self.api.paths.save_dir / self.SAVE_FILENAME
        if not save_file.exists():
            return

        data = pd.read_csv(save_file, converters={"Tags": str, "Coords": str})

        if len(data) == 0:
            return

        # Base segment columns
        base_columns = {'Source', 'SourceName', 'SourceChannel', 'StartIndex', 'StopIndex', 'Tags', 'Coords', 'SegmentID'}
        # Identify feature columns (any column not in base_columns)
        feature_columns = [col for col in data.columns if col not in base_columns and not col.startswith('Unnamed')]

        # Build source lookup and cache
        existing_sources = {(s.name, s.channel): s for s in self.api.get_sources()}

        # Vectorized: get SegmentIDs
        if 'SegmentID' in data.columns:
            segment_ids = data['SegmentID'].tolist()
        else:
            segment_ids = list(range(len(data)))

        # Vectorized: parse Tags column
        def parse_tags(val):
            if val and isinstance(val, str):
                try:
                    return set(json.loads(val))
                except:
                    return set()
            return set()
        tags_list = [parse_tags(v) for v in data['Tags'].values]

        # Vectorized: parse Coords column
        def parse_coords(val):
            if val and isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    return [float(x) for x in parsed] if parsed else None
                except:
                    return None
            return None
        coords_list = [parse_coords(v) for v in data['Coords'].values]

        # Vectorized: build Source objects
        # First, create any missing sources
        if 'SourceName' in data.columns and 'SourceChannel' in data.columns:
            source_keys = list(zip(data['SourceName'].values, data['SourceChannel'].values))
        else:
            # Legacy format - map Source column to existing sources by index
            unique_sources = data['Source'].unique()
            source_mapping = {s: list(existing_sources.keys())[i] for i, s in enumerate(unique_sources)}
            source_keys = [source_mapping[s] for s in data['Source'].values]

        # Create any sources that don't exist
        unique_keys = set(source_keys)
        for key in unique_keys:
            if key not in existing_sources:
                self.api.create_source(key[0], key[1])
                existing_sources[key] = self.api.get_source(key[0], key[1])

        # Map source keys to Source objects
        sources_list = [existing_sources[k] for k in source_keys]

        # Store raw integers instead of ProjectIndex objects for faster loading
        # ProjectIndex will be created on-demand when needed via get_segment_as_project_index()
        start_indices = data['StartIndex'].astype(int).tolist()
        stop_indices = data['StopIndex'].astype(int).tolist()

        # Build the DataFrame directly without row-by-row iteration
        # Use plain lists (not Series) to avoid slow index alignment
        seg_df = {
            'Source': sources_list,
            'StartIndex': start_indices,
            'StopIndex': stop_indices,
            'Tags': tags_list,
            'Coords': coords_list,
            'SegmentID': segment_ids
        }

        # Add feature columns directly from the source DataFrame
        for feat_col in feature_columns:
            seg_df[feat_col] = data[feat_col].tolist()

        # Create DataFrame from dict of lists (faster than dict of Series)
        # Then set index once at the end to avoid alignment overhead
        df = pd.DataFrame(seg_df)
        df.index = segment_ids
        self._segmentation_datastore = df
        # Store the max of the segmentIDs so we can increment
        self._next_seg_id = max(self._segmentation_datastore.index)+1
        #self._segmentation_datastore.sort()

    def on_project_data_loaded(self):
        self.panel.set_data(self._segmentation_datastore,self.api.project)
        self.umap_panel.set_data(self._segmentation_datastore, self.api.plugins["TagPlugin"].get_tag_color)
        self.refresh()

    def needs_saving(self):
        return self._needs_saving

    def _load_from_nwb(self):
        """Load segments from NWB file's intervals group."""
        nwb_path = str(self.api.nwb_path)
        sampling_rate = self.api.project.sampling_rate

        if not NWBFile.has_soundsep_segments(nwb_path):
            return

        try:
            segment_data = NWBFile.read_soundsep_segments(nwb_path, sampling_rate)
        except Exception as e:
            logger.warning(f"Could not load segments from NWB file: {e}")
            return

        if not segment_data:
            return

        seg_df = {
            "Source": [],
            "StartIndex": [],
            "StopIndex": [],
            "Tags": [],
            "Coords": [],
            "SegmentID": []
        }

        source_lookup = set([
            (source.name, source.channel) for source in self.api.get_sources()
        ])

        for row in segment_data:
            source_key = (row["SourceName"], row["SourceChannel"])
            if source_key not in source_lookup:
                source_lookup.add(source_key)
                self.api.create_source(source_key[0], source_key[1])
            source = self.api.get_source(source_key[0], source_key[1])

            seg_df['Source'].append(source)
            # Store raw integers, not ProjectIndex objects
            seg_df['StartIndex'].append(int(row["StartIndex"]))
            seg_df['StopIndex'].append(int(row["StopIndex"]))

            # Parse tags from JSON string
            if row["Tags"]:
                try:
                    seg_df['Tags'].append(set(json.loads(row["Tags"])))
                except:
                    seg_df['Tags'].append(set())
            else:
                seg_df['Tags'].append(set())

            # Parse coords from JSON string
            if row["Coords"]:
                try:
                    coords = json.loads(row["Coords"])
                    seg_df['Coords'].append(list(coords) if coords else None)
                except:
                    seg_df['Coords'].append(None)
            else:
                seg_df['Coords'].append(None)

            seg_df['SegmentID'].append(row['SegmentID'])

        # StartIndex/StopIndex are now raw integers, no need for object dtype

        self._segmentation_datastore = pd.DataFrame(seg_df, index=seg_df['SegmentID'])
        if len(self._segmentation_datastore) > 0:
            self._next_seg_id = max(self._segmentation_datastore.index) + 1

    def save(self):
        """Save pointers within project"""
        # TODO: these pointers could get out of sync with a project if/when files are added.
        # Can we recover from this? or should we hash the project so we can at least
        # warn the user when things dont match up to when the file was saved?

        # Base segment columns that need special handling
        base_columns = {'Source', 'StartIndex', 'StopIndex', 'Tags', 'Coords', 'SegmentID'}

        # Prepare segment data
        segment_data = []
        for idx in self._segmentation_datastore.index:
            row = self._segmentation_datastore.loc[idx]
            row_data = {
                'SourceName': row['Source'].name,
                'SourceChannel': row['Source'].channel,
                'StartIndex': int(row['StartIndex']),
                'StopIndex': int(row['StopIndex']),
                'Tags': json.dumps(list(row['Tags'])),
                'Coords': json.dumps(row['Coords']),
                'SegmentID': idx
            }
            # Include all additional columns (features, PCA, UMAP, etc.)
            for col in self._segmentation_datastore.columns:
                if col not in base_columns:
                    row_data[col] = row[col]
            segment_data.append(row_data)

        # Save to NWB if in NWB mode
        if self.api.is_nwb_mode and self.api.nwb_path:
            nwb_path = str(self.api.nwb_path)
            sampling_rate = self.api.project.sampling_rate

            # Close files before writing (they'll reopen lazily)
            self.api.project.close_files()

            try:
                NWBFile.write_soundsep_segments(nwb_path, segment_data, sampling_rate)
                logger.info(f"Saved {len(segment_data)} segments to NWB file")
            except Exception as e:
                logger.error(f"Could not save segments to NWB file: {e}")
                raise
        else:
            # Save to CSV file - build DataFrame from segment_data
            if segment_data:
                out_csv_df = pd.DataFrame(segment_data)
            else:
                out_csv_df = pd.DataFrame(columns=['SourceName', 'SourceChannel', 'StartIndex', 'StopIndex', 'Tags', 'Coords', 'SegmentID'])
            out_csv_df.to_csv(self.api.paths.save_dir / self.SAVE_FILENAME, index=False)

        self._needs_saving = False

    def on_sources_changed(self):
        # We need to check if any sources have been deleted and remove their segments
        valid_sources = self.api.get_sources()
        invalid_mask = ~self._segmentation_datastore['Source'].isin(valid_sources)
        invalid_seg_ids = self._segmentation_datastore[invalid_mask].index.tolist()

        if invalid_seg_ids:
            # Remove from datastore
            self._segmentation_datastore = self._segmentation_datastore[~invalid_mask]
            # Use batch operations for efficiency
            self.panel.remove_rows_by_segIDs(invalid_seg_ids)
            self.api.segments_deleted(invalid_seg_ids)

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
        # Convert to int since datastore stores raw integers
        ws0_int, ws1_int = int(ws0), int(ws1)

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
        segs_in_view = self._segmentation_datastore[ (self._segmentation_datastore['StartIndex'] < ws1_int) & (self._segmentation_datastore['StopIndex'] > ws0_int) ]
        selection = self.api.get_fine_selection()
        
        # Go through each source view and draw the segments
        for source in self.api.get_sources():
            source_view = self.gui.source_views[source.index]
            source_segs = segs_in_view[ segs_in_view['Source'] == source ]
            # Avoid .iterrows() which triggers pandas type inference
            seg_ids = list(source_segs.index)
            for idx in seg_ids:
                segment_row = source_segs.loc[idx]
                tags = segment_row["Tags"]
                # get the color of the first tag TODO maybe make this different than tags
                if len(tags) == 0:
                    c = "#00ff00"
                else:
                    t = list(tags)[0]
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
                selection.source,
                tags=set(),
                coords=list()
            )

    def create_segments_batch(
        self,
        segment_data: List[Tuple[ProjectIndex, ProjectIndex, Source]],
        skip_delete_check: bool = False
    ):
        """Create multiple segments efficiently using batch operations.

        Arguments
        ---------
        segment_data : List[Tuple[ProjectIndex, ProjectIndex, Source]]
            List of (start, stop, source) tuples for each segment to create
        skip_delete_check : bool
            If True, skip checking for overlapping segments to delete.
            Use this when you've already deleted segments in the range (e.g., from AutoSegmentPlugin).
        """
        if not segment_data:
            return

        # Build all segment data at once
        new_rows = []
        start_seg_id = self._next_seg_id

        for start, stop, source in segment_data:
            new_rows.append({
                'StartIndex': int(start),  # Store raw integer, not ProjectIndex
                'StopIndex': int(stop),    # Store raw integer, not ProjectIndex
                'Source': source,
                'Tags': set(),
                'Coords': list()
            })

        # Update next ID
        self._next_seg_id = start_seg_id + len(segment_data)

        # Create empty DataFrame first, then assign values to avoid pandas type inference
        # (ProjectIndex objects can't be compared with ints during type inference)
        new_df = pd.DataFrame(
            index=range(start_seg_id, self._next_seg_id),
            columns=['StartIndex', 'StopIndex', 'Source', 'Tags', 'Coords']
        )
        for i, row_data in enumerate(new_rows):
            seg_id = start_seg_id + i
            new_df.at[seg_id, 'StartIndex'] = row_data['StartIndex']
            new_df.at[seg_id, 'StopIndex'] = row_data['StopIndex']
            new_df.at[seg_id, 'Source'] = row_data['Source']
            new_df.at[seg_id, 'Tags'] = row_data['Tags']
            new_df.at[seg_id, 'Coords'] = row_data['Coords']

        # Optionally check for overlaps (slow, so skip when safe)
        if not skip_delete_check:
            for start, stop, source in segment_data:
                self.delete_segments_between(start, stop, source, refresh=False)

        # Concatenate with existing datastore
        self._segmentation_datastore = pd.concat([self._segmentation_datastore, new_df])

        # Emit batch signal for other plugins (e.g., FeaturePlugin)
        self.api.segments_created(list(new_df.index))

        # Batch update UI
        self.panel.add_rows_batch(new_df, self.api.project)
        self.umap_panel.add_spots_batch(new_df, self.api.plugins["TagPlugin"].get_tag_color)

        self.gui.show_status(f"Created {len(segment_data)} segments")
        logger.debug(f"Created {len(segment_data)} segments in batch")
        self._needs_saving = True
        self.refresh()


    def create_segment(self, start: ProjectIndex, stop: ProjectIndex, source: Source, tags: set = set(), coords: list = list()):
        self.delete_segments_between(start, stop, source)

        segID = self._next_seg_id
        assert(segID not in self._segmentation_datastore.index)
        self._next_seg_id += 1
        self._segmentation_datastore.loc[segID] = pd.Series()
        self._segmentation_datastore.at[segID,'StartIndex'] = int(start)  # Store raw integer
        self._segmentation_datastore.at[segID,'StopIndex'] = int(stop)    # Store raw integer
        self._segmentation_datastore.at[segID,'Source'] = source
        self._segmentation_datastore.at[segID,'Tags'] = tags
        self._segmentation_datastore.at[segID,'Coords'] = coords
        
        
        
        # Notify other plugins (using batch signal with single-element list)
        self.api.segments_created([segID])
        self.panel.add_row(self._segmentation_datastore.loc[segID], self.api.project)
        self.umap_panel.add_spot(self._segmentation_datastore.loc[segID], self.api.plugins["TagPlugin"].get_tag_color)
        
        self.gui.show_status("Created segment {} to {}".format(start, stop))
        logger.debug("Created segment {} to {}".format(start, stop))
        self._needs_saving = True
        self.refresh()



    def delete_segments_between(self, start: ProjectIndex, stop: ProjectIndex, source: Source, refresh: bool = True):
        # Delete all segments from this source who have a start OR stop index within the range
        # Convert ProjectIndex to int since datastore stores raw integers
        start_int, stop_int = int(start), int(stop)
        segs_to_delete = ((self._segmentation_datastore['StopIndex'].between(start_int, stop_int) |\
                                self._segmentation_datastore['StartIndex'].between(start_int, stop_int)) &\
                            (self._segmentation_datastore['Source'] == source))
        
        deleted_inds = self._segmentation_datastore[segs_to_delete].index
        if len(deleted_inds) == 0:
            return
        self.delete_segments(deleted_inds)

    def delete_segments(self, seg_ids, refresh: bool = True):
        # Delete all segments from this source who have a start OR stop index within the range
        n_deleted = len(seg_ids)
        self._segmentation_datastore.drop(seg_ids, inplace=True)
        self.gui.show_status("Deleting {} segments".format(n_deleted))
        logger.debug("Deleting {} segments".format(n_deleted))

        # Use batch operations for efficiency - single model reset instead of N resets
        seg_ids_list = list(seg_ids)  # Ensure it's a list for batch operations
        self.panel.remove_rows_by_segIDs(seg_ids_list)
        self.api.segments_deleted(seg_ids_list)  # Batch signal for other plugins
        self.umap_panel.remove_spots(seg_ids_list)
        self._needs_saving = True
        if refresh:
            self.refresh()

    def merge_segments(self, start: ProjectIndex, stop: ProjectIndex, source: Source):
        # Merge all segments from this source who have a start OR stop index within the range
        # Convert ProjectIndex to int since datastore stores raw integers
        start_int, stop_int = int(start), int(stop)
        segs_to_merge = self._segmentation_datastore[((self._segmentation_datastore['StopIndex'].between(start_int, stop_int) |\
                            self._segmentation_datastore['StartIndex'].between(start_int, stop_int)) &\
                            (self._segmentation_datastore['Source'] == source))]

        if not len(segs_to_merge):
            return

        self.gui.show_status("Merging {} segments from {} to {}".format(len(segs_to_merge), start, stop))
        logger.debug("Merging {} segments from {} to {}".format(len(segs_to_merge), start, stop))
        new_tags = set.union(*list(segs_to_merge['Tags'].values))
        # StartIndex/StopIndex are now raw integers, convert to ProjectIndex for create_segment
        new_start = self.api.make_project_index(min(segs_to_merge['StartIndex']))
        new_stop = self.api.make_project_index(max(segs_to_merge['StopIndex']))
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
