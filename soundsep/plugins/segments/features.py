import bisect
import logging
import json
import time
import traceback
import os
from multiprocessing import Queue, Process, Event
import threading
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple

from soundsig.sound import BioSound
import soundsig.sound as sound
import warnings
from scipy.signal import firwin, filtfilt



import PyQt6.QtWidgets as widgets
import pyqtgraph as pg
import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QPoint, QThread, pyqtSignal, pyqtSlot, QObject, QAbstractTableModel, QModelIndex, QItemSelectionModel
from PyQt6 import QtGui

import soundfile
import umap
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA

from soundsep.core.base_plugin import BasePlugin
from soundsep.core.models import Source, ProjectIndex, StftIndex
from soundsep.core.segments import Segment
from soundsep.core.utils import hhmmss

logger = logging.getLogger(__name__)

# TODO move to core or something
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
class WorkerSignals(QObject):
    done_adding = pyqtSignal()
    finished = pyqtSignal()
    progress = pyqtSignal(int)

@dataclass
class SegmentLoadInfo:
    """Information needed to load a segment's audio independently."""
    seg_id: int
    file_path: str
    file_type: str  # 'wav', 'nwb', 'dat'
    start_index: int  # Index within the file
    stop_index: int
    channel: int  # Channel within the file
    sampling_rate: int
    lowpass: int = 6000
    highpass: int = 200

@dataclass
class SegmentFeatureResult:
    """Result from processing a single segment."""
    segmentID: int
    features: dict  # Detected intervals relative to block start
    error: Optional[str] = None

@dataclass
class BlockSegInfo:
    """Info for a single segment within a block."""
    seg_id: int
    local_start: int   # Start index relative to block
    local_stop: int    # Stop index relative to block
    channel: int       # Channel index within the block

@dataclass
class BlockFeatureInfo:
    """Info needed to extract features from all segments in a block."""
    block_index: int
    block: object      # The Block object (handles WAV, NWB, DAT internally)
    sampling_rate: int
    segments: List[BlockSegInfo]
    lowpass: int = 6000
    highpass: int = 200

class VisualizationPanel(widgets.QWidget):
    segmentSelectionChanged = pyqtSignal(object)
    def __init__(self, parent=None, api=None):
        super().__init__(parent)
        self.api=api
        self.init_ui()
        self.init_actions()
        self.npoints = 0
        self._func_get_color = None  # Store color function for tag-based coloring

    def init_ui(self):
        # setup a 2d plot
        layout = widgets.QVBoxLayout()
        # Add two drop down menus for x and y axis features
        self.x_axis = widgets.QComboBox()
        self.y_axis = widgets.QComboBox()
        x_y_layout = widgets.QHBoxLayout()
        x_y_layout.addWidget(widgets.QLabel("X-Axis:"))
        x_y_layout.addWidget(self.x_axis)
        x_y_layout.addWidget(widgets.QLabel("Y-Axis:"))
        x_y_layout.addWidget(self.y_axis)
        layout.addLayout(x_y_layout)
        # Add the plot
        self.plot = pg.plot()
        self.scatter = pg.ScatterPlotItem()
        self.plot.addItem(self.scatter)
        #layout = widgets.QGridLayout()
        layout.addWidget(self.plot)
        self.setLayout(layout)

    def add_features_to_dropdown(self, features):
        self.x_axis.addItems(features)
        self.y_axis.addItems(features)
    
    def init_actions(self):
        self.x_axis.currentIndexChanged.connect(self.on_x_axis_change)
        self.y_axis.currentIndexChanged.connect(self.on_y_axis_change)
        self.scatter.sigClicked.connect(self.on_click)
        return
    
    def on_click(self, plot, points):
        if len(points) > 0:
            # TODO what to do for multiselect
            self.segmentSelectionChanged.emit([points[0].data()])
        return

    def on_x_axis_change(self, ind):
        self.update_spots()

    def on_y_axis_change(self, ind):
        self.update_spots()

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
    
    def set_data(self, features, func_get_color=None):
        # Store color function for later use in update_spots
        if func_get_color is not None:
            self._func_get_color = func_get_color

        spots = []
        seg_db = self.api.get_mut_datastore().get('segments')
        for ix, feat_row in features.iterrows():
            # Get segment info (tags, coords) from segment datastore
            if seg_db is None or ix not in seg_db.index:
                continue
            seg_row = seg_db.loc[ix]
            tags = self.api.get_plugin('SegmentPlugin').get_tags_for_segment(ix)
            if self._func_get_color and len(tags) > 0:
                c = self._func_get_color(list(tags)[0])
            else:
                c = 'r'
            coords = seg_row.get('Coords') if hasattr(seg_row, 'get') else seg_row['Coords'] if 'Coords' in seg_db.columns else None
            if coords is not None and len(coords) >= 2:
                spots.append(dict({
                    'pos': coords[:2],
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
    
    def remove_spots(self, segIDs):
        visibilities = self.scatter.data['visible']
        spot_seg_IDs = [spot['data'] for spot in self.scatter.data]
        for segID in segIDs:
            if segID in spot_seg_IDs:
                visibilities[spot_seg_IDs.index(segID)] = False
                #spot_inds.append(seg_IDs.index(segID))
        self.scatter.setPointsVisible(visibilities)

    def add_spot(self, segID, coords, color='r'):
        self.scatter.addPoints(
            pos=[coords],
            data=segID,
            brush=pg.mkBrush(color),
            size=10
        )
        self.npoints += 1

    def update_spots(self, func_get_color=None):
        # Update stored color function if provided
        if func_get_color is not None:
            self._func_get_color = func_get_color

        spot_seg_IDs = self.scatter.data['data']
        mut_ds = self.api.get_mut_datastore()
        seg_db = mut_ds.get('segments')
        if seg_db is None:
            return

        # Features are now columns in the segments datastore
        x_axis = self.x_axis.currentText()
        y_axis = self.y_axis.currentText()

        if x_axis == "" or y_axis == "":
            return

        if x_axis not in seg_db.columns or y_axis not in seg_db.columns:
            return

        data = self.scatter.data
        if spot_seg_IDs.size > 0:
            data['x'] = seg_db[x_axis].loc[spot_seg_IDs]
            data['y'] = seg_db[y_axis].loc[spot_seg_IDs]

            # Update colors based on tags
            if self._func_get_color is not None:
                new_brushes = []
                for seg_id in spot_seg_IDs:
                    tags = self.api.get_plugin('SegmentPlugin').get_tags_for_segment(seg_id)
                    if len(tags) > 0:
                        c = self._func_get_color(list(tags)[0])
                    else:
                        c = 'r'
                    new_brushes.append(pg.mkBrush(c))
                self.scatter.setBrush(new_brushes)

            self.scatter.updateSpots()
            vb = self.scatter.getViewBox()
            xrange = self.scatter.dataBounds(0)
            if xrange[0] is not None and xrange[1] is not None:
                vb.setXRange(xrange[0], xrange[1])
            yrange = self.scatter.dataBounds(1)
            if yrange[0] is not None and yrange[1] is not None:
                vb.setYRange(yrange[0], yrange[1])

        segs_to_add = []
        segments_not_present = seg_db[~seg_db.index.isin(spot_seg_IDs)]
        for ix in segments_not_present.index:
            x_val = seg_db.at[ix, x_axis]
            y_val = seg_db.at[ix, y_axis]
            if not pd.isna(x_val) and not pd.isna(y_val):
                # Get color based on tag
                tags = self.api.get_plugin('SegmentPlugin').get_tags_for_segment(ix)
                if self._func_get_color and len(tags) > 0:
                    c = self._func_get_color(list(tags)[0])
                else:
                    c = 'r'
                segs_to_add.append((ix, [x_val, y_val], c))

        # now add the ones that were not present
        for seg_id, coords, color in segs_to_add:
            self.add_spot(seg_id, coords, color)

class DimensionalityReductionWizard(widgets.QWidget):
    """ Window for selecting features to include in PCA"""
    feature_generation_signal = pyqtSignal(object, str, str)
    def __init__(self,parent=None, features=None, feature_percents=None, feat_check_callback=None):
        super().__init__(parent)
        self.features = features
        self.feat_check_callback = feat_check_callback
        self.feature_percents = feature_percents
        self.init_ui()
    
    def init_ui(self):
        layout = widgets.QVBoxLayout()
        # Add checkboxes for each feature class
        self.feature_checkboxes = {}
        checkbox_layout = widgets.QHBoxLayout()
        sub_layouts = {}
        for k in self.features.keys():
            sub_layouts[k] = widgets.QVBoxLayout()
            sub_layouts[k].addWidget(widgets.QLabel(k))
            for ix,kk in enumerate(self.features[k]):
                self.feature_checkboxes[kk] = widgets.QCheckBox(kk)
                sub_layouts[k].addWidget(self.feature_checkboxes[kk])
                pcen = self.feature_percents[k][ix]
                sub_layouts[k].addWidget(widgets.QLabel("NaNs: %.2f" % (pcen*100)))
                self.feature_checkboxes[kk].setChecked(bool(pcen < .1))
                self.feature_checkboxes[kk].stateChanged.connect(self.on_box_checked)
            checkbox_layout.addLayout(sub_layouts[k])
        layout.addLayout(checkbox_layout)

        # add a label for total number of stims
        n_good, n_total = self.feat_check_callback(self.get_selected_features())
        self.n_stims_label = widgets.QLabel("Number of stims: %d/%d" % (n_good, n_total))
        layout.addWidget(self.n_stims_label)

        # add a drop down for type of dimensionality reduction
        self.dim_reduction_type = widgets.QComboBox()
        self.dim_reduction_type.addItems(["PCA", "UMAP"])
        layout.addWidget(self.dim_reduction_type)

        # add a clear all boxes button
        self.clear_all_button = widgets.QPushButton("Uncheck All")
        self.clear_all_button.clicked.connect(self.clear_checkboxes)
        layout.addWidget(self.clear_all_button)

        # TODO can add some params here

        # Add a generate button with a field for the name of the new feature
        generation_layout = widgets.QHBoxLayout()
        self.feat_name = widgets.QLineEdit("DimRed1")
        generation_layout.addWidget(widgets.QLabel("Feature Name:"))
        generation_layout.addWidget(self.feat_name)
        self.generate_button = widgets.QPushButton("Generate")
        self.generate_button.clicked.connect(self.on_generate_button_press)
        generation_layout.addWidget(self.generate_button)
        layout.addLayout(generation_layout)
        self.setLayout(layout)

    def on_box_checked(self, state):
        # check how many stims are good for these features
        if self.feat_check_callback:
            n_good, n_total = self.feat_check_callback(self.get_selected_features())
            self.n_stims_label.setText("Number of stims: %d/%d" % (n_good, n_total))
    def clear_checkboxes(self):
        for k,v in self.feature_checkboxes.items():
            v.setChecked(False)
    def get_selected_features(self):
        selected_features = []
        for k,v in self.feature_checkboxes.items():
            if v.isChecked():
                selected_features.append(k)
        return selected_features

    def on_generate_button_press(self):
        selected_features = self.get_selected_features()
        dim_reduction_type = self.dim_reduction_type.currentText()
        new_feature_name = self.feat_name.text()
        self.feature_generation_signal.emit(selected_features, dim_reduction_type, new_feature_name)


class FeatureTableModel(QAbstractTableModel):
    """High-performance table model for features using model-view architecture.

    Instead of creating QTableWidgetItem objects for every cell, this model
    directly references the feature DataFrame and provides data on-demand.
    Only visible rows are rendered, making it efficient for large datasets.
    """

    def __init__(self, features_list, parent=None):
        super().__init__(parent)
        self._feature_df = pd.DataFrame()
        self._features_list = features_list  # List of feature column names
        self._sorted_indices = []  # Sorted row indices into _feature_df
        self._sort_column = 0
        self._sort_order = Qt.SortOrder.AscendingOrder

    def set_data(self, feature_df):
        """Replace all data in the model."""
        self.beginResetModel()
        self._feature_df = feature_df if len(feature_df) > 0 else pd.DataFrame()
        self._rebuild_sorted_indices()
        self.endResetModel()

    def add_rows(self, new_feature_df):
        """Add new rows to the model."""
        if len(new_feature_df) == 0:
            return
        self._feature_df = pd.concat([self._feature_df, new_feature_df])
        self.beginResetModel()
        self._rebuild_sorted_indices()
        self.endResetModel()

    def update_row(self, seg_id, feature_row):
        """Update an existing row in the model."""
        if seg_id in self._feature_df.index:
            self._feature_df.loc[seg_id] = feature_row
            # Find the view row and emit dataChanged
            view_row = self.get_row_for_seg_id(seg_id)
            if view_row is not None:
                top_left = self.index(view_row, 0)
                bottom_right = self.index(view_row, self.columnCount() - 1)
                self.dataChanged.emit(top_left, bottom_right)

    def remove_row_by_id(self, seg_id):
        """Remove a row by segment ID."""
        if seg_id in self._feature_df.index:
            self.beginResetModel()
            self._feature_df = self._feature_df.drop(seg_id)
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
        ids_to_remove = [sid for sid in seg_ids if sid in self._feature_df.index]
        if not ids_to_remove:
            return 0

        self.beginResetModel()
        self._feature_df = self._feature_df.drop(ids_to_remove)
        self._rebuild_sorted_indices()
        self.endResetModel()
        return len(ids_to_remove)

    def add_feature_column(self, feature_name):
        """Add a new feature column."""
        if feature_name not in self._features_list:
            self._features_list.append(feature_name)
            if len(self._feature_df) > 0 and feature_name not in self._feature_df.columns:
                self._feature_df[feature_name] = np.nan
            self.beginResetModel()
            self.endResetModel()

    def _rebuild_sorted_indices(self):
        """Rebuild the sorted index mapping."""
        if len(self._feature_df) == 0:
            self._sorted_indices = []
            return

        # Sort by the current sort column
        if self._sort_column == 0:  # SegID
            sort_values = list(self._feature_df.index)
        elif self._sort_column - 1 < len(self._features_list):
            feat_name = self._features_list[self._sort_column - 1]
            if feat_name in self._feature_df.columns:
                sort_values = self._feature_df[feat_name].fillna(float('inf')).tolist()
            else:
                sort_values = list(range(len(self._feature_df)))
        else:
            sort_values = list(range(len(self._feature_df)))

        indexed_values = list(enumerate(sort_values))
        reverse = self._sort_order == Qt.SortOrder.DescendingOrder
        indexed_values.sort(key=lambda x: x[1], reverse=reverse)
        self._sorted_indices = [idx for idx, _ in indexed_values]

    def rowCount(self, parent=QModelIndex()):
        return len(self._sorted_indices)

    def columnCount(self, parent=QModelIndex()):
        return 1 + len(self._features_list)  # SegID + features

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        if role != Qt.ItemDataRole.DisplayRole:
            return None

        df_row_idx = self._sorted_indices[index.row()]
        col = index.column()

        if col == 0:  # SegID
            return str(self._feature_df.index[df_row_idx])
        else:
            feat_idx = col - 1
            if feat_idx < len(self._features_list):
                feat_name = self._features_list[feat_idx]
                if feat_name in self._feature_df.columns:
                    val = self._feature_df.iloc[df_row_idx][feat_name]
                    if pd.isna(val):
                        return ""
                    return "%.2f" % val
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if section == 0:
                return "segID"
            elif section - 1 < len(self._features_list):
                return self._features_list[section - 1]
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
            return self._feature_df.index[df_row_idx]
        return None

    def get_row_for_seg_id(self, seg_id):
        """Get the view row index for a segment ID."""
        if seg_id not in self._feature_df.index:
            return None
        df_row_idx = self._feature_df.index.get_loc(seg_id)
        try:
            return self._sorted_indices.index(df_row_idx)
        except ValueError:
            return None

    def set_column_hidden_state(self, feature_name, hidden):
        """Track hidden state for features (used by parent widget)."""
        pass  # Column visibility is handled by the view, not the model


class FeatureGenerationPanel(widgets.QWidget):
    segmentSelectionChanged = pyqtSignal(object)
    def __init__(self, parent=None,features=None):
        super().__init__(parent)

        self.FEATUREDICT=features
        self.indiv_features = []
        if features is not None:
            for k,v in features.items():
                self.indiv_features.extend(v)
        self.init_ui()

    
    def init_ui(self):
        layout = widgets.QVBoxLayout()
        # add a generate button
        self.generate_button = widgets.QPushButton("Generate")
        layout.addWidget(self.generate_button)
        self.generate_button.clicked.connect(self.on_generate_button_press)

        # Add checkboxes for each feature class
        self.feature_checkboxes = {}
        checkbox_layout = widgets.QHBoxLayout()
        for k in self.FEATUREDICT.keys():
            self.feature_checkboxes[k] = widgets.QCheckBox(k)
            self.feature_checkboxes[k].setChecked(True)
            checkbox_layout.addWidget(self.feature_checkboxes[k])
        layout.addLayout(checkbox_layout)

        # Use QTableView with custom model for performance
        self.feature_to_column = dict(zip(self.indiv_features, range(1, len(self.indiv_features)+1)))
        self.model = FeatureTableModel(self.indiv_features.copy(), self)
        self.table = widgets.QTableView()
        self.table.setModel(self.model)

        self.table.setEditTriggers(widgets.QTableView.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        self.table.setSelectionBehavior(widgets.QTableView.SelectionBehavior.SelectRows)
        self.table.setColumnHidden(0, True)  # Hide SegID column
        self.table.setSortingEnabled(True)

        header = self.table.horizontalHeader()
        for i in range(1, len(self.indiv_features)+1):
            header.setSectionResizeMode(i, widgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        # Dim reduction generation button
        self.generate_DR_button = widgets.QPushButton("Generate Dimensionality Reduction")
        layout.addWidget(self.generate_DR_button)

        self.setLayout(layout)

        # init actions
        self.table.selectionModel().selectionChanged.connect(self.on_click)

    
    def set_feature_selection(self, feature_selections):
        for k,v in feature_selections.items():
            if self.feature_checkboxes[k].isChecked() != v:
                self.feature_checkboxes[k].setChecked(v)
            for feature in self.FEATUREDICT[k]:
                if feature in self.feature_to_column:
                    self.table.setColumnHidden(self.feature_to_column[feature], not v)

    def on_generate_button_press(self):
        # Add a progress bar into the qvboxlayout
        self.progress = widgets.QProgressBar(self)
        self.progress.setGeometry(200, 80, 250, 20)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.layout().insertWidget(1, self.progress)

    def on_progress_signal(self,progress):
        self.progress.setValue(progress)
    
    def on_finished_signal(self):
        self.progress.deleteLater()

    def add_feature(self, feature_name, df_feature_data):
        """Add a new feature column to the table."""
        if feature_name not in self.indiv_features:
            self.indiv_features.append(feature_name)
            self.feature_to_column[feature_name] = len(self.indiv_features)
        self.model.add_feature_column(feature_name)
        # Update the model's internal dataframe with the new feature data
        if len(df_feature_data) > 0:
            for seg_id in df_feature_data.index:
                if seg_id in self.model._feature_df.index:
                    self.model._feature_df.at[seg_id, feature_name] = df_feature_data.at[seg_id]
            self.model.beginResetModel()
            self.model.endResetModel()
        # Hide the new column by default
        self.table.setColumnHidden(self.feature_to_column[feature_name], True)

    def add_or_edit_row(self, feature_row):
        """Add or update a row in the model."""
        seg_id = feature_row.name
        if seg_id in self.model._feature_df.index:
            self.model.update_row(seg_id, feature_row)
        else:
            self.add_row(feature_row)

    def add_row(self, feature_row):
        """Add a single row to the model."""
        segment_df = feature_row.to_frame().T
        self.model.add_rows(segment_df)

    def add_rows_batch(self, feature_df):
        """Add multiple rows at once - efficient batch operation."""
        if len(feature_df) == 0:
            return
        self.model.add_rows(feature_df)

    # Selection functions
    def on_click(self):
        selection = self.get_selection()
        self.segmentSelectionChanged.emit(selection)

    def on_selection_changed(self, selection):
        self.table.selectionModel().selectionChanged.disconnect(self.on_click)
        self.set_selection(selection)
        self.table.selectionModel().selectionChanged.connect(self.on_click)

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

    def set_selection(self, selection):
        """Set the selection by segment IDs."""
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

    def set_data(self, feature_db):
        """Replace all data in the table - now very fast with model-view architecture."""
        self.model.set_data(feature_db)
        # The model handles everything, no row-by-row iteration needed
        return

class FeaturePlugin(BasePlugin):

    # Features are now stored directly in the segments datastore (no separate file)
    FEATUREDICT = dict({
        "Amplitude":["rms","meantime","stdT","skewT","kurtT","entT","maxAmp"],
        "Fundamental":["fund","sal","fund2","sal2","maxfund","minfund","cvfund","cvfund2","devfund"],
        "Formant":["F1","F2","F3"],
        "Spectrum": ["meanS","stdS","skewS","kurtS","entS","q1","q2","q3"]
    })
    @property
    def featurelist(self):
        features = self.FEATUREDICT
        indiv_features = []
        for k,v in features.items():
            indiv_features.extend(v)
        return indiv_features

    def current_featurelist(self):
        feature_selections = self.feature_selections
        features = self.FEATUREDICT
        indiv_features = []
        for k,v in features.items():
            if feature_selections[k]:
                indiv_features.extend(v)
        return indiv_features

    def get_custom_features(self):
        # Compare against ALL standard features, not just currently selected ones
        feature_list = self.featurelist
        # Base segment columns that are not features
        base_segment_columns = {'Source', 'StartIndex', 'StopIndex', 'Tags', 'Coords', 'SegmentID'}
        # Get custom features from the segments datastore
        seg_db = self._segments_datastore
        if seg_db is None:
            return []
        custom_features = []
        for col in seg_db.columns:
            if col not in feature_list and col not in base_segment_columns:
                custom_features.append(col)
        return custom_features
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.worker_signals = WorkerSignals()
        self.config = dict()
        self.reset_config()
        self.panel = FeatureGenerationPanel(features=self.FEATUREDICT)
        self.vis_panel = VisualizationPanel(api=self.api)
        self.feature_selections = { k: True for k in self.FEATUREDICT.keys()}

        self.connect_events()

        self._needs_saving = False

    def feature_selection_changed(self, feature_class, state):
        self.feature_selections[feature_class] = state == 2
        self.panel.set_feature_selection(self.feature_selections)

    def connect_events(self):
        # TODO Connect panel events
        self.panel.generate_button.clicked.connect(self.launch_feature_generation_async)
        for k in self.FEATUREDICT.keys():
            self.panel.feature_checkboxes[k].stateChanged.connect(partial(self.feature_selection_changed, k))
        self.panel.segmentSelectionChanged.connect(self.api.set_segment_selection)
        self.panel.generate_DR_button.clicked.connect(self.on_dim_reduce_button_press)
        self.vis_panel.segmentSelectionChanged.connect(self.api.set_segment_selection)

        self.worker_signals.finished.connect(self.panel.on_finished_signal)
        self.worker_signals.progress.connect(self.panel.on_progress_signal)
        
        # connect to api
        self.api.segmentSelectionChanged.connect(self.on_segment_selection_changed)
        self.api.projectLoaded.connect(self.on_project_ready)
        self.api.projectDataLoaded.connect(self.on_project_data_loaded)
        self.api.segmentsDeleted.connect(self.on_segments_deleted)
        self.api.segmentsCreated.connect(self.on_segments_created)


    def on_segment_selection_changed(self):
        selection = self.api.get_segment_selection()
        self.panel.on_selection_changed(selection)
        self.vis_panel.on_selection_changed(selection)

    def plugin_panel_widget(self):
        return [self.panel, self.vis_panel]
    # def add_plugin_menu(self, menu_parent):
    #     menu = menu_parent.addMenu("&Segments")
    #     menu.addAction(self.generate_all_features)
    #     menu.addAction(self.delete_selection_action)
    #     menu.addAction(self.merge_selection_action)
    #     return menu
    def reset_config(self):
        self.config['ncores'] = 6
        self.config['overwrite'] = False

    @property
    def _datastore(self):
        return self.api.get_mut_datastore()

    @property
    def _segments_datastore(self):
        """Returns the segments datastore, ensuring feature columns exist"""
        datastore = self._datastore
        if 'segments' not in datastore:
            return None
        seg_df = datastore['segments']
        # Ensure all feature columns exist in the segments datastore
        for feat in self.featurelist:
            if feat not in seg_df.columns:
                seg_df[feat] = np.nan
        return seg_df

    def _get_feature_value(self, seg_id, feature):
        """Get a feature value for a segment"""
        seg_df = self._segments_datastore
        if seg_df is None or seg_id not in seg_df.index:
            return np.nan
        return seg_df.at[seg_id, feature]

    def _set_feature_value(self, seg_id, feature, value):
        """Set a feature value for a segment"""
        seg_df = self._segments_datastore
        if seg_df is not None and seg_id in seg_df.index:
            seg_df.at[seg_id, feature] = value

    def needs_saving(self):
        return self._needs_saving

    def on_project_ready(self):
        """Called once - features are now loaded as part of segments.csv by the segments plugin"""
        # Features are stored directly in the segments datastore
        # The segments plugin handles loading, we just ensure columns exist when accessed
        pass

    def on_project_data_loaded(self):
        """Called each time project data is loaded"""
        # Features are now stored directly in the segments datastore
        seg_db = self._segments_datastore
        if seg_db is None:
            return

        # Extract just the feature columns for the panel
        feature_df = seg_db[self.featurelist].copy()
        self.panel.set_data(feature_df)
        self.vis_panel.add_features_to_dropdown(self.featurelist)

        # Also add any custom features (PCA, UMAP, etc.) that were saved
        custom_features = self.get_custom_features()
        if custom_features:
            self.vis_panel.add_features_to_dropdown(custom_features)
            # Also add custom features to the panel table
            for feat in custom_features:
                if feat not in self.panel.indiv_features:
                    self.panel.add_feature(feat, seg_db[feat])

        # Pass the tag color function for coloring dots by label
        func_get_color = self.api.get_plugin("TagPlugin").get_tag_color
        self.vis_panel.update_spots(func_get_color)
    
    def get_feat_percent(self, feature):
        seg_db = self._segments_datastore
        if seg_db is None or feature not in seg_db.columns:
            return 1.0  # 100% null if no data
        return seg_db[feature].isnull().mean()

    def get_number_of_stim_for_selection(self, features):
        seg_db = self._segments_datastore
        if seg_db is None or not features:
            return 0, 0
        feature_db = seg_db[features]
        feature_db_tmp = feature_db.dropna()
        return len(feature_db_tmp), len(feature_db)

    def on_dim_reduce_button_press(self):
        # Make a popup window to select the features to include
        # Then generate the PCA features
        features_to_include = self.FEATUREDICT.copy()
        custom_feats = self.get_custom_features()
        if len(custom_feats) > 0:
            features_to_include['Custom'] = self.get_custom_features()
        feat_percents = {}
        for k in features_to_include.keys():
            feat_percents[k] = [self.get_feat_percent(f) for f in features_to_include[k]]

        self.dim_red_window = DimensionalityReductionWizard(features=features_to_include, feature_percents=feat_percents, feat_check_callback = self.get_number_of_stim_for_selection)
        self.dim_red_window.feature_generation_signal.connect(self.on_dim_reduction_generate)
        self.dim_red_window.show()
    
    def on_dim_reduction_generate(self, features, dim_reduction_type, new_feature_name):
        print(features, dim_reduction_type, new_feature_name)
        if dim_reduction_type == "PCA":
            self.generate_PCA_Feature(new_feature_name, features)
        elif dim_reduction_type == "UMAP":
            self.generate_UMAP_Feature(new_feature_name, features)
        self.dim_red_window.close()

    def generate_PCA_Feature(self, feat_name, features):
        """Generates PCA Feature based on currently visible features"""
        seg_db = self._segments_datastore
        if seg_db is None:
            return
        feature_db = seg_db[features].dropna()
        data = feature_db.to_numpy()
        # todo could balance across channels
        Zdata = (data - data.mean(axis=0))/ np.std(data,axis=0,ddof=1)

        # PCA the data
        pca = PCA(n_components=10, svd_solver='full')
        Z_PCA_DATA = pca.fit_transform(Zdata)
        # Add the PCA data directly to the segments datastore
        for i in range(Z_PCA_DATA.shape[1]):
            col_name = feat_name + str(i)
            seg_db.loc[feature_db.index, col_name] = Z_PCA_DATA[:,i]
            self.panel.add_feature(col_name, seg_db[col_name])
            self.vis_panel.add_features_to_dropdown([col_name])
        self._needs_saving = True

    def generate_UMAP_Feature(self, feat_name, features):
        """Generates UMAP Feature based on selected features"""
        seg_db = self._segments_datastore
        if seg_db is None:
            return
        feature_db = seg_db[features].dropna()
        data = feature_db.to_numpy()

        # Z-score normalize
        Zdata = (data - data.mean(axis=0)) / np.std(data, axis=0, ddof=1)

        # UMAP the data
        reducer = umap.UMAP(n_components=2)
        Z_UMAP_DATA = reducer.fit_transform(Zdata)

        # Add the UMAP data directly to the segments datastore
        for i in range(Z_UMAP_DATA.shape[1]):
            col_name = feat_name + str(i)
            seg_db.loc[feature_db.index, col_name] = Z_UMAP_DATA[:, i]
            self.panel.add_feature(col_name, seg_db[col_name])
            self.vis_panel.add_features_to_dropdown([col_name])
        self._needs_saving = True


# FEATURE GENERATION
    def launch_feature_generation_async(self):
        # ParallelFeatureWorker is a QThread, so just call generate_all_features
        # directly from the main thread — QThread.start() handles the background work
        self.generate_all_features()

    def on_segments_created(self, seg_ids):
        """Handle segment creation - efficiently add rows (works for single or batch)"""
        seg_db = self._segments_datastore
        if seg_db is None:
            return

        # Feature columns are already NaN by default (ensured by _segments_datastore property)
        # seg_ids are guaranteed to exist since SegmentPlugin adds to datastore before emitting signal
        feature_df = seg_db.loc[seg_ids, self.featurelist]
        self.panel.add_rows_batch(feature_df)

    def on_segments_deleted(self, seg_ids):
        """Handle segment deletion - efficiently remove rows (works for single or batch)"""
        # Segment is already removed from datastore by segments plugin
        # Use batch removal for efficiency
        self.panel.remove_rows_by_segIDs(seg_ids)
        self.vis_panel.remove_spots(seg_ids)

    def _get_cached_filters(self, sr, lowpass=6000, highpass=200):
        """Return cached (highpass, lowpass) filter coefficients, recomputing only if sr changes."""
        cache_key = (sr, lowpass, highpass)
        if not hasattr(self, '_filter_cache') or self._filter_cache_key != cache_key:
            nfilt = 1024
            self._cached_highpass_filter = firwin(nfilt - 1, 2.0 * highpass / sr, pass_zero=False)
            self._cached_lowpass_filter = firwin(nfilt, 2.0 * lowpass / sr)
            self._filter_cache_key = cache_key
        return self._cached_highpass_filter, self._cached_lowpass_filter

    def get_segment_audio(self, segmentID, lowpass=6000, highpass=200):
        """Get the audio data for a segment"""
        seg = self._datastore['segments'].loc[segmentID]
        sr = self.api.project.sampling_rate
        # StartIndex/StopIndex are now raw integers, convert to ProjectIndex for API call
        start_idx = self.api.make_project_index(seg.StartIndex)
        stop_idx = self.api.make_project_index(seg.StopIndex)
        t, audio = self.api.get_signal(start_idx, stop_idx)
        audio = audio[:,seg.Source.channel]

        # Apply filtering using cached coefficients
        highpassFilter, lowpassFilter = self._get_cached_filters(sr, lowpass, highpass)

        soundLen = len(audio)
        padlen = min(soundLen-10, 3*len(highpassFilter))
        soundIn = filtfilt(highpassFilter, [1.0], audio, padlen=padlen)

        padlen = min(soundLen-10, 3*len(lowpassFilter))
        soundIn = filtfilt(lowpassFilter, [1.0], soundIn, padlen=padlen)

        return soundIn, sr

    def get_segment_load_info(self, segmentID, lowpass=6000, highpass=200) -> SegmentLoadInfo:
        """Get info needed to load a segment's audio independently (for parallel I/O).

        This returns file path, indices, and channel so that a worker thread
        can open its own file handle and read the data.
        """
        seg = self._datastore['segments'].loc[segmentID]
        project = self.api.project
        sr = project.sampling_rate

        # Get project-level indices
        start_idx = int(seg.StartIndex)
        stop_idx = int(seg.StopIndex)

        # Find which block this segment is in
        block_start_frames = project._block_start_frames
        block_idx = bisect.bisect_right(block_start_frames, start_idx) - 1
        block = project.blocks[block_idx]

        # Convert to block-local indices
        block_start = block_start_frames[block_idx]
        local_start = start_idx - block_start
        local_stop = stop_idx - block_start

        # Get file path and channel within file for this segment's source channel
        file_path, file_channel = block.get_channel_info(seg.Source.channel)

        # Convert Path to string if needed
        file_path_str = str(file_path)

        # Determine file type
        if file_path_str.lower().endswith('.nwb'):
            file_type = 'nwb'
        elif file_path_str.lower().endswith('.dat'):
            file_type = 'dat'
        else:
            file_type = 'wav'

        return SegmentLoadInfo(
            seg_id=segmentID,
            file_path=file_path_str,
            file_type=file_type,
            start_index=local_start,
            stop_index=local_stop,
            channel=file_channel,
            sampling_rate=sr,
            lowpass=lowpass,
            highpass=highpass
        )

    def generate_all_features(self, overwrite=False):
        """Generate features for all segments"""
        # first go through all segments and identify the segments
        # that have not been processed yet
        # then generate features for those segments
        # and update the segments datastore

        seg_db = self._segments_datastore
        if seg_db is None:
            return

        all_segIDs = seg_db.index
        if overwrite:
            unprocessed_segIDs = list(all_segIDs)
        else:
            # Vectorized: find segments where any selected feature category has all NaN values
            needs_processing = pd.Series(False, index=all_segIDs)
            for feature_cat, selected in self.feature_selections.items():
                if selected:
                    cat_features = self.FEATUREDICT[feature_cat]
                    # A category needs processing if ALL its features are NaN for a segment
                    all_nan_in_cat = seg_db[cat_features].isna().all(axis=1)
                    needs_processing |= all_nan_in_cat
            unprocessed_segIDs = list(all_segIDs[needs_processing])

        self.n_to_process = len(unprocessed_segIDs)
        print(f"Unprocessed segments: {len(unprocessed_segIDs)}", flush=True)
        print(f"Feature selections: {self.feature_selections}", flush=True)

        if self.n_to_process == 0:
            print("No segments to process.", flush=True)
            return

        # Initialize batch buffer for progress updates
        self._progress_buffer = {}
        self._PROGRESS_FLUSH_INTERVAL = max(1, self.n_to_process // 20)  # ~20 UI updates total

        # Group segments by block
        project = self.api.project
        sr = project.sampling_rate
        block_start_frames = project._block_start_frames
        blocks_by_idx: dict = {}

        for seg_id in unprocessed_segIDs:
            try:
                seg = self._datastore['segments'].loc[seg_id]
                start_idx = int(seg.StartIndex)
                stop_idx = int(seg.StopIndex)
                block_idx = bisect.bisect_right(block_start_frames, start_idx) - 1
                block = project.blocks[block_idx]
                block_start = block_start_frames[block_idx]
                local_start = start_idx - block_start
                local_stop = stop_idx - block_start
                _, file_channel = block.get_channel_info(seg.Source.channel)

                if block_idx not in blocks_by_idx:
                    blocks_by_idx[block_idx] = BlockFeatureInfo(
                        block_index=block_idx,
                        block=block,
                        sampling_rate=sr,
                        segments=[]
                    )
                blocks_by_idx[block_idx].segments.append(
                    BlockSegInfo(seg_id, local_start, local_stop, file_channel)
                )
            except Exception as e:
                print(f"Failed to get block info for segment {seg_id}: {e}", flush=True)

        blocks_info = list(blocks_by_idx.values())
        nworkers = min(16, len(blocks_info))
        print(f"Processing {self.n_to_process} segments across {len(blocks_info)} block(s) "
              f"using {nworkers} thread(s)...", flush=True)

        self.parallel_worker = ParallelFeatureWorker(
            blocks_info=blocks_info,
            feature_selections=self.feature_selections,
            max_workers=nworkers
        )
        self.parallel_worker.progress.connect(self._on_parallel_progress)
        self.parallel_worker.start()

    
    def _on_parallel_progress(self, segID: int, result: SegmentFeatureResult, current: int, total: int, message: str):
        """Update progress during parallel processing.

        Accumulates results and flushes to the datastore/UI in batches
        to avoid per-segment overhead from pandas .at[] and panel updates.
        """
        self.n_to_process -= 1

        if result.features is not None:
            # Flatten nested feature dict and buffer it
            flat = {}
            for category, feat_dict in result.features.items():
                flat.update(feat_dict)
            self._progress_buffer[segID] = flat

        # Flush buffer every _PROGRESS_FLUSH_INTERVAL results, or when done
        if len(self._progress_buffer) >= self._PROGRESS_FLUSH_INTERVAL or self.n_to_process == 0:
            self._flush_progress_buffer()

        self.worker_signals.progress.emit(int(current / total * 100))

        if self.n_to_process == 0:
            self._on_parallel_finished()

    def _flush_progress_buffer(self):
        """Write buffered feature results to the datastore and update UI in one batch."""
        if not self._progress_buffer:
            return

        seg_db = self.api.get_mut_datastore().get('segments')
        featurelist = self.featurelist

        # Build a DataFrame from the buffer and write all at once
        batch_df = pd.DataFrame.from_dict(self._progress_buffer, orient='index')
        for col in batch_df.columns:
            if col in seg_db.columns:
                seg_db.loc[batch_df.index, col] = batch_df[col]

        # Batch update the panel
        feature_df = seg_db.loc[list(self._progress_buffer.keys()), featurelist]
        self.panel.model.set_data(
            seg_db[featurelist].copy()
        )

        self._progress_buffer.clear()

    def _on_parallel_finished(self):
        """Handle completed parallel feature generation."""
        self.worker_signals.finished.emit()
        self.parallel_worker = None
        self._needs_saving = True


    def save(self):
        """Features are saved as part of segments.csv by the segments plugin"""
        # The segments plugin handles saving all columns including features
        self._needs_saving = False

def progress_updater(progress_queue_in, progress_signal_out, n):
    n_done = 0
    # loop until done or sigabbrt
    while n_done < n:
        seg_id_done = progress_queue_in.get()
        n_done += 1
        progress_signal_out.emit(n_done / n * 100)

def load_all_audio(load_func, seg_ids, queue, max_queue_size, done_event):
    print(f"Loading thread started, {len(seg_ids)} segments to load", flush=True)
    for i, seg_id in enumerate(seg_ids):
        audio, sr = load_func(seg_id)
        while queue.qsize() > max_queue_size:
            time.sleep(.1)
        queue.put((seg_id, audio, sr))
        if i < 3 or i % 10 == 0:
            print(f"Loaded segment {seg_id} ({i+1}/{len(seg_ids)}), queue size={queue.qsize()}", flush=True)
    print(f"Loading thread done, setting done_event", flush=True)
    done_event.set()
class FeatureExtractionProcess(Process):
    def __init__(self, in_queue, out_queue, stop_signal, feature_selections):
        super().__init__()
        self.input_queue = in_queue
        self.stop_signal = stop_signal
        self.output_queue = out_queue
        self.feature_selections = feature_selections

    def run(self):
        try:
            pid = os.getpid()
            print(f"Beginning Feature Extraction Process (pid={pid})", flush=True)
            print(f"[pid={pid}] Feature selections: {self.feature_selections}", flush=True)
            
            processed_count = 0
            # Continue while stop signal is not set, OR while there might still be items
            while True:
                # Check if we should stop
                if self.stop_signal.is_set():
                    # Do one final check for remaining items
                    try:
                        segmentID, audio, sr = self.input_queue.get_nowait()
                    except:
                        print(f"[pid={pid}] Stop signal received, no more items. Processed {processed_count} segments.", flush=True)
                        break
                else:
                    # Normal operation - wait for items
                    try:
                        segmentID, audio, sr = self.input_queue.get(timeout=0.5)
                    except Empty:
                        # Timeout is normal - just loop again
                        continue
                
                # Process the segment
                try:
                    print(f"[pid={pid}] Processing segment {segmentID}", flush=True)
                    features = generate_audio_features(audio, sr, segmentID, self.feature_selections)
                    self.output_queue.put(features)
                    processed_count += 1
                    if processed_count % 10 == 0:
                        print(f"[pid={pid}] Processed {processed_count} segments", flush=True)
                except Exception as e:
                    print(f"[pid={pid}] Error processing segment {segmentID}: {e}", flush=True)
                    traceback.print_exc()
                    continue
                    
            print(f"[pid={pid}] Exiting feature extraction process. Total processed: {processed_count}", flush=True)
        except Exception as e:
            print(f"[pid={os.getpid()}] Fatal error in process: {e}", flush=True)
            traceback.print_exc()





def _load_audio_standalone(load_info: SegmentLoadInfo) -> Tuple[int, np.ndarray, int]:
    """Load audio from a file with its own file handle (thread-safe).

    This function creates its own file handle, reads the data, and closes it.
    Safe to call from multiple threads simultaneously.

    Returns
    -------
    seg_id : int
    audio : np.ndarray
    sr : int
    """
    seg_id = load_info.seg_id
    path = load_info.file_path
    i0 = load_info.start_index
    i1 = load_info.stop_index
    channel = load_info.channel
    sr = load_info.sampling_rate

    if load_info.file_type == 'nwb':
        # NWB files need special handling
        with NWBHDF5IO(path, 'r') as io:
            nwbfile = io.read()
            # Find microphone data
            if 'audio' in nwbfile.acquisition:
                mic_series = nwbfile.acquisition['audio']
            else:
                acquisition_names = list(nwbfile.acquisition.keys())
                mic_series = nwbfile.acquisition[acquisition_names[0]]

            data = mic_series.data[i0:i1]
            if len(data.shape) == 1:
                audio = data.astype(np.float32)
            else:
                audio = data[:, channel].astype(np.float32)
    else:
        # WAV and other soundfile-supported formats
        with soundfile.SoundFile(path, 'r') as f:
            f.seek(i0)
            data = f.read(i1 - i0, dtype=np.float32, always_2d=True)
            audio = data[:, channel]

    # Apply filtering
    nfilt = 1024
    highpassFilter = firwin(nfilt - 1, 2.0 * load_info.highpass / sr, pass_zero=False)
    lowpassFilter = firwin(nfilt, 2.0 * load_info.lowpass / sr)

    soundLen = len(audio)
    if soundLen > 10:
        padlen = min(soundLen - 10, 3 * len(highpassFilter))
        audio = filtfilt(highpassFilter, [1.0], audio, padlen=padlen)
        padlen = min(soundLen - 10, 3 * len(lowpassFilter))
        audio = filtfilt(lowpassFilter, [1.0], audio, padlen=padlen)

    return seg_id, audio, sr


class ParallelFeatureWorker(QThread):
    """Worker thread for parallel feature extraction across blocks.

    Mirrors the ParallelSegmentWorker pattern: one task per block,
    each task reads the block once and processes all its segments.
    """

    progress = pyqtSignal(int, object, int, int, str)  # segmentID, result, current, total, message
    error = pyqtSignal(str)

    def __init__(
        self,
        blocks_info: List[BlockFeatureInfo],
        feature_selections,
        max_workers: int = 4
    ):
        super().__init__()
        self.blocks_info = blocks_info
        self.feature_selections = feature_selections
        self.max_workers = max_workers

    def _process_block(self, block_info: BlockFeatureInfo) -> List[SegmentFeatureResult]:
        """Read a block once and extract features for all its segments."""
        results = []
        sr = block_info.sampling_rate
        nfilt = 1024
        hp = firwin(nfilt - 1, 2.0 * block_info.highpass / sr, pass_zero=False)
        lp = firwin(nfilt, 2.0 * block_info.lowpass / sr)

        for seg in block_info.segments:
            try:
                audio = block_info.block.read(seg.local_start, seg.local_stop, channels=[seg.channel])
                audio = audio[:, 0].astype(np.float32)
                n = len(audio)
                if n > 10:
                    audio = filtfilt(hp, [1.0], audio, padlen=min(n - 10, 3 * len(hp)))
                    audio = filtfilt(lp, [1.0], audio, padlen=min(n - 10, 3 * len(lp)))
                features = _extract_features(audio, sr, feature_selections=self.feature_selections, normalize=True)
                results.append(SegmentFeatureResult(seg.seg_id, features))
            except Exception as e:
                logger.exception(f"Error processing segment {seg.seg_id} in block {block_info.block_index}")
                results.append(SegmentFeatureResult(seg.seg_id, None, str(e)))

        return results

    def run(self):
        try:
            total_segments = sum(len(b.segments) for b in self.blocks_info)
            total_blocks = len(self.blocks_info)
            if total_segments == 0:
                return

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_block = {
                    executor.submit(self._process_block, block_info): block_info
                    for block_info in self.blocks_info
                }

                completed = 0
                for future in as_completed(future_to_block):
                    block_info = future_to_block[future]
                    try:
                        block_results = future.result()
                    except Exception as e:
                        logger.exception(f"Error in block {block_info.block_index}")
                        block_results = [
                            SegmentFeatureResult(seg.seg_id, None, str(e))
                            for seg in block_info.segments
                        ]

                    for result in block_results:
                        completed += 1
                        self.progress.emit(
                            result.segmentID, result, completed, total_segments,
                            f"Extracted features {completed}/{total_segments}"
                        )

        except Exception as e:
            logger.exception("Error during parallel feature extraction")
            self.error.emit(str(e))


#from numba import jit
#@jit(nogil=True)
def _extract_features( audio: np.ndarray, sr: int, feature_selections: dict, normalize: bool) -> List[float]:
    """Extract features from audio"""
    if normalize:
        audio = audio / np.max(audio)
    output_features = dict() 
    if feature_selections is None or feature_selections['Amplitude']:
        output_features['Amplitude'] = features_ampenv(audio, sr)
    if feature_selections is None or feature_selections['Fundamental']:
        output_features['Fundamental'] = features_fundamental(audio, sr)
    if feature_selections is None or feature_selections['Formant']:
        output_features['Formant'] = features_formants(audio, sr)
    if feature_selections is None or feature_selections['Spectrum']:
        output_features['Spectrum'] = features_spectrum(audio, sr)
    return output_features


# from numba import jit
# @jit(nogil=True)
def generate_audio_features(audio, sr, segmentID, feature_selections = None, normalize=True):
    """Generate features for audio data"""
    return segmentID, _extract_features(audio, sr, feature_selections = feature_selections, normalize=normalize)

def generate_segment_features(segmentID, seg_datastore, api):
    """Generate features for a single segment"""
    # get the segment
    seg = seg_datastore.loc[segmentID]
    sr = api.project.sampling_rate
    # get the audio data for the segment
    # StartIndex/StopIndex are now raw integers, convert to ProjectIndex for API call
    start_idx = api.make_project_index(seg.StartIndex)
    stop_idx = api.make_project_index(seg.StopIndex)
    # TODO could pad small segments here
    t, audio = api.get_signal(start_idx, stop_idx)
    audio = audio[:,seg.Source.channel]
    # get the sampling rate
    # get the features
    features = _extract_features(audio, sr, segmentID=segmentID, normalize=True)
    return features


# TODO Move this to a utils file
def features_ampenv(audio, sr, cutoff_freq = 20, amp_sample_rate = 1000):
    # Calculates the amplitude enveloppe and related parameters
    (amp, tdata)  = sound.temporal_envelope(audio, sr, cutoff_freq=cutoff_freq, resample_rate=amp_sample_rate)
    
    # Here are the parameters
    ampdata = amp/np.sum(amp)
    meantime = np.sum(tdata*ampdata)
    stdtime = np.sqrt(np.sum(ampdata*((tdata-meantime)**2)))
    skewtime = np.sum(ampdata*(tdata-meantime)**3)
    skewtime = skewtime/(stdtime**3)
    kurtosistime = np.sum(ampdata*(tdata-meantime)**4)
    kurtosistime = kurtosistime/(stdtime**4)
    indpos = np.where(ampdata>0)[0]
    entropytime = -np.sum(ampdata[indpos]*np.log2(ampdata[indpos]))/np.log2(np.size(indpos))
    return dict({
        "rms": audio.std(),
        "meantime": meantime,
        "stdT": stdtime,
        "skewT": skewtime,
        "kurtT": kurtosistime,
        "entT": entropytime,
        "maxAmp": max(amp),
    })



def features_fundamental(audio, sr, maxFund = 1500, minFund = 300, lowFc = 200, highFc = 6000, minSaliency = 0.5, method='HPS'):
    funds_salience = sound.fundEstOptim(audio, sr, maxFund = maxFund, minFund = minFund, nofilt=True, lowFc = lowFc, highFc = highFc, minSaliency = minSaliency, method = method)
    f0 = funds_salience[:,0]
    f0_2 = funds_salience[:,1]
    sal = funds_salience[:,2]
    sal_2 = funds_salience[:,3]
    if np.isnan(f0).all():
        fund = np.nan
        meansal = np.nan
        maxfund = np.nan
        minfund = np.nan
        cvfund = np.nan
        devfund = np.nan
    else:
        meansal = np.nanmean(sal)
        fund = np.nanmean(f0)
        maxfund = np.nanmax(f0)
        minfund = np.nanmin(f0)
        cvfund = np.nanstd(f0)/fund
        devfund = np.nanmean(np.diff(f0))
    if np.isnan(f0_2).all():
        fund2 = np.nan
        meansal2 = np.nan
        cvfund2 = np.nan
    else:
        fund2 = np.nanmean(f0_2)
        cvfund2 = np.nanstd(f0_2)/fund2
        meansal2 = np.nanmean(sal_2)
    return dict({
        "fund":fund,
        "sal":meansal,
        "fund2":fund2,
        "sal2":meansal2,
        "maxfund":maxfund,
        "minfund":minfund,
        "cvfund":cvfund,
        "cvfund2":cvfund2,
        "devfund":devfund
        })
    #self.voice2percent = np.nanmean(funds_salience[:,4])*100

def features_formants(audio, sr,  lowFc = 200, highFc = 6000, minFormantFreq = 500, maxFormantBW = 1000, windowFormant = 0.1):
    formants = sound.formantEstimator(audio, sr, nofilt=True, lowFc=lowFc, highFc=highFc, windowFormant = windowFormant,
                                    minFormantFreq = minFormantFreq, maxFormantBW = maxFormantBW )
    F1 = formants[:,0]
    F2 = formants[:,1]
    F3 = formants[:,2]

    # Take the time average formants only if there are some non nan numbers
    if np.sum(~np.isnan(F1)) > 0:
        meanF1 = np.nanmean(F1)
    else:
        meanF1 = np.nan
    if np.sum(~np.isnan(F2)) > 0:
        meanF2 = np.nanmean(F2)
    else:
        meanF2 = np.nan
    if np.sum(~np.isnan(F3)) > 0:
        meanF3 = np.nanmean(F3)
    else:
        meanF3 = np.nan

    return dict({
        "F1":meanF1,
        "F2":meanF2,
        "F3":meanF3
    })


def features_spectrum(audio, sr, f_high = 10000):
    Pxx, Freqs = sound.mlab.psd(audio,Fs=sr,NFFT=1024, noverlap=512)
    
    # Find quartile power
    cum_power = np.cumsum(Pxx)
    tot_power = np.sum(Pxx)
    quartile_freq = np.zeros(3, dtype = 'int')
    quartile_values = [0.25, 0.5, 0.75]
    nfreqs = np.size(cum_power)
    iq = 0
    for ifreq in range(nfreqs):
        if (cum_power[ifreq] > quartile_values[iq]*tot_power):
            quartile_freq[iq] = ifreq
            iq = iq+1
            if (iq > 2):
                break
                
    # Find skewness, kurtosis and entropy for power spectrum below f_high
    fmax_candidates = np.where(Freqs > f_high)[0]
    # If f_high is at or above the Nyquist frequency (e.g. low sampling rate
    # projects), no bin exceeds it -- fall back to using the whole spectrum
    # instead of crashing.
    ind_fmax = fmax_candidates[0] if len(fmax_candidates) else len(Freqs)

    # Description of spectral shape
    spectdata = Pxx[0:ind_fmax]
    freqdata = Freqs[0:ind_fmax]
    spectdata = spectdata/np.sum(spectdata)
    meanspect = np.sum(freqdata*spectdata)
    stdspect = np.sqrt(np.sum(spectdata*((freqdata-meanspect)**2)))
    skewspect = np.sum(spectdata*(freqdata-meanspect)**3)
    skewspect = skewspect/(stdspect**3)
    kurtosisspect = np.sum(spectdata*(freqdata-meanspect)**4)
    kurtosisspect = kurtosisspect/(stdspect**4)
    entropyspect = -np.sum(spectdata*np.log2(spectdata))/np.log2(ind_fmax)

    # Storing the values       
    meanspect = meanspect
    stdspect = stdspect
    skewspect = skewspect
    kurtosisspect = kurtosisspect
    entropyspect = entropyspect
    q1 = Freqs[quartile_freq[0]]
    q2 = Freqs[quartile_freq[1]]
    q3 = Freqs[quartile_freq[2]]

    return dict({
        "meanS":meanspect,
        "stdS":stdspect,
        "skewS":skewspect,
        "kurtS":kurtosisspect,
        "entS":entropyspect,
        "q1":q1,
        "q2":q2,
        "q3":q3
    })