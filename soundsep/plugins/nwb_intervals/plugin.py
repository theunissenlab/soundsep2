import logging
import os
from typing import List

import PyQt6.QtWidgets as widgets
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6 import QtGui

from soundsep.core.base_plugin import BasePlugin
from soundsep.core.models import NWBFile
from soundsep.core.utils import hhmmss

try:
    from pynwb import NWBHDF5IO
    HAS_NWB = True
except ImportError:
    HAS_NWB = False


logger = logging.getLogger(__name__)


class TimeQTableWidgetItem(widgets.QTableWidgetItem):
    def __init__(self, time: float):
        """A QTableWidgetItem that sorts by time"""
        super().__init__(hhmmss(time, dec=3))
        self.time = time

    def __lt__(self, other):
        return self.time < other.time


class NWBIntervalsPanel(widgets.QWidget):
    """Panel for displaying and navigating NWB file intervals"""

    intervalSelected = pyqtSignal(float)  # Emits start time when interval is clicked
    intervalsChanged = pyqtSignal(list)  # Emits list of (start, stop) tuples when intervals change

    def __init__(self, parent=None):
        super().__init__(parent)
        self.nwb_files = []
        self.current_interval_type = None
        self.current_label_column = None
        self.init_ui()
        self.init_actions()

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        
        # Add dropdown for selecting interval type
        selector_layout = widgets.QHBoxLayout()
        selector_layout.addWidget(widgets.QLabel("Interval Type:"))
        self.interval_type_combo = widgets.QComboBox()
        self.interval_type_combo.addItem("(Select interval type)")
        selector_layout.addWidget(self.interval_type_combo)
        selector_layout.addStretch()
        
        # Add dropdown for selecting label column
        selector_layout.addWidget(widgets.QLabel("Label Column:"))
        self.label_column_combo = widgets.QComboBox()
        self.label_column_combo.addItem("(None)")
        selector_layout.addWidget(self.label_column_combo)
        
        layout.addLayout(selector_layout)
        
        # Add table for displaying intervals
        self.table = widgets.QTableWidget(0, 5)
        self.table.setEditTriggers(widgets.QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(widgets.QTableWidget.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        self.table.setHorizontalHeaderLabels([
            "File",
            "Start Time",
            "Stop Time",
            "Duration",
            "Label"
        ])
        header.setSectionResizeMode(0, widgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, widgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, widgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, widgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, widgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        
        # Add info label
        self.info_label = widgets.QLabel("No NWB files loaded")
        self.info_label.setStyleSheet("color: gray;")
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)

    def init_actions(self):
        self.interval_type_combo.currentIndexChanged.connect(self.on_interval_type_changed)
        self.label_column_combo.currentIndexChanged.connect(self.on_label_column_changed)
        self.table.cellDoubleClicked.connect(self.on_cell_clicked)

    def set_nwb_files(self, nwb_files: List[NWBFile]):
        """Set the NWB files to display intervals from"""
        self.nwb_files = nwb_files
        self.refresh_interval_types()
        
        if not nwb_files:
            self.info_label.setText("No NWB files loaded")
            self.info_label.setStyleSheet("color: gray;")
        else:
            self.info_label.setText(f"Loaded {len(nwb_files)} NWB file(s)")
            self.info_label.setStyleSheet("color: green;")

    def refresh_interval_types(self):
        """Refresh the list of available interval types from NWB files"""
        self.interval_type_combo.clear()
        self.interval_type_combo.addItem("(Select interval type)")
        
        if not HAS_NWB or not self.nwb_files:
            return
        
        # Collect all unique interval types from all NWB files
        interval_types = set()
        for nwb_file in self.nwb_files:
            try:
                with NWBHDF5IO(nwb_file.path, 'r') as io:
                    nwbfile = io.read()
                    if hasattr(nwbfile, 'intervals'):
                        interval_types.update(nwbfile.intervals.keys())
            except Exception as e:
                logger.error(f"Error reading intervals from {nwb_file.path}: {e}")
        
        # Add to combo box
        for interval_type in sorted(interval_types):
            self.interval_type_combo.addItem(interval_type)

    def on_interval_type_changed(self, index):
        """Called when user selects a different interval type"""
        if index <= 0:
            self.current_interval_type = None
            self.current_label_column = None
            self.table.setRowCount(0)
            self.label_column_combo.clear()
            self.label_column_combo.addItem("(None)")
            return
        
        self.current_interval_type = self.interval_type_combo.currentText()
        self.refresh_label_columns()
        self.refresh_intervals()
    
    def refresh_label_columns(self):
        """Refresh the list of available columns for the selected interval type"""
        self.label_column_combo.clear()
        self.label_column_combo.addItem("(None)")
        
        if not self.current_interval_type or not HAS_NWB or not self.nwb_files:
            return
        
        # Collect all unique column names from all NWB files for this interval type
        column_names = set()
        for nwb_file in self.nwb_files:
            try:
                with NWBHDF5IO(nwb_file.path, 'r') as io:
                    nwbfile = io.read()
                    
                    if not hasattr(nwbfile, 'intervals'):
                        continue
                    
                    if self.current_interval_type not in nwbfile.intervals:
                        continue
                    
                    intervals_table = nwbfile.intervals[self.current_interval_type]
                    
                    # Get column names (excluding start_time and stop_time)
                    if hasattr(intervals_table, 'colnames'):
                        for col in intervals_table.colnames:
                            if col not in ['start_time', 'stop_time']:
                                column_names.add(col)
                    
            except Exception as e:
                logger.error(f"Error reading columns from {nwb_file.path}: {e}")
        
        # Add to combo box
        for col_name in sorted(column_names):
            self.label_column_combo.addItem(col_name)
        
        # Auto-select 'label' if it exists
        label_index = self.label_column_combo.findText('label')
        if label_index > 0:
            self.label_column_combo.setCurrentIndex(label_index)
    
    def on_label_column_changed(self, index):
        """Called when user selects a different label column"""
        if index <= 0:
            self.current_label_column = None
        else:
            self.current_label_column = self.label_column_combo.currentText()
        
        # Refresh the table to show new labels
        if self.current_interval_type:
            self.refresh_intervals()

    def refresh_intervals(self):
        """Refresh the table with intervals of the selected type"""
        self.table.setRowCount(0)
        
        if not self.current_interval_type or not HAS_NWB:
            self.intervalsChanged.emit([])  # Clear intervals from scrollbar
            return
        
        row = 0
        all_intervals = []  # Collect all intervals for the scrollbar
        
        for nwb_file in self.nwb_files:
            try:
                with NWBHDF5IO(nwb_file.path, 'r') as io:
                    nwbfile = io.read()
                    
                    if not hasattr(nwbfile, 'intervals'):
                        continue
                    
                    if self.current_interval_type not in nwbfile.intervals:
                        continue
                    
                    intervals_table = nwbfile.intervals[self.current_interval_type]
                    
                    # Read intervals data
                    start_times = intervals_table.start_time[:]
                    stop_times = intervals_table.stop_time[:]
                    
                    # Add to the list for scrollbar visualization
                    for i in range(len(start_times)):
                        all_intervals.append((start_times[i], stop_times[i]))
                    
                    # Try to get label data from selected column
                    labels = None
                    if self.current_label_column:
                        try:
                            if hasattr(intervals_table, self.current_label_column):
                                label_col = getattr(intervals_table, self.current_label_column)
                                if label_col is not None:
                                    labels = label_col[:]
                        except Exception as e:
                            logger.warning(f"Could not read column '{self.current_label_column}': {e}")
                    
                    # Add each interval to the table
                    for i in range(len(start_times)):
                        self.table.insertRow(row)
                        
                        # File name - use os.path.basename for cross-platform compatibility
                        file_name = os.path.basename(nwb_file.path)
                        file_item = widgets.QTableWidgetItem(file_name)
                        file_item.setData(Qt.ItemDataRole.UserRole, (nwb_file, start_times[i], stop_times[i]))
                        self.table.setItem(row, 0, file_item)
                        
                        # Start time
                        start_item = TimeQTableWidgetItem(start_times[i])
                        self.table.setItem(row, 1, start_item)
                        
                        # Stop time
                        stop_item = TimeQTableWidgetItem(stop_times[i])
                        self.table.setItem(row, 2, stop_item)
                        
                        # Duration
                        duration = stop_times[i] - start_times[i]
                        duration_item = TimeQTableWidgetItem(duration)
                        self.table.setItem(row, 3, duration_item)
                        
                        # Label
                        label_text = labels[i] if labels is not None and i < len(labels) else ""
                        if isinstance(label_text, bytes):
                            label_text = label_text.decode('utf-8')
                        label_item = widgets.QTableWidgetItem(str(label_text))
                        self.table.setItem(row, 4, label_item)
                        
                        row += 1
                        
            except Exception as e:
                logger.error(f"Error reading intervals from {nwb_file.path}: {e}")
        
        if row == 0:
            self.info_label.setText(f"No intervals of type '{self.current_interval_type}' found")
            self.info_label.setStyleSheet("color: orange;")
            self.intervalsChanged.emit([])  # Clear intervals from scrollbar
        else:
            self.info_label.setText(f"Showing {row} interval(s) of type '{self.current_interval_type}'")
            self.info_label.setStyleSheet("color: green;")
            self.intervalsChanged.emit(all_intervals)  # Update scrollbar with intervals

    def on_cell_clicked(self, row, column):
        """Called when user double-clicks a cell - navigate to interval start or stop time"""
        file_item = self.table.item(row, 0)
        if file_item:
            nwb_file, start_time, stop_time = file_item.data(Qt.ItemDataRole.UserRole)
            
            # If user clicked on Stop Time column (column 2), navigate to stop time
            # Otherwise, navigate to start time
            if column == 2:
                self.intervalSelected.emit(stop_time)
            else:
                self.intervalSelected.emit(start_time)


class NWBIntervalsPlugin(BasePlugin):
    """Plugin for viewing and navigating NWB file intervals"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not HAS_NWB:
            logger.warning("pynwb not installed - NWB intervals plugin will not function")
        
        self.panel = NWBIntervalsPanel()
        self.nwb_files = []
        
        if not HAS_NWB:
            self.panel.info_label.setText("pynwb not installed - pip install pynwb")
            self.panel.info_label.setStyleSheet("color: red;")
            self.panel.interval_type_combo.setEnabled(False)
        
        self.connect_events()

    def connect_events(self):
        self.panel.intervalSelected.connect(self.on_interval_selected)
        self.panel.intervalsChanged.connect(self.on_intervals_changed)
        self.api.projectLoaded.connect(self.on_project_ready)

    def on_project_ready(self):
        """Called when project is loaded - scan for NWB files"""
        try:
            if not HAS_NWB:
                logger.warning("pynwb not installed - NWB intervals plugin will not function")
                return
            
            # Find all NWB files in the project
            self.nwb_files = []
            project = self.api.project
            
            for block in project.blocks:
                for audio_file in block._files:
                    if isinstance(audio_file, NWBFile):
                        if audio_file not in self.nwb_files:
                            self.nwb_files.append(audio_file)
            
            self.panel.set_nwb_files(self.nwb_files)
        except Exception as e:
            logger.error(f"Error in NWBIntervalsPlugin.on_project_ready: {e}")
            self.panel.info_label.setText(f"Error loading NWB files: {str(e)}")
            self.panel.info_label.setStyleSheet("color: red;")

    def on_interval_selected(self, start_time_seconds: float):
        """Navigate to the selected interval's start time"""
        try:
            # Convert time in seconds to project frames
            sampling_rate = self.api.project.sampling_rate
            start_frame = int(start_time_seconds * sampling_rate)
            
            # Create project index
            start_index = self.api.make_project_index(start_frame)
            
            # Convert to STFT index
            start_stft = self.api.convert_project_index_to_stft_index(start_index)
            
            # Get current workspace duration
            ws_start, ws_stop = self.api.workspace_get_lim()
            duration = ws_stop - ws_start
            
            # Set new workspace centered on the interval start
            new_start = max(self.api.create_stftindex(0), start_stft - duration // 4)
            new_stop = start_stft + duration * 3 // 4

            self.api.workspace_set_position(new_start, new_stop)
        except Exception as e:
            logger.error(f"Error navigating to interval: {e}")
            self.panel.info_label.setText(f"Error: {str(e)}")
            self.panel.info_label.setStyleSheet("color: red;")

    def on_intervals_changed(self, intervals_data):
        """Update the scrollbar with interval rectangles"""
        try:
            if hasattr(self.gui, 'scrollbar'):
                self.gui.scrollbar.add_intervals(intervals_data)
        except Exception as e:
            logger.error(f"Error updating scrollbar with intervals: {e}")

    def plugin_panel_widget(self):
        """Return the panel widget to be displayed in the plugin toolbox"""
        return [self.panel]

    def needs_saving(self):
        return False

    def save(self):
        pass

    def refresh(self):
        pass
