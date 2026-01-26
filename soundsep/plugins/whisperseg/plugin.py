import logging
from typing import Optional

import numpy as np
import PyQt6.QtWidgets as widgets
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from soundsep.core.base_plugin import BasePlugin


logger = logging.getLogger(__name__)


# Check if whisperseg is available
try:
    from whisperseg import WhisperSegmenter
    HAS_WHISPERSEG = True
except ImportError:
    HAS_WHISPERSEG = False
    logger.warning("whisperseg not installed. WhisperSegPlugin will have limited functionality.")


class SegmentationWorker(QThread):
    """Worker thread for running WhisperSeg segmentation."""

    finished = pyqtSignal(list)  # Emits list of (start, stop) tuples in samples
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, audio: np.ndarray, sampling_rate: int, model_name: str):
        super().__init__()
        self.audio = audio
        self.sampling_rate = sampling_rate
        self.model_name = model_name

    def run(self):
        try:
            self.progress.emit("Loading WhisperSeg model...")
            segmenter = WhisperSegmenter(model_path=self.model_name)

            self.progress.emit("Running segmentation...")
            # WhisperSeg expects audio as 1D array and sampling rate
            result = segmenter.segment(
                self.audio,
                sr=self.sampling_rate,
            )

            # Convert results to sample indices
            # WhisperSeg returns onset/offset in seconds
            segments = []
            if result and 'onset' in result and 'offset' in result:
                for onset, offset in zip(result['onset'], result['offset']):
                    start_sample = int(onset * self.sampling_rate)
                    stop_sample = int(offset * self.sampling_rate)
                    segments.append((start_sample, stop_sample))

            self.finished.emit(segments)

        except Exception as e:
            logger.exception("Error during WhisperSeg segmentation")
            self.error.emit(str(e))


class WhisperSegPanel(widgets.QWidget):
    """Panel widget for WhisperSeg configuration and controls."""

    segmentRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = widgets.QLabel("WhisperSeg")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Model selector
        model_layout = widgets.QHBoxLayout()
        model_label = widgets.QLabel("Model:")
        self.model_selector = widgets.QComboBox()
        self.model_selector.addItems([
            "nccratliri/whisperseg-animal-vad-ct2",
            "nccratliri/whisperseg-large-ms-ct2",
            "nccratliri/whisperseg-base-animal-vad-ct2",
            "Custom...",
        ])
        self.model_selector.setEditable(True)
        self.model_selector.setInsertPolicy(widgets.QComboBox.InsertPolicy.NoInsert)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_selector, 1)
        layout.addLayout(model_layout)

        # Channel selector
        channel_layout = widgets.QHBoxLayout()
        channel_label = widgets.QLabel("Channel:")
        self.channel_selector = widgets.QComboBox()
        channel_layout.addWidget(channel_label)
        channel_layout.addWidget(self.channel_selector, 1)
        layout.addLayout(channel_layout)

        # Time range inputs
        time_group = widgets.QGroupBox("Time Range")
        time_layout = widgets.QFormLayout()

        self.start_time_input = widgets.QDoubleSpinBox()
        self.start_time_input.setDecimals(3)
        self.start_time_input.setSuffix(" s")
        self.start_time_input.setRange(0, 999999)
        self.start_time_input.setSingleStep(0.1)

        self.end_time_input = widgets.QDoubleSpinBox()
        self.end_time_input.setDecimals(3)
        self.end_time_input.setSuffix(" s")
        self.end_time_input.setRange(0, 999999)
        self.end_time_input.setSingleStep(0.1)

        time_layout.addRow("Start:", self.start_time_input)
        time_layout.addRow("End:", self.end_time_input)
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)

        # Use selection button
        self.use_selection_button = widgets.QPushButton("Use Current Selection")
        self.use_selection_button.setToolTip("Set time range from current selection in spectrogram")
        layout.addWidget(self.use_selection_button)

        # Status label
        self.status_label = widgets.QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_label)

        # Segment button
        self.segment_button = widgets.QPushButton("Segment")
        self.segment_button.setStyleSheet("font-weight: bold;")
        self.segment_button.clicked.connect(self.segmentRequested.emit)
        layout.addWidget(self.segment_button)

        # Progress indicator
        self.progress_bar = widgets.QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        self.setLayout(layout)

    def set_status(self, message: str, is_error: bool = False):
        self.status_label.setText(message)
        if is_error:
            self.status_label.setStyleSheet("color: red;")
        else:
            self.status_label.setStyleSheet("color: gray; font-style: italic;")

    def set_processing(self, processing: bool):
        self.segment_button.setEnabled(not processing)
        self.progress_bar.setVisible(processing)
        if processing:
            self.set_status("Processing...")


class WhisperSegPlugin(BasePlugin):
    """Plugin for automatic audio segmentation using WhisperSeg."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.worker: Optional[SegmentationWorker] = None
        self._pending_channel = None

        self.init_ui()
        self.init_actions()
        self.connect_events()

    def init_ui(self):
        self.panel = WhisperSegPanel()

        # Toolbar button
        self.toolbar_button = widgets.QPushButton("WhisperSeg")
        self.toolbar_button.setToolTip("Run WhisperSeg automatic segmentation")

    def init_actions(self):
        self.segment_action = QtGui.QAction("Run WhisperSeg segmentation", self)
        self.segment_action.triggered.connect(self.on_segment_requested)

    def connect_events(self):
        self.toolbar_button.clicked.connect(self.on_segment_requested)
        self.panel.segmentRequested.connect(self.on_segment_requested)
        self.panel.use_selection_button.clicked.connect(self.on_use_selection)

        # Connect to API signals
        self.api.projectLoaded.connect(self.on_project_loaded)
        self.api.selectionChanged.connect(self.on_selection_changed)

    def on_project_loaded(self):
        """Update channel selector when project is loaded."""
        self.panel.channel_selector.clear()

        n_channels = self.api.project.channels
        for i in range(n_channels):
            self.panel.channel_selector.addItem(f"Channel {i}", i)

        # Set end time to project duration
        duration = self.api.project.frames / self.api.project.sampling_rate
        self.panel.end_time_input.setMaximum(duration)
        self.panel.start_time_input.setMaximum(duration)
        self.panel.end_time_input.setValue(min(60.0, duration))  # Default to first 60s or full duration

    def on_selection_changed(self):
        """Called when user selection changes - could auto-update time range."""
        pass

    def on_use_selection(self):
        """Set time range from current selection."""
        selection = self.api.get_fine_selection()
        if selection:
            sr = self.api.project.sampling_rate
            start_time = selection.x0 / sr
            end_time = selection.x1 / sr

            self.panel.start_time_input.setValue(start_time)
            self.panel.end_time_input.setValue(end_time)

            # Also set channel from selection if available
            if selection.source:
                channel = selection.source.channel
                idx = self.panel.channel_selector.findData(channel)
                if idx >= 0:
                    self.panel.channel_selector.setCurrentIndex(idx)

            self.panel.set_status(f"Set range: {start_time:.3f}s - {end_time:.3f}s")
        else:
            self.panel.set_status("No selection available", is_error=True)

    def on_segment_requested(self):
        """Run WhisperSeg segmentation on the specified audio range."""
        if not HAS_WHISPERSEG:
            self.panel.set_status(
                "WhisperSeg not installed. Install with: pip install whisperseg",
                is_error=True
            )
            return

        if self.worker is not None and self.worker.isRunning():
            self.panel.set_status("Segmentation already in progress", is_error=True)
            return

        # Get parameters
        model_name = self.panel.model_selector.currentText()
        channel = self.panel.channel_selector.currentData()
        start_time = self.panel.start_time_input.value()
        end_time = self.panel.end_time_input.value()

        if channel is None:
            self.panel.set_status("Please select a channel", is_error=True)
            return

        if end_time <= start_time:
            self.panel.set_status("End time must be greater than start time", is_error=True)
            return

        # Convert times to sample indices
        sr = self.api.project.sampling_rate
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Read audio data
        try:
            audio = self.api.project[start_sample:end_sample, channel]
            audio = audio.astype(np.float32)

            # Normalize if needed
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

        except Exception as e:
            self.panel.set_status(f"Error reading audio: {e}", is_error=True)
            return

        # Store info for when segmentation completes
        self._pending_channel = channel
        self._pending_start_sample = start_sample

        # Start segmentation in background thread
        self.panel.set_processing(True)

        self.worker = SegmentationWorker(audio, sr, model_name)
        self.worker.finished.connect(self.on_segmentation_finished)
        self.worker.error.connect(self.on_segmentation_error)
        self.worker.progress.connect(lambda msg: self.panel.set_status(msg))
        self.worker.start()

    def on_segmentation_finished(self, segments: list):
        """Handle completed segmentation."""
        self.panel.set_processing(False)

        if not segments:
            self.panel.set_status("No segments detected")
            return

        # Get source for the channel
        source = self._get_or_create_source_for_channel(self._pending_channel)
        if source is None:
            self.panel.set_status("Could not find or create source for channel", is_error=True)
            return

        # Convert relative sample indices to absolute project indices
        start_offset = self._pending_start_sample
        absolute_segments = [
            (start_offset + start, start_offset + stop, source)
            for start, stop in segments
        ]

        # Create segments using SegmentPlugin
        try:
            segment_plugin = self.api.plugins.get("SegmentPlugin")
            if segment_plugin:
                segment_plugin.create_segments_batch(absolute_segments)
                self.panel.set_status(f"Created {len(segments)} segments")
            else:
                self.panel.set_status("SegmentPlugin not available", is_error=True)
        except Exception as e:
            logger.exception("Error creating segments")
            self.panel.set_status(f"Error creating segments: {e}", is_error=True)

        self.worker = None

    def on_segmentation_error(self, error_msg: str):
        """Handle segmentation error."""
        self.panel.set_processing(False)
        self.panel.set_status(f"Error: {error_msg}", is_error=True)
        self.worker = None

    def _get_or_create_source_for_channel(self, channel: int):
        """Get an existing source for the channel or create a new one."""
        sources = self.api.get_sources()

        # Try to find existing source for this channel
        for source in sources:
            if source.channel == channel:
                return source

        # Create new source for this channel
        return self.api.create_source(f"WhisperSeg Ch{channel}", channel)

    def plugin_toolbar_items(self):
        return [self.toolbar_button]

    def add_plugin_menu(self, menu_parent):
        menu = menu_parent.addMenu("&WhisperSeg")
        menu.addAction(self.segment_action)
        return menu

    def plugin_panel_widget(self):
        return [self.panel]

    def setup_plugin_shortcuts(self):
        self.segment_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+W"))
