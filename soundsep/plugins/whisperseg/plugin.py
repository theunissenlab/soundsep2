import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import PyQt6.QtWidgets as widgets
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

from soundsep.core.base_plugin import BasePlugin
from soundsep.core.ampenv import advanced_seg, filter_and_ampenv
from soundsep.plugins.detect import threshold_events


logger = logging.getLogger(__name__)


@dataclass
class BlockReadInfo:
    """Info needed to read audio data for a block."""
    block_index: int
    block_start_sample: int  # Absolute project index where this block starts
    block: object  # The Block object to read from
    channel: int  # Channel index to read
    read_start: int  # Start index within the block
    read_end: int  # End index within the block


@dataclass
class BlockSegmentResult:
    """Result from processing a single block."""
    block_index: int
    block_start_sample: int  # Absolute project index where this block starts
    intervals: np.ndarray  # Detected intervals relative to block start
    error: Optional[str] = None


# Check if whisperseg is available
try:
    from whisperseg import WhisperSegmenter
    HAS_WHISPERSEG = True
except ImportError:
    HAS_WHISPERSEG = False
    logger.warning("whisperseg not installed. WhisperSeg method will be unavailable.")


class WhisperSegWorker(QThread):
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


class ParallelSegmentWorker(QThread):
    """Worker thread for running segmentation across multiple blocks in parallel."""

    finished = pyqtSignal(list)  # Emits list of BlockSegmentResult
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)  # current_block, total_blocks, message

    def __init__(
        self,
        blocks_info: List[BlockReadInfo],  # Block metadata for workers to read files
        sampling_rate: int,
        method: str,  # "basic" or "advanced"
        params: dict,  # Method-specific parameters
        max_workers: int = 4
    ):
        super().__init__()
        self.blocks_info = blocks_info
        self.sampling_rate = sampling_rate
        self.method = method
        self.params = params
        self.max_workers = max_workers

    def _process_block(self, block_info: BlockReadInfo) -> BlockSegmentResult:
        """Process a single block and return detected intervals."""
        try:
            # Read audio data using the Block's read method (handles WAV, NWB, DAT)
            audio = block_info.block.read(
                block_info.read_start,
                block_info.read_end,
                channels=[block_info.channel]
            )
            # Squeeze to 1D (block.read returns shape (samples, channels))
            audio = audio[:, 0].astype(np.float32)
            logger.debug(f"Processing block {block_info.block_index}: {len(audio)} samples starting at {block_info.block_start_sample}")

            if self.method == "basic":
                # Basic threshold detection
                f0 = self.params.get('f0', 500)
                f1 = self.params.get('f1', self.sampling_rate / 4)
                rectify_lowpass = self.params.get('rectify_lowpass', 100)
                threshold = self.params.get('threshold', 0.01)
                min_peak = self.params.get('min_peak', False)

                filtered, ampenv = filter_and_ampenv(audio, self.sampling_rate, f0, f1, rectify_lowpass)

                intervals = threshold_events(
                    ampenv,
                    threshold,
                    sampling_rate=self.sampling_rate,
                    ignore_width=self.params.get('ignore_width', 0.01),
                    min_size=self.params.get('min_size', 0.01),
                    fuse_duration=self.params.get('fuse_duration', 0.01),
                    min_peak=min_peak
                )

            elif self.method == "advanced":
                # Advanced multi-band segmentation
                onsets, offsets = advanced_seg(audio, self.sampling_rate, self.params)
                if len(onsets) > 0:
                    intervals = np.column_stack([onsets, offsets])
                else:
                    intervals = np.array([])
            else:
                raise ValueError(f"Unknown method: {self.method}")

            n_intervals = len(intervals) if len(intervals.shape) > 0 and intervals.shape[0] > 0 else 0
            logger.debug(f"Block {block_info.block_index}: detected {n_intervals} intervals")

            return BlockSegmentResult(
                block_index=block_info.block_index,
                block_start_sample=block_info.block_start_sample,
                intervals=intervals
            )

        except Exception as e:
            logger.exception(f"Error processing block {block_info.block_index}")
            return BlockSegmentResult(
                block_index=block_info.block_index,
                block_start_sample=block_info.block_start_sample,
                intervals=np.array([]),
                error=str(e)
            )

    def run(self):
        try:
            results = []
            total_blocks = len(self.blocks_info)

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_block = {
                    executor.submit(self._process_block, block_info): block_info.block_index
                    for block_info in self.blocks_info
                }

                completed = 0
                for future in as_completed(future_to_block):
                    block_idx = future_to_block[future]
                    completed += 1
                    self.progress.emit(completed, total_blocks, f"Processed block {completed}/{total_blocks}")

                    result = future.result()
                    results.append(result)

            # Sort results by block index to maintain order
            results.sort(key=lambda r: r.block_index)
            self.finished.emit(results)

        except Exception as e:
            logger.exception("Error during parallel segmentation")
            self.error.emit(str(e))


class AutoSegmentPanel(widgets.QWidget):
    """Panel widget for AutoSegment configuration and controls."""

    segmentRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = widgets.QLabel("Auto Segment")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Method selector
        method_layout = widgets.QHBoxLayout()
        method_label = widgets.QLabel("Method:")
        self.method_selector = widgets.QComboBox()
        self.method_selector.addItems(["WhisperSeg", "Basic", "Advanced"])
        self.method_selector.currentTextChanged.connect(self._on_method_changed)
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_selector, 1)
        layout.addLayout(method_layout)

        # Model selector (WhisperSeg only)
        self.model_group = widgets.QWidget()
        model_layout = widgets.QHBoxLayout()
        model_layout.setContentsMargins(0, 0, 0, 0)
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
        self.model_group.setLayout(model_layout)
        layout.addWidget(self.model_group)

        # Detect mode info label (Basic/Advanced only)
        self.detect_info_label = widgets.QLabel("")
        self.detect_info_label.setWordWrap(True)
        self.detect_info_label.setStyleSheet("color: #666; font-size: 11px;")
        self.detect_info_label.hide()
        layout.addWidget(self.detect_info_label)

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

        # Use scrollbar selection button
        self.use_scrollbar_selection_button = widgets.QPushButton("Use Scrollbar Selection")
        self.use_scrollbar_selection_button.setToolTip("Set time range from scrollbar selection (Shift+drag on scrollbar)")
        layout.addWidget(self.use_scrollbar_selection_button)

        # Parallel processing checkbox
        self.parallel_checkbox = widgets.QCheckBox("Process blocks in parallel")
        self.parallel_checkbox.setChecked(True)
        self.parallel_checkbox.setToolTip("Process each audio block in a separate thread for faster segmentation")
        layout.addWidget(self.parallel_checkbox)

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

    def _on_method_changed(self, method: str):
        """Update UI based on selected method."""
        is_whisperseg = (method == "WhisperSeg")
        self.model_group.setVisible(is_whisperseg)
        self.detect_info_label.setVisible(not is_whisperseg)

        if not is_whisperseg:
            self.detect_info_label.setText(
                f"Using {method} detection settings from Detect plugin.\n"
                "Configure frequency bands and threshold there first."
            )

    def set_status(self, message: str, is_error: bool = False):
        self.status_label.setText(message)
        if is_error:
            self.status_label.setStyleSheet("color: red;")
        else:
            self.status_label.setStyleSheet("color: gray; font-style: italic;")

    def set_processing(self, processing: bool, determinate: bool = False, total: int = 0):
        self.segment_button.setEnabled(not processing)
        self.progress_bar.setVisible(processing)
        if processing:
            if determinate and total > 0:
                self.progress_bar.setRange(0, total)
                self.progress_bar.setValue(0)
            else:
                self.progress_bar.setRange(0, 0)  # Indeterminate
            self.set_status("Processing...")

    def set_progress(self, current: int, total: int):
        """Update progress bar value."""
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)


class AutoSegmentPlugin(BasePlugin):
    """Plugin for automatic audio segmentation using multiple methods."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.worker: Optional[WhisperSegWorker] = None
        self.parallel_worker: Optional[ParallelSegmentWorker] = None
        self._pending_channel = None
        self._pending_start_sample = None
        self._pending_end_sample = None

        self.init_ui()
        self.init_actions()
        self.connect_events()

    def init_ui(self):
        self.panel = AutoSegmentPanel()

        # Toolbar button
        self.toolbar_button = widgets.QPushButton("Auto Segment")
        self.toolbar_button.setToolTip("Run automatic segmentation")

    def init_actions(self):
        self.segment_action = QtGui.QAction("Run auto segmentation", self)
        self.segment_action.triggered.connect(self.on_segment_requested)

    def connect_events(self):
        self.toolbar_button.clicked.connect(self.on_segment_requested)
        self.panel.segmentRequested.connect(self.on_segment_requested)
        self.panel.use_selection_button.clicked.connect(self.on_use_selection)
        self.panel.use_scrollbar_selection_button.clicked.connect(self.on_use_scrollbar_selection)

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
        """Called when user selection changes."""
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

    def on_use_scrollbar_selection(self):
        """Set time range from scrollbar selection."""
        scrollbar = self.gui.scrollbar
        selection = scrollbar.get_selection_seconds()
        if selection:
            self.panel.start_time_input.setValue(selection[0])
            self.panel.end_time_input.setValue(selection[1])
            self.panel.set_status(f"Set range: {selection[0]:.3f}s - {selection[1]:.3f}s")
        else:
            self.panel.set_status("No scrollbar selection. Shift+drag on scrollbar to select.", is_error=True)

    def on_segment_requested(self):
        """Run segmentation with the selected method."""
        method = self.panel.method_selector.currentText()

        if method == "WhisperSeg":
            self._run_whisperseg()
        elif method == "Basic":
            self._run_basic_segmentation()
        elif method == "Advanced":
            self._run_advanced_segmentation()

    def _run_whisperseg(self):
        """Run WhisperSeg segmentation in background thread."""
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
        self._pending_end_sample = end_sample

        # Start segmentation in background thread
        self.panel.set_processing(True)

        self.worker = WhisperSegWorker(audio, sr, model_name)
        self.worker.finished.connect(self._on_whisperseg_finished)
        self.worker.error.connect(self._on_segmentation_error)
        self.worker.progress.connect(lambda msg: self.panel.set_status(msg))
        self.worker.start()

    def _run_basic_segmentation(self):
        """Run basic threshold-based segmentation."""
        channel = self.panel.channel_selector.currentData()
        start_time = self.panel.start_time_input.value()
        end_time = self.panel.end_time_input.value()

        if channel is None:
            self.panel.set_status("Please select a channel", is_error=True)
            return

        if end_time <= start_time:
            self.panel.set_status("End time must be greater than start time", is_error=True)
            return

        # Get DetectPlugin for settings
        detect_plugin = self.api.get_plugin("DetectPlugin", required=False)
        if not detect_plugin:
            self.panel.set_status("DetectPlugin not available", is_error=True)
            return

        sr = self.api.project.sampling_rate
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Get frequency band - use default wide range
        # or from current selection if available
        selection = self.api.get_fine_selection()
        if selection and selection.f0 is not None and selection.f1 is not None:
            f0, f1 = selection.f0, selection.f1
        else:
            # Default: 500Hz to half Nyquist
            f0 = 500
            f1 = sr / 4

        # Get threshold from DetectPlugin
        threshold = detect_plugin._threshold
        if threshold is None:
            # We'll compute a default per-block if needed
            threshold = 0.01  # Fallback

        # Get peak threshold if enabled
        min_peak = False
        if detect_plugin.using_peak_threshold:
            min_peak = detect_plugin._peak_threshold
            if min_peak is None:
                min_peak = 2 * threshold

        # Build params dict for worker
        params = {
            'f0': f0,
            'f1': f1,
            'rectify_lowpass': self.api.config.get("detection.rectify_lowpass", 100),
            'threshold': threshold,
            'min_peak': min_peak,
            'ignore_width': self.api.config.get("detection.ignore_width", 0.01),
            'min_size': self.api.config.get("detection.min_size", 0.01),
            'fuse_duration': self.api.config.get("detection.fuse_duration", 0.01),
        }

        # Check if parallel processing is enabled
        use_parallel = self.panel.parallel_checkbox.isChecked()

        if use_parallel:
            self._run_parallel_segmentation("basic", channel, start_sample, end_sample, params)
        else:
            # Original single-threaded processing
            try:
                self.panel.set_status("Running basic segmentation...")
                audio = self.api.project[start_sample:end_sample, channel]
                audio = audio.astype(np.float32)

                # Compute amplitude envelope
                filtered, ampenv = filter_and_ampenv(audio, sr, f0, f1, params['rectify_lowpass'])

                # Run threshold detection
                intervals = threshold_events(
                    ampenv,
                    threshold,
                    sampling_rate=sr,
                    ignore_width=params['ignore_width'],
                    min_size=params['min_size'],
                    fuse_duration=params['fuse_duration'],
                    min_peak=min_peak
                )

                # Create segments
                self._create_segments_from_intervals(intervals, channel, start_sample, end_sample)

            except Exception as e:
                logger.exception("Error during basic segmentation")
                self.panel.set_status(f"Error: {e}", is_error=True)

    def _run_advanced_segmentation(self):
        """Run advanced multi-band segmentation."""
        channel = self.panel.channel_selector.currentData()
        start_time = self.panel.start_time_input.value()
        end_time = self.panel.end_time_input.value()

        if channel is None:
            self.panel.set_status("Please select a channel", is_error=True)
            return

        if end_time <= start_time:
            self.panel.set_status("End time must be greater than start time", is_error=True)
            return

        # Get DetectPlugin for settings
        detect_plugin = self.api.get_plugin("DetectPlugin", required=False)
        if not detect_plugin:
            self.panel.set_status("DetectPlugin not available", is_error=True)
            return

        sr = self.api.project.sampling_rate
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Get parameters from DetectPlugin's advanced controls
        signal_band = detect_plugin.advanced_preview.get_signal_band()
        noise_band = detect_plugin.advanced_preview.get_noise_band()
        threshold = detect_plugin.advanced_preview.get_threshold()

        params = detect_plugin.detect_controls.advanced_panel.get_advanced_params(
            sr,
            signal_band,
            noise_band,
            threshold
        )

        # Log the params for debugging
        logger.info(f"Advanced segmentation params: signal_band={signal_band}, noise_band={noise_band}, "
                    f"threshold={threshold:.6f}, software_gain={params['software_gain']}, "
                    f"signal_gain={params['signal_gain']}, noise_gain={params['noise_gain']}")

        # Check if parallel processing is enabled
        use_parallel = self.panel.parallel_checkbox.isChecked()

        if use_parallel:
            self._run_parallel_segmentation("advanced", channel, start_sample, end_sample, params)
        else:
            # Original single-threaded processing
            try:
                self.panel.set_status("Running advanced segmentation...")
                audio = self.api.project[start_sample:end_sample, channel]
                audio = audio.astype(np.float32)

                # Run advanced segmentation
                onsets, offsets = advanced_seg(audio, sr, params)

                if len(onsets) > 0:
                    intervals = np.column_stack([onsets, offsets])
                else:
                    intervals = np.array([])

                # Create segments
                self._create_segments_from_intervals(intervals, channel, start_sample, end_sample)

            except Exception as e:
                logger.exception("Error during advanced segmentation")
                self.panel.set_status(f"Error: {e}", is_error=True)

    def _run_parallel_segmentation(self, method: str, channel: int, start_sample: int, end_sample: int, params: dict):
        """Run segmentation in parallel across blocks."""
        if self.parallel_worker is not None and self.parallel_worker.isRunning():
            self.panel.set_status("Segmentation already in progress", is_error=True)
            return

        sr = self.api.project.sampling_rate
        project = self.api.project

        # Log the params being used for debugging
        logger.info(f"Running parallel {method} segmentation with params: {params}")

        # Build block metadata for workers to read files themselves
        # This avoids loading all data into memory before parallelizing
        blocks_info = []
        block_idx = 0

        try:
            for (i0, i1), block in project.iter_blocks():
                block_start = int(i0)
                block_end = int(i1)

                # Skip blocks outside our range
                if block_end <= start_sample:
                    continue
                if block_start >= end_sample:
                    break

                # Calculate read range within this block
                read_start_in_block = max(start_sample - block_start, 0)
                read_end_in_block = min(end_sample - block_start, block.frames)

                # Calculate the absolute start sample for results
                # This is where in the project this chunk starts
                abs_start = block_start + read_start_in_block

                blocks_info.append(BlockReadInfo(
                    block_index=block_idx,
                    block_start_sample=abs_start,
                    block=block,
                    channel=channel,
                    read_start=read_start_in_block,
                    read_end=read_end_in_block
                ))
                block_idx += 1

        except Exception as e:
            self.panel.set_status(f"Error building block info: {e}", is_error=True)
            return

        if not blocks_info:
            self.panel.set_status("No audio blocks found in range", is_error=True)
            return

        # Store pending info
        self._pending_channel = channel
        self._pending_start_sample = start_sample
        self._pending_end_sample = end_sample

        # Start parallel worker
        num_blocks = len(blocks_info)
        self.panel.set_processing(True, determinate=True, total=num_blocks)
        self.panel.set_status(f"Processing {num_blocks} blocks in parallel...")

        self.parallel_worker = ParallelSegmentWorker(
            blocks_info=blocks_info,
            sampling_rate=sr,
            method=method,
            params=params,
            max_workers=min(16, num_blocks)
        )
        self.parallel_worker.finished.connect(self._on_parallel_finished)
        self.parallel_worker.error.connect(self._on_segmentation_error)
        self.parallel_worker.progress.connect(self._on_parallel_progress)
        self.parallel_worker.start()

    def _on_parallel_progress(self, current: int, total: int, message: str):
        """Update progress during parallel processing."""
        self.panel.set_progress(current, total)
        self.panel.set_status(message)

    def _on_parallel_finished(self, results: List[BlockSegmentResult]):
        """Handle completed parallel segmentation."""
        self.panel.set_processing(False)

        # Get source for the channel
        source = self._get_or_create_source_for_channel(self._pending_channel)
        if source is None:
            self.panel.set_status("Could not find or create source for channel", is_error=True)
            return

        # Delete existing segments in the range first
        segment_plugin = self.api.get_plugin("SegmentPlugin", required=False)
        if segment_plugin:
            segment_plugin.delete_segments_between(
                self.api.make_project_index(self._pending_start_sample),
                self.api.make_project_index(self._pending_end_sample),
                source
            )

        # Collect all intervals from all blocks, converting to absolute project indices
        all_segments = []
        errors = []

        for result in results:
            if result.error:
                errors.append(f"Block {result.block_index}: {result.error}")
                continue

            if len(result.intervals) > 0:
                for interval in result.intervals:
                    # interval is relative to block start, add block_start_sample to get absolute
                    abs_start = result.block_start_sample + int(interval[0])
                    abs_end = result.block_start_sample + int(interval[1])
                    all_segments.append((
                        self.api.make_project_index(abs_start),
                        self.api.make_project_index(abs_end),
                        source
                    ))

        if errors:
            logger.warning(f"Errors during parallel segmentation: {errors}")

        if not all_segments:
            msg = "No segments detected (existing segments in range were deleted)"
            if errors:
                msg += f". Errors in {len(errors)} blocks."
            self.panel.set_status(msg)
            self.parallel_worker = None
            return

        # Create segments using SegmentPlugin
        try:
            if segment_plugin:
                segment_plugin.create_segments_batch(all_segments, skip_delete_check=True)
                msg = f"Created {len(all_segments)} segments from {len(results)} blocks"
                if errors:
                    msg += f" ({len(errors)} blocks had errors)"
                self.panel.set_status(msg)
            else:
                self.panel.set_status("SegmentPlugin not available", is_error=True)
        except Exception as e:
            logger.exception("Error creating segments")
            self.panel.set_status(f"Error creating segments: {e}", is_error=True)

        self.parallel_worker = None

    def _create_segments_from_intervals(self, intervals: np.ndarray, channel: int, start_offset: int, end_offset: int):
        """Create segments from detected intervals, deleting existing segments in the range first."""
        # Get source for the channel
        source = self._get_or_create_source_for_channel(channel)
        if source is None:
            self.panel.set_status("Could not find or create source for channel", is_error=True)
            return

        # Delete existing segments in the range first
        segment_plugin = self.api.get_plugin("SegmentPlugin", required=False)
        if segment_plugin:
            segment_plugin.delete_segments_between(
                self.api.make_project_index(start_offset),
                self.api.make_project_index(end_offset),
                source
            )

        if len(intervals) == 0:
            self.panel.set_status("No segments detected (existing segments in range were deleted)")
            return

        # Convert to absolute project indices (using ProjectIndex)
        absolute_segments = [
            (
                self.api.make_project_index(start_offset + int(interval[0])),
                self.api.make_project_index(start_offset + int(interval[1])),
                source
            )
            for interval in intervals
        ]

        # Create segments using SegmentPlugin
        try:
            segment_plugin = self.api.get_plugin("SegmentPlugin", required=False)
            if segment_plugin:
                segment_plugin.create_segments_batch(absolute_segments, skip_delete_check=True)
                self.panel.set_status(f"Created {len(intervals)} segments")
            else:
                self.panel.set_status("SegmentPlugin not available", is_error=True)
        except Exception as e:
            logger.exception("Error creating segments")
            self.panel.set_status(f"Error creating segments: {e}", is_error=True)

    def _on_whisperseg_finished(self, segments: list):
        """Handle completed WhisperSeg segmentation."""
        self.panel.set_processing(False)

        # Get source for the channel
        source = self._get_or_create_source_for_channel(self._pending_channel)
        if source is None:
            self.panel.set_status("Could not find or create source for channel", is_error=True)
            return

        # Delete existing segments in the range first
        segment_plugin = self.api.get_plugin("SegmentPlugin", required=False)
        if segment_plugin:
            segment_plugin.delete_segments_between(
                self.api.make_project_index(self._pending_start_sample),
                self.api.make_project_index(self._pending_end_sample),
                source
            )

        if not segments:
            self.panel.set_status("No segments detected (existing segments in range were deleted)")
            self.worker = None
            return

        # Convert relative sample indices to absolute project indices (using ProjectIndex)
        start_offset = self._pending_start_sample
        absolute_segments = [
            (
                self.api.make_project_index(start_offset + start),
                self.api.make_project_index(start_offset + stop),
                source
            )
            for start, stop in segments
        ]

        # Create segments using SegmentPlugin
        try:
            if segment_plugin:
                segment_plugin.create_segments_batch(absolute_segments, skip_delete_check=True)
                self.panel.set_status(f"Created {len(segments)} segments")
            else:
                self.panel.set_status("SegmentPlugin not available", is_error=True)
        except Exception as e:
            logger.exception("Error creating segments")
            self.panel.set_status(f"Error creating segments: {e}", is_error=True)

        self.worker = None

    def _on_segmentation_error(self, error_msg: str):
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
        return self.api.create_source(f"AutoSeg Ch{channel}", channel)

    def plugin_toolbar_items(self):
        return [self.toolbar_button]

    def add_plugin_menu(self, menu_parent):
        menu = menu_parent.addMenu("&Auto Segment")
        menu.addAction(self.segment_action)
        return menu

    def plugin_panel_widget(self):
        return [self.panel]

    def setup_plugin_shortcuts(self):
        self.segment_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+A"))


# Keep backward compatibility alias
WhisperSegPlugin = AutoSegmentPlugin
