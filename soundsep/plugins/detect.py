import logging

import PyQt6.QtWidgets as widgets
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, pyqtSignal

from soundsep.api import SignalTooShort
from soundsep.core.ampenv import advanced_filter_and_ampenv, advanced_seg
from soundsep.core.base_plugin import BasePlugin
from soundsep.widgets.preview_plot import AdvancedPreviewWidget


logger = logging.getLogger(__name__)


def threshold_events(
        signal,
        threshold,
        polarity=1,
        sampling_rate=1,
        ignore_width=None,
        min_size=1,
        fuse_duration=0,
        min_peak=False,
    ) -> np.ndarray:
    """Detect intervals crossing a threshold

    Arguments
    ----------
    signal : np.ndarray
        Array of shape (n,) where n is the length of the signal to be thresholded
        e.g. an amplitude envelope
    threshold : float
        Floating point threshold on the signal
    polarity : -1 or 1
        Detect threshold crossings in the negative (-1) or positive (1) direction
    sampling_rate : int
        Number of samples per second in signal
    ignore_width : float
        Threshold crossings that are shorter than ignore_width (in seconds) are
        not considered when determining thresholded intervals
    min_size : float
        Reject all intervals that come out to be shorter than min_size (in seconds)
    fuse_duration : float
        Intervals initally detected that occur within fuse_duration (seconds)
        of each other will be merged into one period
    min_peak : Optional[float]
        If provided, only intervals with a peak value exceeding min_peak will be included
    """
    if polarity not in (-1, 1):
        raise ValueError("Polarity must equal +/- 1")

    if isinstance(threshold, np.ndarray):
        starts_on = (polarity * signal[0] >= polarity * threshold)[0]
    else:
        starts_on = (polarity * signal[0] >= polarity * threshold)

    crossings = np.diff((polarity * signal >= polarity * threshold).astype(int))
    interval_starts = np.where(crossings > 0)[0] + 1
    interval_stops = np.where(crossings < 0)[0] + 1

    if starts_on:
        interval_starts = np.concatenate([[0], interval_starts])

    if len(interval_stops) < len(interval_starts):
        interval_stops = np.concatenate([interval_stops, [len(signal)]])

    # Ignore events that are too short
    intervals = np.array([
        (i, j) for i, j in zip(interval_starts, interval_stops)
        if (not ignore_width or ((j - i) / sampling_rate) > ignore_width)
    ])
    if not len(intervals):
        return np.array([])

    gaps = (intervals[1:, 0] - intervals[:-1, 1]) / sampling_rate
    gaps = np.concatenate([gaps, [np.inf]])

    fused_intervals = []
    current_interval_start = None
    for (i, j), gap in zip(intervals, gaps):
        if current_interval_start is None:
            current_interval_start = i
        if gap > fuse_duration:
            fused_intervals.append((current_interval_start, j))
            current_interval_start = None

    if min_peak:
        fused_intervals = [(i, j) for i, j in fused_intervals if np.max(signal[i:j]) > min_peak]

    return np.array([(i, j) for i, j in fused_intervals if ((j - i) / sampling_rate) >= min_size])


class BasicDetectControls(widgets.QWidget):
    """Controls shown above the preview plot (mode selector + basic threshold controls)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_ui()

    def init_ui(self):
        layout = widgets.QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)

        layout.addWidget(widgets.QLabel("Mode:"))
        self.mode_selector = widgets.QComboBox()
        self.mode_selector.addItems(["Basic", "Advanced"])
        layout.addWidget(self.mode_selector)

        layout.addSpacing(20)

        self.use_peak_threshold_button = widgets.QCheckBox("Use peak threshold")
        self.reset_thresholds_button = widgets.QPushButton("Reset")
        layout.addWidget(self.use_peak_threshold_button)
        layout.addWidget(self.reset_thresholds_button)
        layout.addStretch()

        self.setLayout(layout)


class AdvancedControlsPanel(widgets.QWidget):
    """Side panel with advanced segmentation parameters (non-frequency params only)"""

    parameterChanged = pyqtSignal()  # Emitted when any parameter changes

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_ui()

    def init_ui(self):
        main_layout = widgets.QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = widgets.QLabel("Parameters")
        title.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(title)

        # Form layout for parameters
        form_layout = widgets.QFormLayout()
        form_layout.setContentsMargins(0, 5, 0, 0)
        form_layout.setRowWrapPolicy(widgets.QFormLayout.RowWrapPolicy.WrapAllRows)

        # Software gain
        self.software_gain_spin = widgets.QDoubleSpinBox()
        self.software_gain_spin.setRange(0.001, 1000)
        self.software_gain_spin.setDecimals(3)
        self.software_gain_spin.setValue(1.0)
        self.software_gain_spin.valueChanged.connect(self._on_param_changed)
        form_layout.addRow("Gain:", self.software_gain_spin)

        # Signal and Noise gains
        self.signal_gain_spin = widgets.QDoubleSpinBox()
        self.signal_gain_spin.setRange(0, 1000)
        self.signal_gain_spin.setDecimals(3)
        self.signal_gain_spin.setValue(1.0)
        self.signal_gain_spin.valueChanged.connect(self._on_param_changed)
        self.noise_gain_spin = widgets.QDoubleSpinBox()
        self.noise_gain_spin.setRange(0, 1000)
        self.noise_gain_spin.setDecimals(3)
        self.noise_gain_spin.setValue(1.0)
        self.noise_gain_spin.valueChanged.connect(self._on_param_changed)
        form_layout.addRow("Sig Gain:", self.signal_gain_spin)
        form_layout.addRow("Noise Gain:", self.noise_gain_spin)

        # Smoothing
        self.smooth_ms_spin = widgets.QDoubleSpinBox()
        self.smooth_ms_spin.setRange(0.1, 1000)
        self.smooth_ms_spin.setDecimals(1)
        self.smooth_ms_spin.setSuffix(" ms")
        self.smooth_ms_spin.setValue(2.0)
        self.smooth_ms_spin.valueChanged.connect(self._on_param_changed)
        form_layout.addRow("Smooth:", self.smooth_ms_spin)

        # Duration parameters
        self.min_gap_spin = widgets.QDoubleSpinBox()
        self.min_gap_spin.setRange(0, 1)
        self.min_gap_spin.setDecimals(3)
        self.min_gap_spin.setSuffix(" s")
        self.min_gap_spin.setValue(0.01)
        self.min_dur_spin = widgets.QDoubleSpinBox()
        self.min_dur_spin.setRange(0, 1)
        self.min_dur_spin.setDecimals(3)
        self.min_dur_spin.setSuffix(" s")
        self.min_dur_spin.setValue(0.01)
        self.max_dur_spin = widgets.QDoubleSpinBox()
        self.max_dur_spin.setRange(0, 100)
        self.max_dur_spin.setDecimals(2)
        self.max_dur_spin.setSuffix(" s")
        self.max_dur_spin.setValue(10.0)
        form_layout.addRow("Min Gap:", self.min_gap_spin)
        form_layout.addRow("Min Dur:", self.min_dur_spin)
        form_layout.addRow("Max Dur:", self.max_dur_spin)

        main_layout.addLayout(form_layout)
        main_layout.addStretch()

        self.setLayout(main_layout)
        self.setFixedWidth(130)

    def _on_param_changed(self):
        self.parameterChanged.emit()

    def get_advanced_params(self, sampling_rate: int, signal_band: tuple, noise_band: tuple, threshold: float) -> dict:
        """Build the params dict for advanced_seg.

        Args:
            sampling_rate: Audio sampling rate
            signal_band: (low, high) frequencies for signal band
            noise_band: (low, high) frequencies for noise band
            threshold: Detection threshold value
        """
        return {
            'fs': sampling_rate,
            'software_gain': self.software_gain_spin.value(),
            'signal_low': signal_band[0],
            'signal_high': signal_band[1],
            'noise_low': noise_band[0],
            'noise_high': noise_band[1],
            'signal_gain': self.signal_gain_spin.value(),
            'noise_gain': self.noise_gain_spin.value(),
            'smooth_ms': self.smooth_ms_spin.value(),
            'threshold': threshold,
            'min_gap_sec': self.min_gap_spin.value(),
            'min_dur_sec': self.min_dur_spin.value(),
            'max_dur_sec': self.max_dur_spin.value(),
        }

    def load_from_config(self, config: dict):
        """Load advanced parameter values from config"""
        self.software_gain_spin.setValue(config.get("detection.advanced.software_gain", 1.0))
        self.signal_gain_spin.setValue(config.get("detection.advanced.signal_gain", 1.0))
        self.noise_gain_spin.setValue(config.get("detection.advanced.noise_gain", 1.0))
        self.smooth_ms_spin.setValue(config.get("detection.advanced.smooth_ms", 2.0))
        self.min_gap_spin.setValue(config.get("detection.advanced.min_gap_sec", 0.01))
        self.min_dur_spin.setValue(config.get("detection.advanced.min_dur_sec", 0.01))
        self.max_dur_spin.setValue(config.get("detection.advanced.max_dur_sec", 10.0))


class DetectControls(widgets.QWidget):
    """Wrapper that provides backward-compatible interface"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.basic_controls = BasicDetectControls()
        self.advanced_panel = AdvancedControlsPanel()

        # Expose controls for external access
        self.mode_selector = self.basic_controls.mode_selector
        self.use_peak_threshold_button = self.basic_controls.use_peak_threshold_button
        self.reset_thresholds_button = self.basic_controls.reset_thresholds_button

    def is_advanced_mode(self) -> bool:
        return self.mode_selector.currentText() == "Advanced"

    def load_from_config(self, config: dict):
        self.advanced_panel.load_from_config(config)


class DetectPlugin(BasePlugin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_ui()
        self.init_actions()
        self.connect_events()

        self._threshold = None
        self._peak_threshold = None

    def init_ui(self):
        self.threshold_preview_plot = pg.InfiniteLine(pos=0, angle=0, movable=True)
        self.threshold_preview_plot.setCursor(Qt.CursorShape.SplitVCursor)
        self.threshold_preview_plot.setPen(pg.mkPen((20, 200, 20), width=3))
        self.threshold_preview_plot.setBounds([0, 1e10])
        self.gui.ui.previewPlot.addItem(self.threshold_preview_plot)

        self.detect_controls = DetectControls()
        preview_box_layout = self.gui.ui.previewBox.layout()

        # Add basic controls above the plot
        preview_box_layout.insertWidget(0, self.detect_controls.basic_controls)

        # Create a horizontal wrapper for advanced panel + plots
        # Get the current preview plot widget (basic mode)
        preview_plot_widget = self.gui.ui.previewPlot

        # Create horizontal container
        self.preview_container = widgets.QWidget()
        container_layout = widgets.QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(5)

        # Add advanced panel (hidden by default)
        self.detect_controls.advanced_panel.setVisible(False)
        container_layout.addWidget(self.detect_controls.advanced_panel)

        # Create stacked widget to switch between basic and advanced preview
        self.preview_stack = widgets.QStackedWidget()

        # Remove plot from original layout and add to stack
        preview_box_layout.removeWidget(preview_plot_widget)
        self.preview_stack.addWidget(preview_plot_widget)  # index 0 = basic

        # Create advanced preview widget (spectrogram + ampenv)
        self.advanced_preview = AdvancedPreviewWidget()
        self.preview_stack.addWidget(self.advanced_preview)  # index 1 = advanced

        # Load default frequency bands from config
        self.advanced_preview.set_signal_band(
            self.api.config.get("detection.advanced.signal_low", 2000),
            self.api.config.get("detection.advanced.signal_high", 10000)
        )
        self.advanced_preview.set_noise_band(
            self.api.config.get("detection.advanced.noise_low", 500),
            self.api.config.get("detection.advanced.noise_high", 1500)
        )
        self.advanced_preview.set_threshold(
            self.api.config.get("detection.advanced.threshold", 0.1)
        )

        container_layout.addWidget(self.preview_stack, 1)  # stretch factor 1

        self.preview_container.setLayout(container_layout)
        preview_box_layout.addWidget(self.preview_container)

        # Load config values into advanced controls
        self.detect_controls.load_from_config(self.api.config)

    def init_actions(self):
        self.detect_action = QtGui.QAction("Detect in selection", self)
        self.detect_action.triggered.connect(self.on_detect_activated)

    def connect_events(self):
        self.button = widgets.QPushButton("Detect")
        self.button.clicked.connect(self.on_detect_activated)

        self.threshold_preview_plot.sigDragged.connect(self.on_threshold_dragged)
        self.api.selectionChanged.connect(self.on_selection_changed)

        self.detect_controls.reset_thresholds_button.clicked.connect(self.reset_thresholds)
        self.detect_controls.use_peak_threshold_button.clicked.connect(self.toggle_peak_threshold)
        self.detect_controls.mode_selector.currentTextChanged.connect(self.on_mode_changed)

        # Connect advanced preview signals for real-time recalculation
        self.advanced_preview.signalBandChanged.connect(self._on_advanced_param_changed)
        self.advanced_preview.noiseBandChanged.connect(self._on_advanced_param_changed)
        self.advanced_preview.thresholdChanged.connect(self._on_threshold_changed)
        self.detect_controls.advanced_panel.parameterChanged.connect(self._on_advanced_param_changed)

    def on_mode_changed(self, mode: str):
        """Update preview when detection mode changes"""
        is_advanced = (mode == "Advanced")
        self.detect_controls.advanced_panel.setVisible(is_advanced)

        # Switch between basic (index 0) and advanced (index 1) preview
        self.preview_stack.setCurrentIndex(1 if is_advanced else 0)

        self.on_selection_changed()

    def _on_advanced_param_changed(self, *args):
        """Recalculate ampenv when any advanced parameter changes"""
        if self.detect_controls.is_advanced_mode():
            self._update_advanced_ampenv()

    def _on_threshold_changed(self, threshold: float):
        """Update threshold value (no full recalculation needed)"""
        # Threshold line is already updated via the drag
        pass

    def _update_advanced_ampenv(self):
        """Compute and display the advanced ampenv based on current parameters"""
        selection = self.api.get_fine_selection()
        if not selection:
            return

        try:
            t, signal = self.api.get_signal(selection.x0, selection.x1)
            signal = signal[:, selection.source.channel]

            # Get parameters from UI
            signal_band = self.advanced_preview.get_signal_band()
            noise_band = self.advanced_preview.get_noise_band()
            threshold = self.advanced_preview.get_threshold()

            params = self.detect_controls.advanced_panel.get_advanced_params(
                self.api.project.sampling_rate,
                signal_band,
                noise_band,
                threshold
            )

            # Compute ampenv
            ampenv = advanced_filter_and_ampenv(signal, self.api.project.sampling_rate, params)

            # Update the ampenv plot
            self.advanced_preview.set_ampenv_data(t, ampenv)

        except Exception as e:
            logger.debug("Error computing advanced ampenv: {}".format(e))

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

    @property
    def using_peak_threshold(self):
        return self.detect_controls.use_peak_threshold_button.checkState() == Qt.CheckState.Checked

    def reset_thresholds(self):
        self._threshold = None
        self._peak_threshold = None
        self.on_selection_changed()

    def toggle_peak_threshold(self):
        if self.using_peak_threshold:
            self.peak_threshold_line = pg.InfiniteLine(pos=self.threshold_preview_plot.pos().y() * 2, angle=0, movable=True)
            self.peak_threshold_line.setCursor(Qt.CursorShape.SplitVCursor)
            self.peak_threshold_line.setPen(pg.mkPen((200, 20, 20), width=3))
            self.peak_threshold_line.setBounds([0, 1e10])
            self.peak_threshold_line.sigDragged.connect(self.on_peak_threshold_dragged)
            self.gui.ui.previewPlot.addItem(self.peak_threshold_line)
        else:
            self.gui.ui.previewPlot.removeItem(self.peak_threshold_line)
            self._peak_threshold = None

    def on_threshold_dragged(self, line):
        self._threshold = line.pos().y()
        if self.using_peak_threshold:
            self.peak_threshold_line.setBounds([self._threshold, 1e10])
            if self._peak_threshold is None:
                self.peak_threshold_line.setValue(2 * self._threshold)

    def on_peak_threshold_dragged(self, line):
        self._peak_threshold = line.pos().y()

    def on_selection_changed(self):
        """Update the preview plot with an ampenv"""
        selection = self.api.get_fine_selection()
        if not selection:
            self.threshold_preview_plot.setValue(0)
            if self.using_peak_threshold:
                self.peak_threshold_line.setValue(0)
            # Clear advanced preview
            self.advanced_preview.clear()
        else:
            # Caching get_signals and filter_and_ampenv would be nice...
            # we call it back to back here and on detect
            t, signal = self.api.get_signal(selection.x0, selection.x1)
            signal = signal[:, selection.source.channel]

            if self.detect_controls.is_advanced_mode():
                # Set spectrogram data
                t_offset = t[0] if len(t) > 0 else 0
                self.advanced_preview.set_spectrogram_data(
                    signal,
                    self.api.project.sampling_rate,
                    t_offset
                )

                # Compute and display ampenv
                self._update_advanced_ampenv()
            else:
                # Use basic ampenv calculation
                try:
                    filtered, ampenv = self.api.filter_and_ampenv(signal, selection.f0, selection.f1)
                except SignalTooShort:
                    logger.debug("Signal was too short for ampenv: {}".format(signal.size))
                    return

                threshold = self.compute_threshold(signal, ampenv)
                self.threshold_preview_plot.setValue(threshold)

                if self.using_peak_threshold:
                    peak_threshold = self.compute_peak_threshold(signal, ampenv)
                    self.peak_threshold_line.setValue(peak_threshold)

    def compute_threshold(self, signal, ampenv) -> float:
        return self._threshold or 0.5 * np.mean(np.abs(ampenv))

    def compute_peak_threshold(self, signal, ampenv) -> float:
        """Peak threshold defaults to twice threshold"""
        return self._peak_threshold or 2 * self.compute_threshold(signal, ampenv)

    def on_detect_activated(self):
        # TODO: throttle this function so it can't be called non-stop (i.e. if shortcut held down)
        selection = self.api.get_fine_selection()
        if not selection:
            return

        self.api.plugins["SegmentPlugin"].delete_segments_between(
            selection.x0,
            selection.x1,
            selection.source
        )

        t, signal = self.api.get_signal(selection.x0, selection.x1)
        signal = signal[:, selection.source.channel]

        if self.detect_controls.is_advanced_mode():
            # Get parameters from the advanced preview widget
            signal_band = self.advanced_preview.get_signal_band()
            noise_band = self.advanced_preview.get_noise_band()
            threshold = self.advanced_preview.get_threshold()

            params = self.detect_controls.advanced_panel.get_advanced_params(
                self.api.project.sampling_rate,
                signal_band,
                noise_band,
                threshold
            )

            # Use advanced segmentation
            onsets, offsets = advanced_seg(signal, self.api.project.sampling_rate, params)
            intervals = np.column_stack([onsets, offsets]) if len(onsets) > 0 else np.array([])
        else:
            # Use basic threshold detection
            filtered, ampenv = self.api.filter_and_ampenv(signal, selection.f0, selection.f1)
            threshold = self.compute_threshold(signal, ampenv)

            intervals = threshold_events(
                ampenv,
                threshold,
                sampling_rate=self.api.project.sampling_rate,
                ignore_width=self.api.config.get("detection.ignore_width", 0.01),
                min_size=self.api.config.get("detection.min_size", 0.01),
                fuse_duration=self.api.config.get("detection.fuse_duration", 0.01),
                min_peak=self.using_peak_threshold and self.compute_peak_threshold(signal, ampenv)
            )

        self.api.plugins["SegmentPlugin"].create_segments_batch([
            (
                selection.x0 + int(interval0),
                selection.x0 + int(interval1),
                selection.source
            ) for interval0, interval1 in intervals
        ])

    def plugin_toolbar_items(self):
        return [self.button]

    def add_plugin_menu(self, menu_parent):
        menu = menu_parent.addMenu("&Detect")
        menu.addAction(self.detect_action)
        return menu

    def plugin_panel_widget(self):
        return []

    def setup_plugin_shortcuts(self):
        self.detect_action.setShortcut(QtGui.QKeySequence("W"))
