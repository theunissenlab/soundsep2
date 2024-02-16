import logging

import PyQt6.QtWidgets as widgets
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtGui
from PyQt6.QtCore import Qt

from soundsep.api import SignalTooShort
from soundsep.core.base_plugin import BasePlugin


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


class ThresholdControls(widgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_ui()

    def init_ui(self):
        layout = widgets.QHBoxLayout()

        self.use_peak_threshold_button = widgets.QCheckBox("Use peak threshold")
        self.reset_thresholds_button = widgets.QPushButton("Reset Thresholds")
        layout.addWidget(self.use_peak_threshold_button)
        layout.addWidget(self.reset_thresholds_button)
        self.setLayout(layout)


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
        self.threshold_preview_plot.setBounds([0, None])
        self.gui.ui.previewPlot.addItem(self.threshold_preview_plot)

        self.threshold_controls = ThresholdControls()
        preview_box_layout = self.gui.ui.previewBox.layout()
        preview_box_layout.insertWidget(0, self.threshold_controls)

    def init_actions(self):
        self.detect_action = QtGui.QAction("Detect in selection", self)
        self.detect_action.triggered.connect(self.on_detect_activated)

    def connect_events(self):
        self.button = widgets.QPushButton("Detect")
        self.button.clicked.connect(self.on_detect_activated)

        self.threshold_preview_plot.sigDragged.connect(self.on_threshold_dragged)
        self.api.selectionChanged.connect(self.on_selection_changed)

        self.threshold_controls.reset_thresholds_button.clicked.connect(self.reset_thresholds)
        self.threshold_controls.use_peak_threshold_button.clicked.connect(self.toggle_peak_threshold)

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
        return self.threshold_controls.use_peak_threshold_button.checkState() == Qt.CheckState.Checked

    def reset_thresholds(self):
        self._threshold = None
        self._peak_threshold = None
        self.on_selection_changed()

    def toggle_peak_threshold(self):
        if self.using_peak_threshold:
            self.peak_threshold_line = pg.InfiniteLine(pos=self.threshold_preview_plot.pos().y() * 2, angle=0, movable=True)
            self.peak_threshold_line.setCursor(Qt.CursorShape.SplitVCursor)
            self.peak_threshold_line.setPen(pg.mkPen((200, 20, 20), width=3))
            self.peak_threshold_line.setBounds([0, None])
            self.peak_threshold_line.sigDragged.connect(self.on_peak_threshold_dragged)
            self.gui.ui.previewPlot.addItem(self.peak_threshold_line)
        else:
            self.gui.ui.previewPlot.removeItem(self.peak_threshold_line)
            self._peak_threshold = None

    def on_threshold_dragged(self, line):
        self._threshold = line.pos().y()
        if self.using_peak_threshold:
            self.peak_threshold_line.setBounds([self._threshold, None])
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
        else:
            # Caching get_signals and filter_and_ampenv would be nice...
            # we call it back to back here and on detect
            t, signal = self.api.get_signal(selection.x0, selection.x1)
            signal = signal[:, selection.source.channel]
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

        self.api.plugins["SegmentPlugin"].delete_segments(
            selection.x0,
            selection.x1,
            selection.source
        )

        t, signal = self.api.get_signal(selection.x0, selection.x1)
        signal = signal[:, selection.source.channel]
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
