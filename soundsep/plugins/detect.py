import logging

import PyQt5.QtWidgets as widgets
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui
from PyQt5.QtCore import Qt

from soundsep.core.base_plugin import BasePlugin


logger = logging.getLogger(__name__)


def threshold_events(
        signal,
        threshold,
        polarity=1,
        sampling_rate=1,
        ignore_width=None,
        min_size=1,
        fuse_duration=0
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
    """
    if polarity not in (-1, 1):
        raise ValueError("Polarity must equal +/- 1")

    if isinstance(threshold, np.ndarray):
        starts_on = (polarity * signal[0] >= polarity * threshold)[0]
    else:
        starts_on = (polarity * signal[0] >= polarity * threshold)

    crossings = np.diff((polarity * signal >= polarity * threshold).astype(np.int))
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

    return np.array([(i, j) for i, j in fused_intervals if ((j - i) / sampling_rate) >= min_size])


class DetectPlugin(BasePlugin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_ui()
        self.init_actions()
        self.connect_events()

        self._threshold = None

    def init_ui(self):
        self.threshold_preview_plot = pg.InfiniteLine(pos=0, angle=0, movable=True)
        self.threshold_preview_plot.setCursor(Qt.SplitVCursor)
        self.threshold_preview_plot.setPen(pg.mkPen((20, 20, 20), width=3))
        self.gui.preview_plot_widget.addItem(self.threshold_preview_plot)

    def init_actions(self):
        self.detect_action = widgets.QAction("Detect in selection", self)
        self.detect_action.triggered.connect(self.on_detect_activated)

    def connect_events(self):
        self.button = widgets.QPushButton("Detect")
        self.button.clicked.connect(self.on_detect_activated)

        self.threshold_preview_plot.sigDragged.connect(self.on_threshold_dragged)
        self.api.selectionChanged.connect(self.on_selection_changed)

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

    def on_threshold_dragged(self, line):
        self._threshold = line.pos().y()

    def on_selection_changed(self):
        """Update the preview plot with an ampenv"""
        selection = self.api.get_selection()
        if not selection:
            self.threshold_preview_plot.setValue(0)
        else:
            # Caching get_signals and filter_and_ampenv would be nice...
            # we call it back to back here and on detect
            t, signal = self.api.get_signal(selection.x0, selection.x1)
            signal = signal[:, selection.source.channel]
            filtered, ampenv = self.api.filter_and_ampenv(signal, selection.f0, selection.f1)
            threshold = self.compute_threshold(signal, ampenv)
            self.threshold_preview_plot.setValue(threshold)

    def compute_threshold(self, signal, ampenv) -> float:
        return self._threshold or 0.5 * np.mean(np.abs(ampenv))

    def on_detect_activated(self):
        selection = self.api.get_selection()
        if not selection:
            return

        self.api.plugins()["SegmentPlugin"].delete_segments(
            selection.x0,
            selection.x1,
            selection.source
        )

        t, signal = self.api.get_signal(selection.x0, selection.x1)
        signal = signal[:, selection.source.channel]
        filtered, ampenv = self.api.filter_and_ampenv(signal, selection.f0, selection.f1)
        threshold = self.compute_threshold(signal, ampenv)

        config = self.api.read_config()

        intervals = threshold_events(
            ampenv,
            threshold,
            sampling_rate=self.api.get_current_project().sampling_rate,
            ignore_width=config.get("detection.ignore_width", 0.01),
            min_size=config.get("detection.min_size", 0.01),
            fuse_duration=config.get("detection.fuse_duration", 0.01)
        )

        for interval0, interval1 in intervals:
            self.api.plugins()["SegmentPlugin"].create_segment(
                selection.x0 + int(interval0),
                selection.x0 + int(interval1),
                selection.source
            )

        logger.info("Detected {} segments".format(len(intervals)))
        self.gui.show_status("Detected {} segments".format(len(intervals)), 2000)

    def plugin_toolbar_items(self):
        return [self.button]

    def add_plugin_menu(self, menu_parent):
        menu = menu_parent.addMenu("&Detect")
        menu.addAction(self.detect_action)
        return menu

    def plugin_panel_widget(self):
        return None

    def setup_plugin_shortcuts(self):
        self.detect_action.setShortcut(QtGui.QKeySequence("W"))
