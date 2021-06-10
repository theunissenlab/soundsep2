import PyQt5.QtWidgets as widgets
from PyQt5 import QtGui

import numpy as np
import pyqtgraph as pg

from soundsep.core.app import Workspace
from soundsep.core.base_plugin import BasePlugin
from soundsep.core.models import Source
from soundsep.core.segments import Segment


import scipy.signal
from soundsig.signal import lowpass_filter, highpass_filter, bandpass_filter
from soundsig.sound import spectrogram


def get_amplitude_envelope(
            data,
            fs=30000.0,
            lowpass=8000.0,
            highpass=1000.0,
            rectify_lowpass=600.0,
        ):
    """Compute an amplitude envelope of a signal

    This can be done in two modes: "broadband" or "max_zscore"

    Broadband mode is the normal amplitude envelope calcualtion. In broadband
    mode, the signal is bandpass filtered, rectified, and then lowpass filtered.

    Max Zscore mode is useful to detect calls which may be more localized in
    the frequency domain from background (white) noise. In max_zscore mode, a
    spectrogram is computed with spec_sample_rate time bins and
    spec_freq_spacing frequency bins. For each time bin, the power in each
    frequency bin is zscored and the max zscored value is assigned to the
    envelope for that bin.
    """
    spectral = True
    filtered = highpass_filter(data.T, fs, highpass, filter_order=10).T
    filtered = lowpass_filter(filtered.T, fs, lowpass, filter_order=10).T

    # Rectify and lowpass filter
    filtered = np.abs(filtered)
    filtered = lowpass_filter(filtered.T, fs, rectify_lowpass).T
    return filtered


class Detection(BasePlugin):
    """Plugin for allowing users to 

    TODO: how to make sure Detection requires Segmentation?
    """

    def __init__(self, api, gui):
        super().__init__(api, gui)
        self._datastore = self.api.get_datastore()

        self.detectButton = widgets.QPushButton("+Detect")
        self.detectButton.clicked.connect(self.on_detect)

        self.setup_shortcuts()

        self.api.selectionChanged.connect(self.on_selection_changed)

        self.ampenv_preview_plot = pg.PlotCurveItem()
        self.ampenv_preview_plot.setPen(pg.mkPen((200, 20, 20), width=3))
        self.threshold_preview_plot = pg.InfiniteLine(pos=0, angle=0, movable=True)
        self.threshold_preview_plot.setPen(pg.mkPen((20, 20, 20), width=3))
        self.gui.preview_plot_widget.addItem(self.ampenv_preview_plot)
        self.gui.preview_plot_widget.addItem(self.threshold_preview_plot)

    def setup_shortcuts(self):
        self.detect_shortcut = widgets.QShortcut(QtGui.QKeySequence("W"), self.gui)
        self.detect_shortcut.activated.connect(self.on_detect)

    def _compute_selected_ampenv(self):
        try:
            (i0, i1), (f0, f1), source = self.api.get_active_selection()
        except TypeError:
            return None
        else:
            sig = self.api.project[i0:i1, source.channel]
            f0 = max(f0, 250.0)
            f1 = min(f1, 10000)
            if f1 <= f0 or i1 - i0 < 33:
                return None
            filtered = bandpass_filter(sig, self.api.project.sampling_rate, f0, f1)
            return (
                get_amplitude_envelope(
                    sig,
                    self.api.project.sampling_rate,
                    f1,
                    f0,
                    rectify_lowpass=100.0,
                ),
                filtered,
            )

    def compute_threshold(self, sig):
        return np.std(np.abs(sig))

    def on_selection_changed(self):
        """Update the preview plot with an ampenv"""
        ampenv = self._compute_selected_ampenv()
        if ampenv is None:
            self.ampenv_preview_plot.setData([], [])
        else:
            ampenv, sig = ampenv
            threshold = self.compute_threshold(sig)
            self.ampenv_preview_plot.setData(np.arange(len(ampenv)), ampenv)
            self.threshold_preview_plot.setValue(threshold)

    def on_detect(self):
        """Keeps the table pointed at a visible segment"""
        ampenv = self._compute_selected_ampenv()
        if ampenv is not None:
            # Do detection
            ampenv, sig = ampenv
            threshold = self.compute_threshold(sig)
            new_segments = []
            self.gui.show_status("Detected {} segments".format(new_segments), 2000)

    def plugin_toolbar_items(self):
        return [self.detectButton]

    def plugin_menu(self):
        return None

    def plugin_panel_widget(self):
        return None

        # First, set up a table in the datastore

ExportPlugin = Detection
