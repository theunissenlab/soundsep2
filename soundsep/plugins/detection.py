import PyQt5.QtWidgets as widgets
from PyQt5 import QtGui

import numpy as np
import pyqtgraph as pg
import scipy.signal

from soundsep.core.app import Workspace
from soundsep.core.base_plugin import BasePlugin
from soundsep.core.models import Source
from soundsep.core.segments import Segment

from soundsig.signal import lowpass_filter, highpass_filter, bandpass_filter
from soundsig.sound import spectrogram


def compute_smart_threshold(amp_env, sampling_rate=30000.0, z=6):
    """Compute a threshold by detecting baseline periods

    A chunk of audio may be mostly silence or may be mostly vocalizations
    with few or no moments of silence available. This makes it difficult to
    define a detection threshold that is robust across many windows.

    This function identifies a baseline that is robust to the overall statistics
    of sound in the window (should generally work when there is at least
    1/10th of a second of silence between vocalizations) by assuming that
    the 0.1s bin with the least variance in peak to peak distance is near
    the silence baseline.

    Parameters
    ==========
    amp_env : numpy array of floats > 0 representing the amplitude envelope of the signal
    sampling_rate (float): sampling rate of the amp env signal
    z : How many multiples of the std of the signal to add to the baseline during detection
    """

    # Find the upper and lower peaks of the amplitude envelope
    high_peaks = scipy.signal.find_peaks(amp_env)[0]
    low_peaks = scipy.signal.find_peaks(-amp_env)[0]

    # Create array of the peak to peak difference of adjacent high and low peaks
    # Generally, fluctionations due to noise will have smaller peak to peak
    # values, and real sounds will have larger peak to peak values
    if len(high_peaks) == len(low_peaks):
        peak_idx = high_peaks[:-1]
        peak2peak = np.abs(amp_env[high_peaks[:-1]] - amp_env[low_peaks[:-1]])
    elif len(high_peaks) > len(low_peaks):
        peak_idx = high_peaks[1:-1]
        peak2peak = np.abs(amp_env[high_peaks[1:-1]] - amp_env[low_peaks[1:]])
    elif len(low_peaks) > len(high_peaks):
        peak_idx = high_peaks[:-1]
        peak2peak = np.abs(amp_env[high_peaks[:-1]] - amp_env[low_peaks[1:-1]])

    # We divide up the signal into windows of equal length (dsamp) to find
    # the window with the smallest variation in peak to peak fluctuation
    # -- which we will assume occurs at the silent baseline -- since
    # When there is sound there will be high peaks and large changes in
    # overall amplitude

    # I found that 1/20th of a second seems to work; not sure if this
    # will always be true...
    dsamp = int(sampling_rate / 30.0)

    windows = []
    window_below_zero = []
    windowed_means = []
    for i in range(0, len(amp_env), dsamp):
        windows.append((peak_idx >= i) & (peak_idx < i + dsamp))
        window_below_zero.append(np.any(amp_env[i:i + dsamp] < 0))
        windowed_means.append(np.mean(amp_env[i:i + dsamp]))
    window_below_zero = np.array(window_below_zero)
    windowed_means = np.array(windowed_means)

    windowed_mean_p2p = np.array([
        np.mean(peak2peak[window]) if len(peak2peak[window]) > 1 else np.inf
        for window in windows
    ])
    windowed_stds = np.array([
        np.std(peak2peak[window]) if len(peak2peak[window]) > 1 else np.inf
        for window in windows
    ])

    # We want to choose the windows with the smallest variance (minimize
    # windowed_stds) but long sustained sounds might also have a similarly
    # low variance in peak to peak amplitude as silence, within a dsamp
    # window.

    # So, we limit our search to windows in the bottom 5% of mean
    # amplitude (for windows that don't go below 0!),
    # and select the window with the smallest peak to peak
    # variance within those options. This is a window we think is
    # best representative of a silent period.
    if not (np.sum(~window_below_zero)):
        mean_cutoff = np.percentile(windowed_means, 5)
    else:
        mean_cutoff = np.percentile(windowed_means[~window_below_zero], 5)
    _best = np.argmin(windowed_stds[windowed_means <= mean_cutoff])
    silence_window = np.where(windowed_means <= mean_cutoff)[0][_best]

    # Now, lets just say all windows that have a mean close enough
    # (2 stds) to the silence window we defined are also silence
    all_silence_windows = [
        i for i in range(len(windows)) if
        np.abs(windowed_means[i] - windowed_means[silence_window]) < 1 * windowed_stds[silence_window]
    ]

    # If we didn't find any candidate silence windows, then we have
    # to just pick a relatively conservative threshold.
    if not len(all_silence_windows):
        return np.mean(amp_env) + z * np.std(amp_env)

    silence_amp_env = np.concatenate([
        amp_env[peak_idx[windows[i]]] for i in all_silence_windows
    ])

    thresh = np.mean(silence_amp_env) + z * np.std(silence_amp_env)

    return thresh


def threshold_events(
        signal,
        threshold,
        polarity=1,
        sampling_rate=1,
        ignore_width=None,
        min_size=1,
        fuse_duration=0
    ):
    """Detect intervals crossing a threshold

    Parameters
    ==========
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


def split_individual_events(
        signal,
        sampling_rate,
        expected_call_max_duration=1.0,
        max_tries=10,
        scale_factor=1.25,
    ):
    """Divide a signal interval into individual putative events

    This function assumes that the input is a signal that already contains a lot
    of sound (detected by thresholding) and wants to split it up into individual
    events by creating a second, more conservative threshold.

    It is recommended to include some padding in the signal so that the detector
    can better find a baseline (e.g. include the period between 1.0s and 4.0s
    for a vocal period detected at 2.0s to 3.0s)

    TODO: do this across two channels?
    """
    if signal.ndim == 1:
        signal = signal - np.mean(signal)
    else:
        signal = (signal - np.mean(signal, axis=0))[:, 0]

    amp_env = get_amplitude_envelope(
        signal,
        sampling_rate,
        highpass=1000,
        lowpass=8000,
    )

    threshold = compute_smart_threshold(amp_env, sampling_rate)
    # Compute intervals within this period, preferentially separating sounds
    # that are separated by more than 20ms of silence.
    # Then, gradually raise the threshold until the longest period detected is
    # no greater than the defined max_length (default 1s)

    idx = 0
    while idx < max_tries:
        intervals = threshold_events(
            amp_env,
            threshold,
            polarity=1,
            sampling_rate=sampling_rate,
            ignore_width=0.02,
            min_size=0.01,
            fuse_duration=0.02,
        )
        durations = [np.diff(x) / sampling_rate for x in intervals]
        if len(durations) and np.max(durations) > expected_call_max_duration:
            threshold *= scale_factor
        else:
            break

        idx += 1

    if not len(intervals):
        return [[0, len(signal)]]
    else:
        return intervals


def threshold_all_events(
        audio_signal,
        window_size=10.0,
        channel=0,
        t_start=None,
        t_stop=None,
        ignore_width=0.05,
        min_size=0.05,
        fuse_duration=0.5,
        threshold_z=3.0,
        highpass=1000,
        lowpass=8000,
    ):
    """Find intervals of potential vocalizations periods (in seconds)

    The last two windows are combined in case the duration is not an
    even multiple of the window_size

    amp_env_mode can be "broadband" or "max_zscore"
    """
    sampling_rate = audio_signal.sampling_rate
    signal_duration = len(audio_signal) / sampling_rate
    if window_size is None:
        window_starts = np.array([0.0 if t_start is None else t_start])
        window_stops = np.array([audio_signal.t_max if t_stop is None else t_stop])

    else:
        window_starts = np.arange(0, signal_duration - window_size, window_size)
        window_stops = window_starts + window_size
        window_stops[-1] = signal_duration

        if t_start:
            mask = window_starts >= t_start
        else:
            mask = np.ones_like(window_starts).astype(np.bool)
        if t_stop:
            mask = mask.astype(np.bool) & (window_stops <= t_stop)

        window_starts = window_starts[mask]
        window_stops = window_stops[mask]

    last_interval_to_check = None
    all_intervals = []

    for window_start, window_stop in zip(window_starts, window_stops):
        t_arr, window_signal = audio_signal.time_slice(window_start, window_stop)
        window_signal = window_signal - np.mean(window_signal, axis=0)
        window_signal = bandpass_filter(window_signal.T, sampling_rate, 1000, 8000).T
        amp_env = get_amplitude_envelope(
            window_signal[:, channel],
            sampling_rate,
            highpass=highpass,
            lowpass=lowpass,
        )

        threshold = compute_smart_threshold(
            amp_env,
            sampling_rate=sampling_rate,
            z=threshold_z
        )

        intervals = threshold_events(
            amp_env,
            threshold,
            sampling_rate=sampling_rate,
            ignore_width=ignore_width,
            min_size=min_size,
            fuse_duration=fuse_duration
        )

        # Here begins the code that merges intervals across windows
        if last_interval_to_check is not None:
            if not len(intervals):
                all_intervals.append(last_interval_to_check)
            elif intervals[0][0] < (0.5 * sampling_rate):
                all_intervals.append((
                    last_interval_to_check[0],
                    intervals[0][1] / sampling_rate + window_start
                ))
                intervals = intervals[1:]
            else:
                all_intervals.append(last_interval_to_check)
                all_intervals.append((
                    intervals[0][0] / sampling_rate + window_start,
                    intervals[0][1] / sampling_rate + window_start
                ))
                intervals = intervals[1:]
            last_interval_to_check = None

        for i0, i1 in intervals:
            if i1 == len(window_signal):
                last_interval_to_check = (
                    i0 / sampling_rate + window_start,
                    i1 / sampling_rate + window_start
                )
                break
            all_intervals.append((
                i0 / sampling_rate + window_start,
                i1 / sampling_rate + window_start
            ))

    if last_interval_to_check is not None:
        all_intervals.append(last_interval_to_check)

    return all_intervals


def fuse_events(events, fuse_duration=0.5, target_duration=5.0):
    """From list of (start, stop) times, create list with them merged
    """
    sorted_events = sorted(events)

    result = []
    for event in sorted_events:
        if len(result) == 0:
            result.append(list(event))
        elif result[-1][1] - result[-1][0] > target_duration:
            if event[0] >= result[-1][1] + fuse_duration:
                result.append(list(event))
            if event[0] < result[-1][1] + fuse_duration:
                result[-1][1] = max(event[1], result[-1][1])
            elif event[0] < result[-1][1] and event[1] < result[-1][1]:
                pass
            elif event[0] < result[-1][1] and event[1] - result[-1][1] < fuse_duration:
                result[-1][1] = event[1]
            elif event[0] < result[-1][1] and event[1] - result[-1][1] >= fuse_duration:
                result.append([result[-1][1], event[1]])
        elif event[1] < result[-1][1]:
            pass
        elif event[0] > result[-1][1] + fuse_duration:
            result.append(list(event))
        elif event[1] > result[-1][1]:
            result[-1][1] = event[1]

    return result


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

    NAME = "Detection"

    def __init__(self, api, gui):
        super().__init__(api, gui)

        self.detectButton = widgets.QPushButton("+Detect")
        self.detectButton.clicked.connect(self.on_detect)

        self._threshold = None

        self.setup_shortcuts()

        self.api.selectionChanged.connect(self.on_selection_changed)

        self.ampenv_preview_plot = pg.PlotCurveItem()
        self.ampenv_preview_plot.setPen(pg.mkPen((200, 20, 20), width=3))
        self.threshold_preview_plot = pg.InfiniteLine(pos=0, angle=0, movable=True)
        self.threshold_preview_plot.setPen(pg.mkPen((20, 20, 20), width=3))
        self.gui.preview_plot_widget.addItem(self.ampenv_preview_plot)
        self.gui.preview_plot_widget.addItem(self.threshold_preview_plot)

        self.threshold_preview_plot.sigDragged.connect(self.on_threshold_dragged)
        # self.api.projectChanged.connect(self.on_project_changed)

    @property
    def _datastore(self):
        return self.api.get_datastore()

    def setup_shortcuts(self):
        self.detect_shortcut = widgets.QShortcut(QtGui.QKeySequence("W"), self.gui)
        self.detect_shortcut.activated.connect(self.on_detect)

    def on_threshold_dragged(self, line):
        self._threshold = line.pos().y()

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

            config = self.api.read_config()
            return (
                (i0, i1, source),
                get_amplitude_envelope(
                    sig,
                    self.api.project.sampling_rate,
                    f1,
                    f0,
                    rectify_lowpass=config.get("detection.ampenv_rectify", 200.0),
                ),
                filtered,
            )

    def compute_threshold(self, sig, ampenv):
        return self._threshold or np.std(np.abs(sig))
        # return compute_smart_threshold(ampenv)

    def on_selection_changed(self):
        """Update the preview plot with an ampenv"""
        ampenv = self._compute_selected_ampenv()
        if ampenv is None:
            self.ampenv_preview_plot.setData([], [])
        else:
            (i0, _, _), ampenv, sig = ampenv
            threshold = self.compute_threshold(sig, ampenv)
            self.ampenv_preview_plot.setData(int(i0) + np.arange(len(ampenv)), ampenv)
            self.threshold_preview_plot.setValue(threshold)

    def on_detect(self):
        """Keeps the table pointed at a visible segment"""
        ampenv = self._compute_selected_ampenv()
        if ampenv is not None:
            # Do detection
            (i0, i1, source), ampenv, sig = ampenv
            threshold = self.compute_threshold(sig, ampenv)

            config = self.api.read_config()

            intervals = threshold_events(
                ampenv,
                threshold,
                sampling_rate=self.api.project.sampling_rate,
                ignore_width=config.get("detection.ignore_width", 0.01),
                min_size=config.get("detection.min_size", 0.01),
                fuse_duration=config.get("detection.fuse_duration", 0.01)
            )

            # Iterate over the intervals, which are the sample index within the selected window
            new_segments = 0
            for interval0, interval1 in intervals:
                new_segments += 1
                self.gui.plugins["Segmentation"].create_segment(
                    i0 + int(interval0),
                    i0 + int(interval1),
                    source
                )

            self.gui.show_status("Detected {} segments".format(new_segments), 2000)

    def plugin_toolbar_items(self):
        return [self.detectButton]

    def plugin_menu(self):
        return None

    def plugin_panel_widget(self):
        return None

        # First, set up a table in the datastore

ExportPlugin = Detection
