import sys
import os
import time

import numpy as np

from PyQt5.QtCore import (Qt, QObject, QProcess, QSettings, QThread, QTimer,
        pyqtSignal, pyqtSlot)
# from PyQt6.QtMultimedia import QAudioFormat, QAudioOutput, QMediaPlayer
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtGui as gui
from PyQt5 import QtCore
from PyQt5 import QtWidgets as widgets

from scipy.fft import fft, fftfreq, next_fast_len

import pyqtgraph as pg

from soundsig.sound import spectrogram
import wavio



class ScrollingSpectrogramExample(pg.GraphicsLayoutWidget):
    """Main App instance with logic for file read/write
    """
    filename = "/home/kevin/Data/from_pepe/2018_02_11 16-31-00/ch0.wav"

    def __init__(self):
        super().__init__()
        self.title = "SoundSep"
        self.wav = wavio.read(self.filename)
        self.init_ui()
        self.init_actions()
        self.render()

    def init_actions(self):
        self.next_shortcut = widgets.QShortcut(
            gui.QKeySequence("D"),
            self
        )
        self.next_shortcut.activated.connect(self.next)

        self.prev_shortcut = widgets.QShortcut(
            gui.QKeySequence("A"),
            self
        )
        self.prev_shortcut.activated.connect(self.prev)

    def init_ui(self):
        self.spectrogram_plot = pg.PlotWidget()# viewBox=self.spectrogram_viewbox)
        self.spectrogram_plot.plotItem.setMouseEnabled(x=False, y=False)
        self.image = pg.ImageItem()
        self.spectrogram_plot.addItem(self.image)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.spectrogram_plot)
        self.setLayout(layout)

    def render(self):
        import time
        t1 = time.time()
        t_spec, f_spec, spec, rms = spectrogram(
                self.wav.data[View.slice(), 0],
                self.wav.rate, 500, 90, min_freq=200, max_freq=10000, cmplx=False) 
        t2 = time.time()
        self.image.setImage(spec.T)
        t3 = time.time()
        print("Computed spectrogram in {}s".format(t2 - t1))
        print("Rendered spectrogram in {}s".format(t3 - t2))

    def prev(self):
        View.prev()
        self.render()

    def next(self):
        View.next()
        self.render()


def _create_gaussian_window(half_window: int, nstd: int):
    t = np.arange(-half_window, half_window + 1)
    std = 2.0 * float(half_window) / float(nstd)
    return (
        np.exp(-t**2 / (2.0 * std**2))
        / (std * np.sqrt(2 * np.pi))
    )


def _iter_gaussian_windows(data, start_index, window_size, window_step, n_windows):
    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2

    gaussian_window = _create_gaussian_window(half_window, 6)

    for i in range(n_windows):
        window_center = (start_index + window_step * i)
        window_start = max(0, window_center - half_window)
        window_end = min(len(data), window_center - half_window + window_size)

        window_data = data[window_start:window_end]
        if window_start == 0:
            window_data = window_data * gaussian_window[-window_data.size:]
        elif window_end == len(data):
            window_data = window_data * gaussian_window[:window_data.size]
        else:
            window_data = window_data * gaussian_window

        yield window_data


def stft(data, start_index, window_size, window_step, n_windows):
    """Iterator over windows 
    """
    window_iterator = _iter_gaussian_windows(
        data,
        start_index,
        window_size,
        window_step,
        n_windows
    )

    fft_len = next_fast_len(window_size)
    freq = fftfreq(fft_len)
    freq_filter = freq >= 0.0

    for window in window_iterator:
        yield fft(window, n=fft_len, overwrite_x=1)[freq_filter]



class SpectrogramWorker(QThread):
    """Async worker for computing spectrogram
    """
    ready = pyqtSignal(int, object)
    finished = pyqtSignal()

    def __init__(self, data, window_size, step, t0, t1, *args, **kwargs):
        super().__init__()
        self._t0 = t0
        self._t1 = t1
        self._step = step
        self._window_size = window_size
        self._cancel_flag = False
        self.data = data
        self.args = args
        self.kwargs = kwargs

    def run(self):
        result = []

        for row in stft(
                    self.data,
                    self._t0,
                    window_size=self._window_size,
                    window_step=self._step,
                    n_windows=(self._t1 - self._t0) // self._step,
                    ):

            result.append(row)
            if self._cancel_flag == True:
                self.finished.emit()
                return

        self.ready.emit(self._t0, result)

    @pyqtSlot()
    def cancel(self):
        self._cancel_flag = True


class SpectrogramCacheExample(pg.GraphicsLayoutWidget):
    """Main App instance with logic for file read/write
    """
    filename = "/home/kevin/Data/from_pepe/2018_02_11 16-31-00/ch0.wav"

    def __init__(self):
        super().__init__()
        self.title = "SoundSep"
        self.thread = None
        self.wav = wavio.read(self.filename)
        self.init_ui()
        self.init_actions()
        self.render()

    def init_actions(self):
        self.next_shortcut = widgets.QShortcut(
            gui.QKeySequence("D"),
            self
        )
        self.next_shortcut.activated.connect(self.next)

        self.prev_shortcut = widgets.QShortcut(
            gui.QKeySequence("A"),
            self
        )
        self.prev_shortcut.activated.connect(self.prev)

    def init_ui(self):
        self.spectrogram_plot = pg.PlotWidget()# viewBox=self.spectrogram_viewbox)
        self.spectrogram_plot.plotItem.setMouseEnabled(x=False, y=False)
        self.image = pg.ImageItem()
        self.spectrogram_plot.addItem(self.image)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.spectrogram_plot)
        self.setLayout(layout)

    def render(self):
        if self.thread is not None:
            self.thread.cancel()

        self.t1 = time.time()
        self.thread = SpectrogramWorker(self.wav.data[:, 0])
        self.thread.ready.connect(self.on_spec)
        self.thread.start()
    
    def on_spec(self, idx_data, spec):
        spec = np.abs(spec)
        self.t2 = time.time()
        self.image.setImage(spec)
        self.t3 = time.time()
        print("Computed spectrogram in {}s".format(self.t2 - self.t1))
        print("Rendered spectrogram in {}s".format(self.t3 - self.t2))

    def prev(self):
        View.prev()
        self.render()

    def next(self):
        View.next()
        self.render()


############################
### Example 3. Live Spec ###
############################

class STFTCache(object):
    def __init__(self, cache_length, f_data, f_stft):
        """A cached STFT queryable by indexes in the data space

                _current_start_index
                |
                v
        [          data_length          ]
               [   cache_length  ]

        The data is indexed by the natural sampling rate of the data, f_data
        (e.g. 44.1kHz). This is different from the sampling rate of the STFT,
        f_stft (e.g. 1kHz).

        The cached data is referenced by integer index value i_stft representing
        multiples of 1/f_stft. However, the data is requested/accessed by
        integer index values representing 1/f_data, i_data. Because f_stft is typically
        smaller than f_data, multiple values of i_data may map onto the same i_stft.

        Thus, queries into the cache accept indices from the natural data,
        but are immediately converted into stft indices for the query
        """
        self._ready = False
        self._data = np.zeros((cache_length, 0))
        self._conversion_factor = f_data / f_stft
        self._current_start_idx_stft = 0

    def _move_stft(self, start_index):
        """Move cached region to a new section
        """
        offset = start_index - self._current_start_idx_stft
        n = np.abs(offset)

        print("Moving from {} to {}".format(self._current_start_idx_stft, start_index))

        self._data = np.roll(self._data, -offset, axis=0)
        if offset < 0:
            self._data[:n] = 0
        elif offset > 0:
            self._data[-n:] = 0

        self._current_start_idx_stft = start_index

    def _data_index_to_stft_index(self, data_index):
        return int(data_index / self._conversion_factor)

    def move(self, idx_data):
        idx_stft = self._data_index_to_stft_index(idx_data)

        print("Mapped {} to {}".format(idx_data, idx_stft))
        self._move_stft(idx_stft)

    def set_data(self, idx_data, data):
        self._ready = True
        idx_stft = self._data_index_to_stft_index(idx_data)
        self._move_stft(idx_stft)
        self._data = data
        # self._current_start_idx_stft = idx_stft

    def ready(self):
        return self._ready

    def read(self):
        if not self.ready():
            raise RuntimeError("STFTCache.read() tried to be called before data was ever set")

        return self._data[:]


class SpectrogramLiveExample(pg.GraphicsLayoutWidget):
    """Main App instance with logic for file read/write
    """
    filename = "/home/kevin/Data/from_pepe/2018_02_11 16-31-00/ch0.wav"

    def __init__(self):
        super().__init__()
        self.title = "SoundSep"
        self.thread = None

        self._step = 22 * 4
        self.stft_cache = STFTCache(0, 22050, 22050 / self._step)
        self._image_set = False
        self.wav = wavio.read(self.filename)
        self.init_ui()
        self.init_actions()
        self.render()

    def init_actions(self):
        self.next_shortcut = widgets.QShortcut(
            gui.QKeySequence("D"),
            self
        )
        self.next_shortcut.activated.connect(self.next)

        self.prev_shortcut = widgets.QShortcut(
            gui.QKeySequence("A"),
            self
        )
        self.prev_shortcut.activated.connect(self.prev)

    def init_ui(self):
        self.spectrogram_plot = pg.PlotWidget()# viewBox=self.spectrogram_viewbox)
        self.spectrogram_plot.plotItem.setMouseEnabled(x=False, y=False)
        self.image = pg.ImageItem()
        self.spectrogram_plot.addItem(self.image)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.spectrogram_plot)
        self.setLayout(layout)

    def render(self):
        if self.thread is not None:
            self.thread.cancel()

        self.t1 = time.time()
        self.thread = SpectrogramWorker(self.wav.data[:, 0], 500, self._step, View.t0, View.t1)
        self.thread.ready.connect(self.on_spec)
        self.thread.start()

        self.stft_cache.move(View.t0)
        self.update_image()

    def update_image(self):
        if self.stft_cache.ready():
            if self._image_set:
                print("setting image")
                self.image.updateImage(self.stft_cache.read())
            else:
                self.image.setImage(self.stft_cache.read())
                self._image_set = True
    
    def on_spec(self, idx_data, spec):
        self.stft_cache.set_data(idx_data, np.abs(spec))
        self.t2 = time.time()
        self.update_image()
        self.t3 = time.time()
        print("Computed spectrogram in {}s".format(self.t2 - self.t1))
        print("Rendered spectrogram in {}s".format(self.t3 - self.t2))

    def prev(self):
        View.prev()
        self.render()

    def next(self):
        View.next()
        self.render()

