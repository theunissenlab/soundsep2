from enum import Enum

from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
import numpy as np
import pyqtgraph as pg
import wavio

from soundsep.core.models import ProjectIndex
from soundsep.core.stft import spectral_derivative, stft_gen, stft_freq


class STFTViewMode(Enum):
    NORMAL = 1
    DERIVATIVE = 2


class _STFTCache(object):
    def __init__(self, f_data, f_stft):
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
        self._data = np.zeros((0, 0))
        self._conversion_factor = f_data / f_stft
        self._current_start_idx_stft = 0

    def _move_stft(self, start_index):
        """Move cached region to a new section
        """
        offset = start_index - self._current_start_idx_stft
        n = np.abs(offset)

        self._data = np.roll(self._data, -offset, axis=0)
        if offset < 0:
            self._data[:n] = 0
        elif offset > 0:
            self._data[-n:] = 0

        self._current_start_idx_stft = start_index

    def _data_index_to_stft_index(self, data_index):
        return int(data_index / self._conversion_factor)

    def move(self, idx_data):
        """Move left edge of cache to given idx_data in audio samples

        Converts the idx_data in audio samples to the corresponding
        stft window, then moves the array to point to that location.
        """
        idx_stft = self._data_index_to_stft_index(idx_data)
        self._move_stft(idx_stft)

    def set_data(self, idx_data, data):
        self._ready = True
        idx_stft = self._data_index_to_stft_index(idx_data)

        # Move the pointer and set data
        # If we instead calculate the exact position to overwrite in self._data
        # we wouldn't need to move the pointer to enforce consistency
        self._move_stft(idx_stft)
        self._data = data

    def ready(self):
        """Returns True if data has been set at least once"""
        return self._ready

    def get_stft_idx_slice(self):
        return slice(self._current_start_idx_stft, self._current_start_idx_stft + len(self._data))

    @property
    def _current_start_idx_data(self):
        return int(self._current_start_idx_stft * self._conversion_factor)

    def get_data_idx_slice(self):
        start_idx = self._current_start_idx_data
        n_samples = int(len(self._data) * self._conversion_factor)
        return slice(start_idx, start_idx + n_samples)

    def read(self):
        if not self.ready():
            raise RuntimeError("_STFTCache.read() tried to be called before data was ever set")

        return self._data[:]


class _STFTWorker(QThread):
    """Async worker for computing spectrogram
    """
    ready = pyqtSignal(int, object)

    def __init__(self, data, window_size, step, t0):
        super().__init__()
        self._t0 = int(t0)
        # self._t1 = t1
        self._step = step
        self._window_size = window_size
        self.data = data
        self._cancel_flag = False

    def run(self):
        result = []

        for row in stft_gen(
                    self.data,
                    0,
                    window_size=self._window_size,
                    window_step=self._step,
                    n_windows=len(self.data) // self._step,
                    ):
            result.append(row)
            if self._cancel_flag == True:
                return

        self.ready.emit(self._t0, np.array(result))

    @pyqtSlot()
    def cancel(self):
        self._cancel_flag = True


class ScrollableSpectrogramConfig:
    def __init__(
            self, 
            window_size: int,
            window_step: int,
            spectrogram_size: int,
            cmap: str,
        ):
        self._window_size = window_size
        self._window_step = window_step
        self._spectrogram_size = spectrogram_size

    @property
    def window_size(self):
        """Width of stft windows in audio samples"""
        return self._window_size

    @property
    def window_step(self):
        """Number of audio samples between stft windows"""
        return self._window_step

    @property
    def spectrogram_size(self):
        """Number of samples to display of spectrogram in (in stft samples)"""
        return self._spectrogram_size

    def set_spectrogram_size(self, size: int):
        self._spectrogram_size = size


class ScrollableSpectrogram(pg.PlotWidget):
    """Scrollable Spectrogram Plot Widget that uses the spectrogram indices as its native coordinate system
    """
    def __init__(
            self,
            project,
            channel,
            config: ScrollableSpectrogramConfig,
            *args,
            **kwargs
        ):
        super().__init__(*args, background=None, **kwargs)
        self.project = project
        self.channel = channel
        self.plotItem.setMouseEnabled(x=False, y=False)

        self.config = config
        # self._wav = wavio.read(filename)  # Should I read every time or once?
        self._stft_cache = _STFTCache(self.project.sampling_rate, self.project.sampling_rate / self.config.window_step)
        self._stft_thread = None
        self.freqs = stft_freq(self.config.window_size, self.project.sampling_rate)

        # TODO hardcoded?
        self._view_mode = STFTViewMode.NORMAL

        if self._view_mode == STFTViewMode.DERIVATIVE:
            self._cmap = pg.colormap.get("gist_yarg", source='matplotlib')
        elif self._view_mode == STFTViewMode.NORMAL:
            self._cmap = pg.colormap.get("turbo", source='matplotlib', skipCache=True)
        else:
            raise RuntimeError("Invalid _view_mode {}".format(self._view_mode))

        self.init_ui()

    def ylim(self):
        freq_values = self.freqs[:len(self.freqs) // 2 - 1]
        return (freq_values[0], freq_values[-1])

    def xlim(self):
        return (
            self._stft_cache._current_start_idx_data,
            self._stft_cache._current_start_idx_data + self.config.spectrogram_size
        )

    def init_ui(self):
        self.image = pg.ImageItem()
        self.addItem(self.image)
        self.image.setLookupTable(self._cmap.getLookupTable(alpha=True))

    def update_image(self):
        if self._stft_cache.ready():
            stft_array = self._stft_cache.read()

            # TODO: transform spectrogram into data coordinates
            '''
            tr = QtGui.QTransform()
            tr.translate(
                self._stft_cache._current_start_idx_data,
                0)
            tr.scale(self._stft_cache._conversion_factor, np.max(self.freqs) / stft_array.shape[1])
            center = stft_array.shape[0] / 2
            print(self._stft_cache._current_start_idx_data)
            '''

            if self._view_mode == STFTViewMode.NORMAL:
                self.image.setImage(stft_array)
                # self.image.setTransform(tr)
            elif self._view_mode == STFTViewMode.DERIVATIVE:
                self.image.setImage(spectral_derivative(stft_array))
                # self.image.setTransform(tr)
            else:
                raise RuntimeError("Invalid _view_mode {}".format(self._view_mode))

    def set_cmap(self, cmap):
        # TODO this doesnt use config at all
        self._cmap = pg.colormap.get(cmap, source='matplotlib', skipCache=True)
        self.image.setLookupTable(self._cmap.getLookupTable(alpha=True))

    def set_view_mode(self, view_mode: STFTViewMode):
        self._view_mode = view_mode

    def set_spec(self, idx_data, spec):
        """Set stft data aligned to idx_data, filtering down to positive frequencies

        :param idx_data: Index in audio samples to assign data to
        :type idx_data: int (wav coordinates)
        :param spec: Array of shape (N, M) of stft data
        :type spec: np.ndarray
        """
        self._stft_cache.set_data(idx_data, np.abs(spec[:, self.freqs > 0.0]))
        self.update_image()

    def scroll_to(self, idx_data, idx_data_end=None):
        """Move visible portion of spectrogram to idx_data

        :param idx_data: Index in audio samples to move view to 
        :type idx_data: int (wav coordinates)
        """
        if idx_data_end is not None:
            self.config.set_spectrogram_size(idx_data_end - idx_data)

        # Cancel any ongoing computation
        if self._stft_thread is not None:
            self._stft_thread.cancel()

        i0 = ProjectIndex(self.project, idx_data)
        i1 = ProjectIndex(self.project, idx_data + self.config.spectrogram_size)

        self._stft_thread = _STFTWorker(
                self.project[i0:i1, self.channel].flatten(),
                self.config.window_size,
                self.config.window_step,
                idx_data,
        )
        self._stft_thread.ready.connect(self.set_spec)
        self._stft_thread.start()

        # Even though the spectrogram may take some time to compute,
        # move the visible portion of the stft over the requested amount
        self._stft_cache.move(idx_data)
        self.update_image()

    def visible_audio(self):
        # first convert to ProjectIndex coordinates
        slice_ = self._stft_cache.get_data_idx_slice()
        i0 = ProjectIndex(self.project, slice_.start)
        i1 = ProjectIndex(self.project, slice_.stop)
        return self.project[i0:i1, self.channel][:, 0]

    def visible_spectrogram(self):
        return self._stft_cache.read()
        
