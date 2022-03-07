import logging
from queue import Empty, Queue

import numpy as np
from PyQt6 import QtWidgets as widgets
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import rfft, rfftfreq

from soundsep.core.models import Project, ProjectIndex, Source, StftIndex
from soundsep.core.stft import create_gaussian_window
from soundsep.core.stft.cache import CacheLayer, StftCache, StftParameters
from soundsep.core.stft.lattice import Bound
from soundsep.core.stft.levels import compute_pad_for_windowing, gen_bounded_layers


logger = logging.getLogger(__name__)


class StftWorker(QThread):
    """Background thread for processing STFT

    Emits resultReady(StftIndex, channel, data) when the fft at a particular
    StftIndex and channel is completed.

    Pushing tuples of (StftIndex, StftIndex) to the StftWorker.queue will
    tell it to compute the fft at every index in that range. If

    Attributes
    ----------
    queue : queue.Queue
        Queue that the worker pulls requests from. Tuples of StftIndex start
        and stop pairs trigger the worker to processing FFTs for every time
        point in that range.
    config : StftConfig
        The configuration dictionary containing the stft.window and stft.step
        parameters
    """

    _CLEAR = object()
    END = object()
    resultReady = pyqtSignal(CacheLayer, int, np.ndarray)

    def __init__(self, queue: Queue, project: Project, stft_params: StftParameters):
        super().__init__()
        self.queue = queue
        self.project = project
        self.params = stft_params

        self.gaussian_window = create_gaussian_window(self.params.half_window, nstd=6)
        self.gaussian_window = self.gaussian_window.reshape((-1, 1))  # for broadcasting
        self._cancel_flag = False

    def cancel(self):
        """Tell the program to stop processing and clear the queue"""
        self._cancel_flag = True
        while not self.queue.empty():
            try:
                self.queue.get(False)
            except Empty:
                continue

        # This tells the worker it can start processing events again
        self.queue.put(StftWorker._CLEAR)

    def run(self):
        while True:
            q = self.queue.get()
            if q is StftWorker._CLEAR:
                self._cancel_flag = False
            elif q is StftWorker.END:
                break
            else:
                self.process_request(q[0], q[1])

    def iter_lattice_windows(
            self,
            bounded_lattice: 'BoundedLattice',
            scale_factor: int,
            half_window: int,
            job,
        ):
        """A generator over windows whose centers lie on a given lattice

        The windows are views into the original array so should relatively fast

        Yields
        ------
        window_idx : StftIndex
            Index in the BoundedLattice's native coordinate system
        window : np.ndarray (view into arr)
            A view into arr of a window centered on window_idx. The central coordinate
            of this window in the signal coordinates should be ``window_idx * scale``.
        """
        scaled_lattice = bounded_lattice * scale_factor
        window_centers = np.array(list(scaled_lattice))

        if window_centers[0] > self.project.frames:
            return

        pad_start, pad_stop, wanted_start, wanted_stop = compute_pad_for_windowing(
            self.project.frames,
            window_centers[0],
            window_centers[-1],
            half_window
        )

        arr = self.project[ProjectIndex(self.project, wanted_start):ProjectIndex(self.project, wanted_stop)]
        padding = ((pad_start, pad_stop),) + tuple([(0, 0) for _ in range(arr.ndim - 1)])

        padded_arr = np.pad(arr, padding, mode="reflect")
        windows = sliding_window_view(padded_arr, 2 * half_window + 1, axis=0)

        for window_idx, window in zip(
                bounded_lattice,
                windows[scaled_lattice.to_slice(relative_to=wanted_start + half_window - pad_start)]
                ):
            yield window_idx, window


    def process_request(self, layer: '_CacheLayer', job: 'Tuple[StftIndex, StftIndex]'):
        logger.debug("Procesing job: %s %s", layer.lattice, job)
        for window_center, window in self.iter_lattice_windows(
                bounded_lattice=layer.lattice.with_bound(Bound(start=int(job[0]), stop=int(job[1]))),
                scale_factor=self.params.hop,
                half_window=self.params.half_window,
                job=job
                ):
            if self._cancel_flag:
                logger.debug("Job canceled")
                return
            window_fft = np.abs(rfft(window.T * self.gaussian_window, n=self.params.n_fft, axis=0))
            self.resultReady.emit(layer, window_center, window_fft)
        logger.debug("Job finished")


class StftService(QObject):
    """A service for caching stft values in and around an active data range

    The StftService provides access to a hierarchical cache of STFT data. The
    hierarchy contains ``n_scales`` levels which increase in size while decreasing
    in temporal resolution.

    Each level contains data samples offset such that no frame number is computed
    more than once across levels, and that computing frames from all levels gives
    the full spectrogram.

    Here is a visual depiction of how the layers are organized:

      ```
      1| . . . . . .
      2|  .   .   .   .   .   .   .
      3|    .       .       .       .       .       .       .       .
      4|        .               .               .               .
      5|.               .               .               .               .
      ```

    Arguments
    ---------
    project : Project
    n_scales : int
        Number of scales at which data can be viewed, each scale increasing by
        a power of 2.
    cache_size : int
        The number of samples stored in each cache layer
    fraction_cached : float
        The fraction of the cache size which triggers reads at the next scale.
        E.g. If each layer is 2000 samples wide and a range of data 1001 samples
        wide is attempted to be read, the StftService will choose to read
        at the 2nd level (every other sample)
    stft_params : StftParameters
        The stft config parameters including stft.half_window and stft.hop
    """

    def __init__(self, project, n_scales: int, cache_size: int, fraction_cached: float, stft_params: 'StftParameters'):
        super().__init__()
        self.project = project
        self.params = stft_params
        self.fraction_cached = fraction_cached

        self._freqs = rfftfreq(self.params.n_fft, 1 / self.project.sampling_rate)
        self._positive_freqs_filter = self._freqs > 0
        self.positive_freqs = self._freqs[self._freqs > 0]

        # Initialize the cache
        layers = gen_bounded_layers(n_scales, cache_size)
        self.cache = StftCache([CacheLayer(l) for l in layers])
        self.cache.set_shape(self.project.channels, self.params.half_window)
        #
        self.queue = Queue()
        self._worker = StftWorker(self.queue, project, self.params)
        self._worker.resultReady.connect(self._on_worker_result)
        self._worker.start()

    @pyqtSlot(CacheLayer, int, np.ndarray)
    def _on_worker_result(self, receiver_layer, i, data):
        """Routes the results of the computations to the right layer"""
        i = StftIndex(self.project, self.params.hop, i)
        receiver_layer.set_data(i, data.T[:, self._positive_freqs_filter])

    def close(self):
        self._worker.cancel()

    @property
    def n_cache_total(self):
        return self.cache.layers[-1].data_range()

    def set_central_range(self, i0: 'StftIndex', i1: 'StftIndex'):
        """
        """
        level = self.cache.choose_level(i1 - i0)
        self.cache.set_central_range(i0, i1, (self.project.frames // self.params.hop) + 1)

        self._worker.cancel()

        for layer in reversed(self.cache.layers[level:]):
            for job in layer.get_primary_jobs(int(i0), int(i1)):
                job = (
                    StftIndex(self.project, self.params.hop, job[0]),
                    StftIndex(self.project, self.params.hop, job[1])
                )
                self._worker.queue.put((layer, job))

        for layer in reversed(self.cache.layers[level:]):
            for job in layer.get_secondary_jobs(int(i0), int(i1), (self.project.frames // self.params.hop) + 1):
                job = (
                    StftIndex(self.project, self.params.hop, job[0]),
                    StftIndex(self.project, self.params.hop, job[1])
                )
                self._worker.queue.put((layer, job))

    def read(self, i0: 'StftIndex', i1: 'StftIndex'):
        level = self.cache.choose_level(i1 - i0, fraction_of_cache=self.fraction_cached)
        logger.debug("Reading stft data at scale: {}".format(pow(2, level)))
        t, data, stale = self.cache.read(int(i0), int(i1), level)
        return t, data, stale[:, 0]
