import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from soundsep.core.utils import DuplicatedRingBuffer, ceil_div, get_blocks_of_ones

from .lattice import (
    Bound,
    BoundedLattice,
    ceil as lattice_ceil,
    floor as lattice_floor,
    overlapping_slice,
    overlapping_range
)
from .levels import gen_bounded_layers
from .params import StftParameters


def create_cache_from_layers(layers: List[BoundedLattice]):
    BoundedLattice


logger = logging.getLogger(__name__)


class _CacheLayer:

    def __init__(self, lattice: 'BoundedLattice'):
        self.lattice = lattice
        self.data = DuplicatedRingBuffer(np.zeros((len(self.lattice), 0, 0)))
        self.stale = DuplicatedRingBuffer(np.ones((len(self.lattice), 0), dtype=np.bool))

    def data_range(self) -> int:
        """

        Return the size of the range spanned by this caching layer in StftIndex samples
        """
        return len(self.data) * self.lattice.step

    def get_lim(self) -> int:
        """Return start (inclusive), stop (non-inclusive) tuple in StftIndex"""
        return self.lattice[0], self.lattice[len(self.lattice)]

    def set_shape(self, channels: int, features: int):
        """Set the size of the buffered data

        Arguments
        ---------
        shape : Tuple[int, int]
            A tuple representing the
        """
        self.data = DuplicatedRingBuffer(np.zeros((len(self.lattice), channels, features)))
        self.stale = DuplicatedRingBuffer(np.ones((len(self.lattice), channels), dtype=np.bool))

    def set_data(self, idx: 'StftIndex', data: np.ndarray):
        if int(idx) == 1997:
            print("Setting {}".format(idx))
            print("Could remap to ", self.lattice.to_position(idx))
            print(self.lattice, self.data.shape)
            print("Or...", int(idx) - self.lattice[0])
        i = self.lattice.to_position(idx)
        self.data[i] = data
        self.stale[i] = False

    def get_bounds_from_central_range(self, i0: int, i1: int, full_size: int) -> 'Bound':
        """Find a new bound centered on (i0, i1), but on the current lattice

        The max i has to be provided since this object doesn't have knowledge
        of the size of the full data array
        """
        layer_size = self.data_range()
        samples_before = ceil_div(layer_size - (i1 - i0), 2)
        potential_start = lattice_floor(i0 - samples_before, self.lattice)
        assert potential_start in self.lattice.without_bound()

        start = max(self.lattice.offset, potential_start)

        if layer_size >= full_size:
            stop = start + layer_size
        elif start + layer_size > full_size:
            stop = lattice_floor(full_size, self.lattice)
            start = stop - layer_size
        else:
            stop = start + layer_size

        assert start in self.lattice.without_bound()

        return Bound(start, stop)

    def set_central_range(self, i0: int, i1: int, full_size: int):
        """Center this CacheLayer on the given position

        The size of the lattice bounds must remain constant
        """
        new_bound = self.get_bounds_from_central_range(i0, i1, full_size)
        if self.lattice.bound.start == new_bound.start:
           return

        offset = (new_bound.start - self.lattice.bound.start) // self.lattice.step
        self.data.roll(offset, fill=0)
        self.stale.roll(offset, fill=True)
        self.lattice.bound = new_bound

    def get_primary_jobs(self, i0: int, i1: int):
        """Return a list of (int, int, BoundedLattice) describing the jobs that must be run.
        """
        request_ranges = [
            (i0, i1)
        ]

        jobs = []
        print("requesting from {}: {} {}".format(self.lattice, i0, i1))
        for start, stop in request_ranges:
            slice_ = slice(*overlapping_slice(start, stop, self.lattice))
            data_start = self.lattice.to_position(slice_.start)
            data_stop = self.lattice.to_position(slice_.stop)
            for a, b in get_blocks_of_ones(self.stale[data_start:data_stop, 0]):
                jobs.append((start + int(a) * self.lattice.step, start + int(b) * self.lattice.step))
        print("got: {}".format(jobs))
        return jobs

    def get_secondary_jobs(self, i0: int, i1: int, full_size):
        cache_bound = self.get_bounds_from_central_range(i0, i1, full_size)
        request_ranges = [
            (i1, cache_bound.stop),
            (cache_bound.start, i0)
        ]

        jobs = []

        for start, stop in request_ranges:
            slice_ = slice(*overlapping_slice(start, stop, self.lattice))
            data_start = self.lattice.to_position(slice_.start)
            data_stop = self.lattice.to_position(slice_.stop)
            for a, b in get_blocks_of_ones(self.stale[data_start:data_stop, 0]):
                jobs.append((start + int(a) * self.lattice.step, start + int(b) * self.lattice.step))

        return jobs

#
# @dataclass
# class OffsetArray:
#     ptr: int
#     data: np.ndarray


# This should only be generated by the gen_bounded_layers functions
class StftCache:
    def __init__(self, layers: List[_CacheLayer]):
        self.layers = layers

    @property
    def n_levels(self) -> int:
        return len(self.layers)

    def choose_level(self, read_size: int):
        """Choose level to read up to
        """
        for i, layer in enumerate(self.layers):
            # If the read_size is within the bounds of the layer, we can read up to that layer
            if read_size < layer.data_range() // 2:
                return i
        else:
            return self.n_levels - 1

    def read(self, i0: int, i1: int, level: int):
        """Read data from the given StftIndex coordinates"""
        arr = np.zeros(shape=(i1 - i0, self.layers[0].data.shape[1], self.layers[0].data.shape[2]))
        stale_mask = np.ones(shape=(i1 - i0, self.layers[0].stale.shape[1]), dtype=np.bool)

        first_offset = self.layers[-1].lattice.offset
        for layer in self.layers[level:]:
            layer_lim = layer.get_lim()
            start, stop, step = overlapping_slice(i0, i1, layer.lattice)
            layer_selector = slice((start - layer_lim[0]) // step, (stop - layer_lim[0]) // step)
            fill_data = layer.data[layer_selector]

            stale_mask[start - i0:stop - i0:step] &= layer.stale[layer_selector]
            first_offset = min(first_offset, start - i0)
            arr[start - i0:stop - i0:step] = fill_data

        # first_offset, _, _ = overlapping_slice(i0, i1, self.layers[level].lattice)
        # first_offset -= self.layers[level].get_lim()[0]
        every_other = pow(2, level)
        return np.arange(i0 + first_offset, i1, every_other), arr[first_offset::every_other], stale_mask[first_offset::every_other]

    def set_shape(self, channels: int, features: int):
        for layer in self.layers:
            layer.set_shape(channels, features)

    def set_central_range(self, i0: int, i1: int, full_size: int):
        """

        Coordinate layers by moving them such that they all center on the given
        pos. Then,
        """
        for layer in self.layers:
            layer.set_central_range(i0, i1, full_size)

# iter_lattice_windows

from scipy.fft import rfft, rfftfreq
from soundsep.core.stft.levels import iter_lattice_windows
from queue import Empty, Queue

from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from soundsep.core.stft import create_gaussian_window
from soundsep.core.models import Project, ProjectIndex, Source, StftIndex

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
    resultReady = pyqtSignal(_CacheLayer, int, np.ndarray)

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

    def process_request(self, layer: '_CacheLayer', job: 'Tuple[StftIndex, StftIndex]'):
        try:
            arr = self.project[job[0].to_project_index():job[1].to_project_index()]
        except:
            # Projabbly is out of range?
            logger.warning("Could not read job {} for project of length {}".format(job, self.project.frames))
            return
        logger.debug("Procesing job", layer.lattice, job)
        for window_center, window in iter_lattice_windows(
                arr=arr,
                bounded_lattice=layer.lattice.with_bound(Bound(start=int(job[0]), stop=int(job[1]))),
                scale_factor=self.params.hop,
                half_window=self.params.half_window
                ):
            if self._cancel_flag:
                logger.debug("Job canceled")
                return
            window_fft = np.abs(rfft(window.T * self.gaussian_window, n=self.params.n_fft, axis=0))
            self.resultReady.emit(layer, window_center, window_fft)
        logger.debug("Job finished")

    # def process_request(self, start: 'StftIndex', stop: 'StftIndex'):
    #     n_fft = 2 * self.config.window + 1
    #     phantom_start = start.to_project_index() - self.config.window
    #     phantom_stop = stop.to_project_index() + self.config.window
    #     arr = self.project[max(ProjectIndex(self.project, 0), phantom_start):phantom_stop]
    #
    #     if int(phantom_start) < 0:
    #         pad_start = int(np.abs(phantom_start))
    #     else:
    #         pad_start = 0
    #
    #     if int(phantom_stop) > self.project.frames:
    #         pad_stop = int(phantom_stop - self.project.frames)
    #     else:
    #         pad_stop = 0
    #
    #     arr = np.pad(arr, ((pad_start, pad_stop), (0, 0)), mode="reflect")
    #
    #     # Produce a view of shape (steps, samples, channels)
    #     strides = sliding_window_view(arr, n_fft, axis=0)[::self.config.step].swapaxes(1, 2)
    #     for window_center_stft, window in zip(StftIndex.range(start, stop), strides):
    #         if self._cancel_flag:
    #             return
    #         window_data = window * self.gaussian_window
    #         window_fft = np.abs(rfft(window_data, n=n_fft, axis=0))
    #         self.resultReady.emit(window_center_stft, window_fft)


class StftService(QObject):
    def __init__(self, project, stft_params: 'StftParameters'):
        super().__init__()
        self.project = project
        self.params = stft_params

        # Initialize the cache
        layers = gen_bounded_layers(8, 2000)
        # stft_params = StftParameters(hop=44, half_window=302)

        self._freqs = rfftfreq(self.params.n_fft, 1 / self.project.sampling_rate)
        self._positive_freqs_filter = self._freqs > 0
        self.positive_freqs = self._freqs[self._freqs > 0]

        self.cache = StftCache([_CacheLayer(l) for l in layers])
        self.cache.set_shape(self.project.channels, self.params.half_window)
        #
        self.queue = Queue()
        self._worker = StftWorker(self.queue, project, self.params)
        self._worker.resultReady.connect(self._on_worker_result)
        self._worker.start()

    @pyqtSlot(_CacheLayer, int, np.ndarray)
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
        # fancy calc
        # send cancel tasks
        # Calculate new jobs send new task to worker
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
        level = self.cache.choose_level(i1 - i0)
        print("reading at level", level)
        t, data, stale = self.cache.read(int(i0), int(i1), level)
        print(np.mean(stale))
        return t, data, stale[:, 0]
