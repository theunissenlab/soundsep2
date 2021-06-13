from queue import Queue
from typing import Tuple

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from scipy.fft import fft, fftfreq, next_fast_len
import numpy as np

from soundsep.core.models import StftIndex, Project, ProjectIndex
from soundsep.core.stft import _create_gaussian_window

import logging


logger = logging.getLogger(__name__)


class StftConfig(dict):
    def __init__(self, window, step):
        self.window = window
        self.step = step


class StftWorker(QThread):
    CLEAR = object()
    END = object()
    requestStft = pyqtSignal(StftIndex, StftIndex)
    resultReady = pyqtSignal(StftIndex, int, object)

    def __init__(self, queue: Queue, project: Project, config: StftConfig):
        super().__init__()
        self.queue = queue
        self.project = project
        self.config = config

        self.gaussian_window = _create_gaussian_window(self.config.window, nstd=6)

        # self.requestStft.connect(self.on_request)
        self._cancel_flag = False

    def cancel(self):
        self._cancel_flag = True

    def run(self):
        while True:
            q = self.queue.get()
            if q is StftWorker.CLEAR:
                self._cancel_flag = False
            elif q is StftWorker.END:
                break
            else:
                self.process_request(q[0], q[1])

    def process_request(self, start: StftIndex, stop: StftIndex):
        # TODO: read the necessary data so we aren't constantly going to disk?
        for window_center_stft in StftIndex.range(start, stop):
            window_center = window_center_stft.to_project_index()
            window_start = max(ProjectIndex(self.project, 0), window_center - self.config.window)
            window_stop = min(ProjectIndex(self.project, self.project.frames), window_center + self.config.window + 1)

            if int(window_start) >= self.project.frames:
                break

            for ch in range(self.project.channels):
                if self._cancel_flag:
                    # End this loop and empty the queue
                    return

                window_data = self.project[window_start:window_stop, ch]

                if int(window_start) == 0:
                    window_data = window_data * self.gaussian_window[-window_data.size:]
                elif int(window_stop) == self.project.frames:
                    window_data = window_data * self.gaussian_window[:window_data.size]
                else:
                    window_data = window_data * self.gaussian_window

                # TODO: benchmark using next_fast_len here?
                window_fft = np.abs(fft(window_data, n=2 * self.config.window + 1))

                self.resultReady.emit(window_center_stft, ch, window_fft)


class StftCache(QObject):
    """A cache for stft values that slides its active data range and caches values around it

    Defines an active range of data of width visible_size and datapoints around it
    equal to pad * 2.
    """
    def __init__(self, project, n_active: int, pad: int, stft_config: StftConfig):
        super().__init__()

        self.project = project

        self._project_stft_steps = self.project.frames // stft_config.step + 1

        self.pad = pad
        # Don't allow an active range larger than the total number of samples in project
        self.n_active = min(n_active, self._project_stft_steps)
        self.n_cache_total = min(self.n_active + 2 * self.pad, self._project_stft_steps - self.n_active)

        self.config = stft_config

        # What index in the project the start of the cache corresponds to (relative to project)
        self._requested_pos = StftIndex(project, stft_config.step, 0)
        # What index should the active window start on (relative to project)
        self._start_ptr = StftIndex(project, stft_config.step, 0)

        self._max_freq = (self.project.sampling_rate * 0.5) * (1 - (1 / (2 * self.config.window + 1)))
        self._data = np.zeros((self.n_cache_total, self.project.channels, 2 * self.config.window + 1))
        self._stale = np.ones(len(self._data)).astype(np.bool)

        self.queue = Queue()
        self._worker = StftWorker(self.queue, project, self.config)
        self._worker.resultReady.connect(self._on_worker_result)
        self._worker.start()

        self.destroyed.connect(self.on_destroy)

    def on_destroy(self):
        self.queue.put(StftThread.CANCEL)
        self._worker.cancel()

    def get_cache_range_from_active_position(self, pos: StftIndex) -> Tuple[StftIndex, StftIndex]:
        """Get the cache range from the active range's start position
        """
        potential_start = pos - self.pad
        start = max(potential_start, StftIndex(self.project, self.config.step, 0))
        stop = min(potential_start + self.n_cache_total, StftIndex(self.project, self.config.step, self._project_stft_steps))

        return (start, stop)

    def get_active_range_from_active_position(self, pos: StftIndex) -> Tuple[StftIndex, StftIndex]:
        return (pos, pos + self.n_active)

    @property
    def _max_start_ptr(self) -> StftIndex:
        if self.project.frames // self.config.step < self.total_size:
            return self._min_start_ptr
        else:
            return StftIndex(
                self.project,
                self.config.step,
                (self.project.frames // self.config.step) - self.total_size
            )

    def set_position(self, pos: StftIndex):
        """Move the left edge of the active range to the given position
        """
        if int(pos) < 0 or int(pos) > self._project_stft_steps - self.n_active:
            raise ValueError

        # First determine hwo much to move the cache
        new_cache_bounds = self.get_cache_range_from_active_position(pos)

        offset = new_cache_bounds[0] - self._start_ptr

        self._data = np.roll(self._data, -offset, axis=0)
        self._stale = np.roll(self._stale, -offset)

        n = np.abs(offset)
        if offset < 0:
            self._data[:n] = 0
            self._stale[:n] = True
        elif offset > 0:
            self._data[-n:] = 0
            self._stale[-n:] = True

        self._start_ptr = new_cache_bounds[0]
        self._requested_pos = pos

        self._trigger_jobs()

    def _on_worker_result(self, idx: StftIndex, ch: int, fft_result):
        if 0 <= int(idx - self._start_ptr) < len(self._data):
            self._data[idx - self._start_ptr, ch] = fft_result
            self._stale[idx - self._start_ptr] = False

    def _trigger_jobs(self):
        # First clear out the worker's queue
        self._worker.cancel()
        while not self._worker.queue.empty():
            try:
                self._worker.queue.get(False)
            except Empty:
                continue
        self._worker.queue.put(StftWorker.CLEAR)

        jobs = []

        # First within the active range
        active_bounds = self.get_active_range_from_active_position(self._requested_pos)
        cache_bounds = self.get_cache_range_from_active_position(self._requested_pos)

        request_ranges = [
            (active_bounds[0], active_bounds[1]),
            (cache_bounds[0], active_bounds[0]),
            (active_bounds[1], cache_bounds[1])
        ]


        for start, stop in request_ranges:
            current_range_start = None
            for stft_idx in StftIndex.range(start, stop):
                i = stft_idx - self._start_ptr
                if self._stale[i] and current_range_start is None:
                    current_range_start = stft_idx
                elif not self._stale[i] and current_range_start is not None:
                    jobs.append((current_range_start, stft_idx))
                    current_range_start = None
            if current_range_start is not None:
                jobs.append((current_range_start, stft_idx + 1))

        logger.debug("Requesting STFT data from {}".format(jobs))
        print("requsting jobs {}".format(jobs))

        for job in jobs:
            self._worker.queue.put((
                StftIndex(self.project, self.config.step, job[0]),
                StftIndex(self.project, self.config.step, job[1])
            ))

    def read(self): # , start: StftIndex, stop: StftIndex):
        """Read from start (inclusive) to stop (exclusive)

        Returns
        -------
        data : np.ndarray[float]
            The 2D array of the stft values from start to stop. Values not in the cache are return as 0
        stale : np.ndarray[bool]
            Boolean indicator of whether a returned index is valid or not computed yet
        """
        x0, x1 = self.get_active_range_from_active_position(self._requested_pos)
        print("req dat from", x0, x1)
        # if start < x0 or stop > x1:
        #     raise ValueError("Attempting read outside of current Cache values. Call StftCache.set_position first?")

        # i0 = start - x0
        # i1 = stop - x0
        i0 = x0 - self._start_ptr
        i1 = x1 - self._start_ptr

        return self._data[i0:i1], self._stale[i0:i1]
