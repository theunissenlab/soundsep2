import logging
from collections import namedtuple
from queue import Queue
from typing import Optional, Tuple, Union

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from scipy.fft import fft, fftfreq, next_fast_len
import numpy as np

from soundsep.core.models import Project, ProjectIndex, Source, StftIndex
from soundsep.core.ampenv import filter_and_ampenv
from soundsep.core.stft import create_gaussian_window


logger = logging.getLogger(__name__)


Selection = namedtuple("Selection", ["x0", "x1", "f0", "f1", "source"])


class StftConfig(dict):
    def __init__(self, window, step):
        self.window = window
        self.step = step


class SelectionService:
    def __init__(self, project: Project):
        self.project = project
        self._selection = None

    def is_set(self) -> bool:
        return self._selection is not None

    def clear(self):
        self._selection = None

    def set_selection(
            self,
            x0: ProjectIndex,
            x1: ProjectIndex,
            f0: float,
            f1: float,
            source: Source,
        ):
        self._selection = Selection(x0, x1, f0, f1, source)

    def move_selection(self, dx: int):
        self._selection = Selection(
            self._selection.x0 + dx,
            self._selection.x1 + dx,
            self._selection.f0,
            self._selection.f1,
            self._selection.source
        )

    def scale_selection(self, n: int):
        self._selection = Selection(
            self._selection.x0 - n // 2,
            self._selection.x1 + n // 2,
            self._selection.f0,
            self._selection.f1,
            self._selection.source
        )

    def get_selection(self) -> Optional[Selection]:
        return self._selection


class SourceService(list):
    """Placeholder service for Source management"""
    def __init__(self, project: Project):
        self.project = project
        super().__init__()

    def create(self, name: str, channel: int) -> Source:
        print("before", list(self))
        new_source = Source(self.project, name, channel, len(self))
        self.append(new_source)
        print(list(self))
        return new_source

    def edit(self, index: int, name: str, channel: int) -> Source:
        self[index].name = name
        self[index].channel = channel
        return self[index]

    def delete(self, index: int) -> "SourceService":
        del self[index]
        for source in self[index:]:
            source.index -= 1
        return self


class AmpenvService:
    """Simple service that cache's the last computation

    This is useful because often you might want to get data multiple times
    """

    def __init__(self, project: Project):
        self.project = project
        self._last_computation = None

    # TODO: as part of api's get_signal etc cleanup, make AmpenvService do its own caching
    # So it can take start, stop values instead of
    def filter_and_ampenv(
            self,
            signal: np.ndarray,
            f0: float,
            f1: float,
            rectify_lowpass: float,
            ) -> Tuple[np.ndarray, np.ndarray]:

        result = filter_and_ampenv(
            signal,
            self.project.sampling_rate,
            f0,
            f1,
            rectify_lowpass
        )

        return result


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
    resultReady = pyqtSignal(StftIndex, int, object)

    def __init__(self, queue: Queue, project: Project, config: StftConfig):
        super().__init__()
        self.queue = queue
        self.project = project
        self.config = config

        self.gaussian_window = create_gaussian_window(self.config.window, nstd=6)
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

    def process_request(self, start: StftIndex, stop: StftIndex):
        # Load the array with padding so that the first and last windows don't get cut in half
        loaded_arr_start = max(ProjectIndex(self.project, 0), start.to_project_index() - self.config.window)
        loaded_arr_stop = min(ProjectIndex(self.project, self.project.frames), stop.to_project_index() + self.config.window)
        arr = self.project[loaded_arr_start:loaded_arr_stop]

        for window_center_stft in StftIndex.range(start, stop):
            window_center = window_center_stft.to_project_index()
            window_start = max(ProjectIndex(self.project, 0), window_center - self.config.window)
            window_stop = min(ProjectIndex(self.project, self.project.frames), window_center + self.config.window + 1)

            if int(window_start) >= self.project.frames:
                break

            for ch in range(self.project.channels):
                if self._cancel_flag:
                    return

                a = max(0, window_start - loaded_arr_start)
                b = window_stop - loaded_arr_start
                window_data = arr[a:b, ch]

                if a <= 0:
                    window_data = window_data * self.gaussian_window[-window_data.size:]
                elif b >= len(arr):
                    window_data = window_data * self.gaussian_window[:window_data.size]
                else:
                    window_data = window_data * self.gaussian_window

                # TODO: benchmark using next_fast_len here?
                window_fft = np.abs(fft(window_data, n=2 * self.config.window + 1))

                self.resultReady.emit(window_center_stft, ch, window_fft)


class StftCache(QObject):
    """A service for caching stft values in and around an active data range

    The StftCache defines a range of data called the "active range" which
    is its primary read region. In the main application this corresponds to the
    region of data visible to the user. This region has width `self.n_active` in
    units of the stft step, and starts at `self._pos`.

    The StftCache also defines a range of data surrounding the "active range"
    called the "cache range". This typically consists of the active range plus
    `self.pad` windows before and after. When the active range is close to the
    endpoints of the project, the endpoints of the cache align with the edge
    of the project such that the size of the "cache range" remains constant.

    Attributes
    ----------
    project : Project
    pad : int
        Requested width of one side of the cache range
    n_active : int
        Requested width of the active region. Will only be shorter if the project
        is not long enough to accomodate it, in which case the entire project
        will lie in the active region
    n_cache_total : int
        The total width of the cache region.
    config : StftConfig
        The stft config parameters including stft.window and stft.width
    _pos : StftIndex
        The index in the project that the start of the active region corresponds to
    _start_ptr : StftIndex
        The index in the project that the start of the cache region corresponds to
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

        # What index in the project the start of the active region corresponds to
        self._pos = StftIndex(project, stft_config.step, 0)
        # What index should the active window start on (relative to project)
        self._start_ptr = StftIndex(project, stft_config.step, 0)

        self._max_freq = (self.project.sampling_rate * 0.5) * (1 - (1 / (2 * self.config.window + 1)))
        self._data = np.zeros((self.n_cache_total, self.project.channels, 2 * self.config.window + 1))
        self._stale = np.ones(len(self._data)).astype(np.bool)

        self.queue = Queue()
        self._worker = StftWorker(self.queue, project, self.config)
        self._worker.resultReady.connect(self._on_worker_result)
        self._worker.start()

        self._trigger_jobs()  # Force it to populate the initial section
        # self.destroyed.connect(self.on_destroy)

    def max_freq(self):
        """Return the largest frequency value in all the land"""
        return self._max_freq

    def get_freqs(self):
        """Return the frequency range of the ffts, this includes negative frequencies"""
        return fftfreq(2 * self.config.window + 1, 1 / self.project.sampling_rate)

    def get_cache_range_from_active_position(self, pos: StftIndex) -> Tuple[StftIndex, StftIndex]:
        """Get the cache range from the active range's start position
        """
        potential_start = pos - self.pad
        start = max(potential_start, StftIndex(self.project, self.config.step, 0))
        stop = min(potential_start + self.n_cache_total, StftIndex(self.project, self.config.step, self._project_stft_steps))

        return (start, stop)

    def get_active_range_from_active_position(self, pos: StftIndex) -> Tuple[StftIndex, StftIndex]:
        """Get the active range from the active range's start position
        """
        return (pos, pos + self.n_active)

    def set_position(self, pos: StftIndex):
        """Move the left edge of the active range to the given position
        """
        if int(pos) < 0 or int(pos) > self._project_stft_steps - self.n_active:
            raise ValueError

        if pos == self._pos:
            return

        # Determine how far to slide the cached data over
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
        self._pos = pos

        self._trigger_jobs()

    def _on_worker_result(self, idx: StftIndex, ch: int, fft_result):
        if 0 <= int(idx - self._start_ptr) < len(self._data):
            self._data[idx - self._start_ptr, ch] = fft_result
            self._stale[idx - self._start_ptr] = False

    def _trigger_jobs(self):
        """Determine what regions of data are stale and queue jobs for the worker to fill them in
        """
        # First clear out the worker's queue
        self._worker.cancel()

        jobs = []

        # First within the active range
        active_bounds = self.get_active_range_from_active_position(self._pos)
        cache_bounds = self.get_cache_range_from_active_position(self._pos)

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

        for job in jobs:
            self._worker.queue.put((
                StftIndex(self.project, self.config.step, job[0]),
                StftIndex(self.project, self.config.step, job[1])
            ))

    def read(self, start: Optional[StftIndex] = None, stop: Optional[StftIndex] = None):
        """Read data from start (inclusive) to stop (exclusive)

        Arguments
        ---------
        start : Optional[StftIndex] (default None)
        stop : Optional[StftIndex] (default None)

        Returns
        -------
        data : np.ndarray[float]
            The 2D array of the stft values from start to stop. Values not in the cache are return as 0
        stale : np.ndarray[bool]
            Boolean indicator of whether a returned index is valid or not computed yet
        """
        x0, x1 = self.get_active_range_from_active_position(self._pos)
        start = start or x0
        stop = stop or x1

        cache_bounds = self.get_cache_range_from_active_position(self._pos)

        logger.debug("Reading cache from {} to {} out of {}".format(start, stop, cache_bounds))

        if start < cache_bounds[0] or stop > cache_bounds[1]:
            raise ValueError("Attempting read outside of current Cache values. Call StftCache.set_position first?")

        start_idx = start - self._start_ptr
        stop_idx = stop - self._start_ptr

        return self._data[start_idx:stop_idx], self._stale[start_idx:stop_idx]
