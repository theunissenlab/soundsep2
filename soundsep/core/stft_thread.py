from queue import Queue

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from scipy.fft import fft, fftfreq, next_fast_len

from soundsep.core.models import StftIndex, Project

from soundsep.core.stft import _create_gaussian_window

class StftConfig(dict):
    def __init__(self, window, step):
        self.window = window
        self.step = step


class StftWorker(QThread):
    END = object()
    requestStft = pyqtSignal(StftIndex, StftIndex)
    resultReady = pyqtSignal(StftIndex, int, object)

    def __init__(self, queue: Queue, project: Project, config: StftConfig):
        super().__init__()
        self.queue = queue
        self.project = project
        self.config = config

        self.gaussian_window = _create_gaussian_window(self.config.window, nstd=6)

        self.requestStft.connect(self.on_request)
        self._cancel_flag = False

    def cancel(self):
        self._cancel_flag = True

    def run(self):
        while True:
            q = self.queue.get()
            if q == StftWorker.END:
                break
            self.process_request(q[0], q[1])

    def process_request(self, start: StftIndex, stop: StftIndex):
        # TODO: read the necessary data so we aren't constantly going to disk?
        for window_center_stft in StftIndex.range(start, stop):
            window_center = window_center_stft.to_project_index()
            window_start = max(ProjectIndex(self.project, 0), window_center - half_window)
            window_stop = min(ProjectIndex(self.project, self.project.frames), window_center + half_window + 1)

            if int(window_start) >= self.project.frames:
                break

            for ch in range(self.project.channels):
                if self._cancel_flag:
                    # End this loop and empty the queue
                    self._cancel_flag = False
                    while not self.queue.empty():
                        try:
                            self.queue.get(False)
                        except Empty:
                            continue
                    return

                window_data = self.project[window_start:window_stop]

                if int(window_start) == 0:
                    window_data *= gaussian_window[-window_data.size:]
                elif int(window_stop) == self.project.frames:
                    window_data *= self.gaussian_window[:window_data.size]
                else:
                    window_data *= self.gaussian_window

                # TODO: benchmark using next_fast_len here?
                window_fft = fft(window_data)

                self.resultReady.emit(window_center, ch, window_fft)
    #
    # def run(self):
    #     while True:
    #         q = q.get()
    #         for interval in q:
    #             if cancel_sigal:
    #                 q.flush()
    #                 break
    #             run
    #             emit()
    #


class StftCache(QObject):
    """A cache for stft values
    """
    def __init__(self, project, visible_size: int, pad: int, stft_config: StftConfig):
        super().__init__()

        self.pad = pad
        self.visible_size = visible_size
        self.total_size = self.pad + self.visible_size

        self.config = stft_config

        self._start_ptr = StftIndex(project, stft_config.step, 0)
        self._max_freq = (project.sampling_rate * 0.5) * (1 - (1 / (2 * self.config.window + 1)))
        self._data = np.zeros((2 * pad + visible_size, 2 * self.config.window + 1))
        self._stale = np.ones(len(self._data)).astype(np.bool)

        self.queue = Queue()
        self._worker = StftWorker(self.queue, project, self.config)
        self._worker.start()
        self._worker.resultReady.connect(self._on_worker_result)

        self.destroyed.connect(self.on_destroy)

    def on_destroy(self):
        self.queue.put(StftThread.CANCEL)
        self._worker.cancel()

    @property
    def min_position(self) -> StftIndex:
        return StftIndex(self.project, self.config.step, self.pad)

    @property
    def max_position(self) -> StftIndex:
        if self.project.frames // self.config.step < self.total_size:
            return self.min_position
        else:
            return StftIndex(
                self.project,
                self.config.step,
                (self.project.frames // self.config.step) - self.pad - self.visible_size
            )

    def _current_bounds(self):
        return self._start_ptr, self._start_ptr + self.total_size

    def set_position(self, pos: StftIndex):
        """Move the read position of the StftCache to pos
        """
        if pos < self.min_position:
            pos = self.min_position
        elif pos > self.max_position:
            pos = self.max_position

        current_position = self._start_ptr + self.pad
        offset = pos - current_position

        self._data = np.roll(self._data, -offset, axis=0)
        self._stale = np.roll(self._stale, -offset)
        if offset < 0:
            self._data[:n] = 0
            self._stale[:n] = True
        elif offset > 0:
            self._data[-n:] = 0
            self._stale[-n:] = True

        self._start_ptr = self._start_ptr + offset

        self._trigger_jobs(self)

    def _on_worker_result(self, idx: StftIndex, ch: int, fft_result):
        if 0 <= int(idx - self._start_ptr) < len(self._data):
            self._data[idx - self._start_ptr, ch] = fft_result
            self._stale[idx - self._start_ptr] = False

    def _trigger_jobs(self):
        # First within the visible range
        jobs = []
        current_range_start = None
        for i in StftIndex.range(self._start_ptr + self.pad, self._start_ptr + self.pad + self.visible_size):
            if self._stale[i] and current_range_start is None:
                current_range_start = i
            elif not self._stale[i] and current_range_start is not None:
                jobs.append((current_range_start, i))
                current_range_start = None
        if current_range_start is not None:
            jobs.append((current_range_start, i + 1))

        for i in StftIndex.range(self._start_ptr, self._start_ptr + self.pad):
            if self._stale[i] and current_range_start is None:
                current_range_start = i
            elif not self._stale[i] and current_range_start is not None:
                jobs.append((current_range_start, i))
                current_range_start = None
        if current_range_start is not None:
            jobs.append((current_range_start, i + 1))

        for i in StftIndex.range(
                self._start_ptr + self.pad + self.visible_size,
                self._start_ptr + self.pad + self.pad + self.visible_size):
            if self._stale[i] and current_range_start is None:
                current_range_start = i
            elif not self._stale[i] and current_range_start is not None:
                jobs.append((current_range_start, i))
                current_range_start = None
        if current_range_start is not None:
            jobs.append((current_range_start, i + 1))

        for job in jobs:
            self._worker.queue.put(
                StftIndex(self.project, self.config.step, job[0]),
                StftIndex(self.project, self.config.step, job[1])
            )

    def read(self, start: StftIndex, stop: StftIndex):
        """Read from start (inclusive) to stop (exclusive)

        Returns
        -------
        data : np.ndarray[float]
            The 2D array of the stft values from start to stop. Values not in the cache are return as 0
        stale : np.ndarray[bool]
            Boolean indicator of whether a returned index is valid or not computed yet
        """
        x0, x1 = self._current_bounds()
        if start < x0 or stop > x1:
            raise ValueError("Attempting read outside of current Cache values. Call StftCache.set_position first?")

        i0 = start - x0
        i1 = stop - x0

        return self._data[i0:i1], self._stale[i0:i1]
