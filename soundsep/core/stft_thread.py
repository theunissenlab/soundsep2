from PyQt5.QtCore import QObject, QThread, pyqtSignal



class StftWorker(QThread):
    resultReady = pyqtSignal(StftIndex, object)

    def __init__(self, project: Project, config: StftConfig):
        super().__init__()
        
    def run(self):
        while True:
            q = q.get()
            for interval in q:
                if cancel_sigal:
                    q.flush()
                    break
                run
                emit()



class StftCache:
    """A cache for stft values
    """
    def __init__(self, project, visible_size: int, pad: int, stft_config: StftConfig):
        self.pad = pad
        self.visible_size = visible_size
        self.total_size = self.pad + self.visible_size

        self.config = stft_config

        self._start_ptr = StftIndex(project, stft_config.step, 0)
        self._max_freq = (project.sampling_rate * 0.5) * (1 - (1 / (2 * self.config.window + 1)))
        self._data = np.zeros((2 * pad + visible_size, 2 * self.config.window + 1))
        self._stale = np.ones(len(self._data)).astype(np.bool)

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

    def _trigger_jobs(self):
        pass

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
