from collections import namedtuple
from enum import Enum
from typing import Optional, Tuple, Union

from PyQt6.QtCore import QObject
import numpy as np

from soundsep.core.models import Project, ProjectIndex, Source, StftIndex
from soundsep.core.ampenv import filter_and_ampenv


Selection = namedtuple("Selection", ["x0", "x1", "f0", "f1", "source"])


class SelectionService:
    """Keeps track of selection state, a rectangle of time and frequency bounds

    Includes a "fine selection", which is a smaller ROI on top of the main roi
    that can be accessed with get_fine_selection()

    Includes a "segment selection", which is a single segment ID
    """
    def __init__(self, project: Project):
        self.project = project
        self._selection = None
        self._fine_selection = None
        self._segment_selection = []

    def is_set(self) -> bool:
        return self._selection is not None

    def clear(self):
        self._selection = None
        self._fine_selection = None

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

    # Fine selection
    def set_fine_selection(self, x0: ProjectIndex, x1: ProjectIndex):
        if not self._selection:
            raise ValueError("Cannot set a fine selection if no current selection exists")

        self._fine_selection = Selection(x0, x1, self._selection.f0, self._selection.f1, self._selection.source)

    def clear_fine_selection(self):
        self._fine_selection = None

    def get_fine_selection(self) -> Optional[Selection]:
        return self._fine_selection or self._selection

    # Segment selection
    def set_segment_selection( self, segIDs):
        self._segment_selection = segIDs.copy()
    
    def get_segment_selection(self):
        return self._segment_selection.copy()
    
    def clear_segment_selection(self):
        self._segment_selection = []


class SourceService(list):
    """Placeholder service for Source management"""
    def __init__(self, project: Project):
        self.project = project
        self._needs_saving = False
        super().__init__()

    @property
    def _existing_names(self):
        return set([s.name for s in self])

    def create(self, name: str, channel: int) -> Source:
        if name in self._existing_names:
            raise ValueError("Cannot create a Source with a non-unique name")

        new_source = Source(self.project, name, channel, len(self))
        self.append(new_source)
        self._needs_saving = True
        return new_source

    def edit(self, index: int, name: str, channel: int) -> Source:
        self[index].name = name
        self[index].channel = channel
        self._needs_saving = True
        return self[index]

    def delete(self, index: int) -> "SourceService":
        del self[index]
        for source in self[index:]:
            source.index -= 1
        self._needs_saving = True
        return self

    def set_needs_saving(self, flag: bool):
        self._needs_saving = flag

    def needs_saving(self) -> bool:
        return self._needs_saving

    def create_template_source(self) -> 'Source':
        """Creates a new source with a generic name on the next available channel
        """
        if len(self):
            next_channel = int(np.max([s.channel for s in self])) + 1
        else:
            next_channel = 0

        if next_channel >= self.project.channels:
            next_channel = 0

        existing_names = self._existing_names
        # Try names until we get a unique one
        name_idx = next_channel
        try_name = "New Source {}".format(name_idx)
        while try_name in existing_names:
            name_idx += 1
            try_name = "New Source {}".format(name_idx)

        return self.create(try_name, next_channel)


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


class Workspace(QObject):
    """Representation of the current working time range in StftIndex units

    The workspace is represented by a start index (inclusive) and end index (non-inclusive)
    """

    class Alignment(Enum):
        Left = "left"
        Center = "center"
        Right = "right"

    def __init__(self, start: StftIndex, stop: StftIndex):
        super().__init__()
        if start.project != stop.project:
            raise TypeError("Cannot instantiate Workspace with StftIndex values from different projects")
        if start.step != stop.step:
            raise TypeError("Cannot instantiate Workspace with StftIndex values with different step sizes")

        self.start = start
        self.stop = stop
        self.set_position(start, stop)

    def __repr__(self):
        return "Workspace<{}, {}>".format(self.start, self.stop)

    @property
    def project(self) -> Project:
        """The project the Workspace is referencing"""
        return self.start.project

    @property
    def step(self) -> int:
        """The step size of the Workspace's StftIndex units"""
        return self.start.step

    @property
    def max_size(self) -> int:
        """Total number of StftIndex frames available in project"""
        return (self.project.frames // self.step) + 1

    @property
    def min_index(self) -> StftIndex:
        return StftIndex(self.project, self.step, 0)

    @property
    def max_index(self) -> StftIndex:
        return StftIndex(self.project, self.step, self.max_size)

    @property
    def size(self) -> int:
        """Return the size of the Workspace in StftIndex units"""
        return self.stop - self.start

    def move_to(self, start: StftIndex):
        """Move the starting point of the Workspace to the given index, preseving size

        Movement will stop when an endpoint is reached

        Arguments
        ---------
        start : StftIndex
        """
        dx = start - self.start
        return self.move_by(dx)

    def move_by(self, dx: int):
        """Move the starting point of the Workspace by the given amount, preserving size

        Movement will stop when an endpoint is reached

        Arguments
        ---------
        dx : int
        """
        self.set_position(self.start + dx, self.stop + dx, preserve_requested_size=True)

    def scale(self, n: int):
        """Increase or decrease the extent of this Workspace by n StftIndex units

        The increments are made alternating endpoints (starting with self.stop) and
        increasing/decreasing the size of the Workspace until the size has changed by n StftIndex
        units. If the endpoints reach 0 or the end of the project, the remainder is added
        to the other end.

        Arguments
        ---------
        n : int
        """
        if n < 0:
            n = max(n, 1 - self.size)
        else:
            n = min(n, self.max_size - self.size)

        start = int(self.start)
        stop = int(self.stop)
        sign = np.sign(n)

        for i in range(abs(n)):
            if i % 2 == 0:
                if int(stop) < self.max_size:
                    stop += sign
                else:
                    start -= sign
            else:
                if int(start) > 0:
                    start -= sign
                else:
                    stop += sign

        self.set_position(StftIndex(self.project, self.step, start), StftIndex(self.project, self.step, stop))

    def get_lim(self, as_: Union[ProjectIndex, StftIndex]) -> Tuple:
        if as_ == ProjectIndex:
            return (self.start.to_project_index(), self.stop.to_project_index())
        elif as_ == StftIndex:
            return (self.start, self.stop)
        else:
            raise TypeError

    def set_position(self, start: StftIndex, stop: StftIndex, preserve_requested_size: bool = False):
        """Attempt to set a new start and stop position

        Arguments
        ---------
        start : StftIndex
        stop : StftIndex
        preserve_requested_size : bool (default False)
            If False, will truncate the endpoints if they flow beyond the ends of the project.
            If True, if the Workspace would overflow the bounds of the project, will adjust the start
            or stop points to guarantee the Workspace size equals the requested stop - start.
        """
        new_start = self.min_index if start < self.min_index else start
        new_stop = self.max_index if stop > self.max_index else stop

        if stop - start < 1:
            raise ValueError("Workspace stop must be after start: got {} to {}".format(start, stop))

        if preserve_requested_size:
            requested_size = stop - start
            if new_stop - new_start == requested_size:
                self.start = new_start
                self.stop = new_stop
            elif new_start == self.min_index:
                self.start = new_start
                self.stop = min(new_start + requested_size, self.max_index)
            elif new_stop == self.max_index:
                self.start = max(new_stop - requested_size, self.min_index)
                self.stop = new_stop
        else:
            self.start = new_start
            self.stop = new_stop
