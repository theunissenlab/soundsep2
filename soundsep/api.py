"""Thin wrapper around the main app controller
"""
import functools
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import numpy as np

from soundsep.app.app import SoundsepController
from soundsep.app.services import SourceService, Selection
from soundsep.core.models import ProjectIndex, StftIndex, Source


logger = logging.getLogger(__name__)


def require_project_loaded(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self._app.has_project_loaded():
            warnings.warn("Cannot call method {} of API without project loaded".format(fn.__name__))
            return
        return fn(self, *args, **kwargs)
    return wrapper


class SoundsepControllerApi(QObject):
    """Soundsep Controller API exposed to plugins and GUI
    """

    projectLoaded = pyqtSignal()
    projectClosed = pyqtSignal()
    workspaceChanged = pyqtSignal(StftIndex, StftIndex)
    sourcesChanged = pyqtSignal(SourceService)
    selectionChanged = pyqtSignal()

    def __init__(self, app: SoundsepController):
        super().__init__()
        self._app = app

        self._cache = {}
        self._plugins = {}

    def load_project(self, directory: Path):
        """Load a new project from a directory

        Arguments
        ---------
        directory : pathlib.Path
            Base directory to load project from. Reads config file if it exists.
        """
        try:
            self._app.load_project(directory)
        except Exception as e:
            logger.error("Failed to open project {}".format(directory))
            self._app.clear_project()
            self.projectClosed.emit()
        else:
            self.projectLoaded.emit()

    @require_project_loaded
    def get_current_project(self):
        return self._app.project

    @require_project_loaded
    def plugins(self):
        return self._app.plugins

    @require_project_loaded
    def get_mut_datastore(self):
        return self._app.datastore

    @require_project_loaded
    def make_project_index(self, i: Union[int, float]) -> ProjectIndex:
        """Shortcut for generating a ProjectIndex by rounding a normal int or float

        Arguments
        ---------
        i : int or float

        Returns
        -------
        pidx : ProjectIndex
        """
        if isinstance(i, float):
            i = np.round(i)

        return ProjectIndex(self._app.project, int(i))

    @require_project_loaded
    def close_project(self):
        self._app.clear_project()
        self.projectClosed.emit()

    @require_project_loaded
    def load_sources(self, save_file: Path):
        """Load sources from a csv file on disk
        """
        self._app.load_sources(save_file)
        self.sourcesChanged.emit(self._app.sources)

    @require_project_loaded
    def create_blank_source(self) -> Source:
        """Shorthand for creating a new blank source on the next available channel

        Shorthand for api.create_source("New Source {ch}", ch), where ch is the next
        channel that does not have a source yet, or channel 0 if all channels
        have been used.

        Returns
        -------
        new_source : Source
        """
        if len(self._app.sources):
            next_channel = int(np.max([s.channel for s in self._app.sources])) + 1
        else:
            next_channel = 0

        if next_channel >= self._app.project.channels:
            next_channel = 0

        new_source = self._app.sources.create("New Source {}".format(next_channel), next_channel)
        self.sourcesChanged.emit(self._app.sources)
        return new_source

    @require_project_loaded
    def create_source(self, source_name: str, source_channel: int) -> Source:
        """Append a new source

        Arguments
        ---------
        source_name : str
        source_channel : int

        Returns
        -------
        new_source : Source
        """
        new_source = self._app.sources.create(source_name, source_channel)
        self.sourcesChanged.emit(self._app.sources)
        return new_source

    @require_project_loaded
    def edit_source(self, source_index: int, source_name: str, source_channel: int) -> Source:
        """Edit an existing source by index

        Arguments
        ---------
        source_index : int
        source_name : str
        source_channel : int

        Returns
        -------
        modified_source : Source
        """
        modified_source = self._app.sources.edit(index, source_name, source_channel)
        self.sourcesChanged.emit(self._app.sources)
        return modified_source

    @require_project_loaded
    def delete_source(self, source_index: int):
        """Delete an existing source by index

        Arguments
        ---------
        source_index : int
        """
        self._app.sources.delete(index)
        self.sourcesChanged.emit(self._app.sources)

    @require_project_loaded
    def get_sources(self) -> List[Source]:
        return list(self._app.sources)

    @require_project_loaded
    def read_config(self):
        return self._app.read_config()

    @require_project_loaded
    def workspace_move_to(self, start: StftIndex):
        """Placeholder

        Arguments
        ---------
        start : StftIndex
        """
        prev_position = self._app.workspace.start
        self._app.workspace.move_to(start)
        self._app.stft.set_position(self._app.workspace.start)
        self._cache["get_workspace_signal"] = None
        if prev_position != start:
            self.workspaceChanged.emit(*self._app.workspace.get_lim(StftIndex))

    @require_project_loaded
    def workspace_move_by(self, dx: int):
        """Placeholder

        Arguments
        ---------
        dx : int
        """
        prev_position = self._app.workspace.start
        self._app.workspace.move_by(dx)
        self._app.stft.set_position(self._app.workspace.start)
        self._cache["get_workspace_signal"] = None
        if dx != 0:
            self.workspaceChanged.emit(*self._app.workspace.get_lim(StftIndex))

    @require_project_loaded
    def workspace_scale(self, n: int):
        """Placeholder

        Arguments
        ---------
        n : int
        """
        prev_size = self._app.workspace.size
        self._app.workspace.scale(n)
        self._app.stft.set_position(self._app.workspace.start)
        self._cache["get_workspace_signal"] = None
        if n != 0:
            self.workspaceChanged.emit(*self._app.workspace.get_lim(StftIndex))

    @require_project_loaded
    def workspace_get_lim(self) -> Tuple:
        """Placeholder
        """
        return self._app.workspace.get_lim(StftIndex)

    @require_project_loaded
    def get_workspace_stft(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the STFT values of the current workspace

        Returns tuple right now?

        Returns
        -------
        result : np.ndarray[float]
        stale : np.ndarray[bool]
        freqs : np.ndarray[float]
            Frequency axis of data
        """
        xlim = self._app.workspace.get_lim(StftIndex)
        data, stale = self._app.stft.read(*xlim)
        return data, stale, self._app.stft.get_freqs()

    @require_project_loaded
    def get_workspace_signal(self) -> np.ndarray:
        """Return the xrange and signal of the current workspace

        Returns
        -------
        result : np.ndarray[float]
        """
        # TODO: clean this up into a service maybe?
        # this is a short quick hacky way to stop reading from disk too much
        if self._cache.get("get_workspace_signal"):
            return self._cache["get_workspace_signal"]

        return self._cache_workspace_signal()

    def _cache_workspace_signal(self):
        xlim = self._app.workspace.get_lim(ProjectIndex)
        data = self._app.project[xlim[0]:xlim[1]]
        result = list(ProjectIndex.range(xlim[0], xlim[1])), data
        self._cache["get_workspace_signal"] = result
        return result

    @require_project_loaded
    def get_signal(self, x0: ProjectIndex, x1: ProjectIndex) -> Tuple[np.ndarray, np.ndarray]:
        """Return the signal in the given range

        Arguments
        ---------
        x0 : ProjectIndex
        x1 : ProjectIndex

        Returns
        -------
        result : np.ndarray[float]
        """
        # TODO: Clean this up along with the other workspace signal cache cleanup
        # For now, use it to see if we can just read from the cache
        xlim = self._app.workspace.get_lim(ProjectIndex)
        # print(x0, x1, xlim, self._cache)
        if x0 >= xlim[0] and x1 <= xlim[1]:
            if not self._cache.get("get_workspace_signal"):
                self._cache_workspace_signal()
            cached_t, cached_data = self._cache["get_workspace_signal"]
            i0 = x0 - xlim[0]
            i1 = x1 - xlim[0]
            return cached_t[i0:i1], cached_data[i0:i1]

        data = self._app.project[x0:x1]
        return list(ProjectIndex.range(x0, x1)), data

    @require_project_loaded
    def workspace_set_position(self, start: StftIndex, stop: StftIndex):
        """Placeholder

        Arguments
        ---------
        start : StftIndex
        stop : StftIndex
        """
        old_position = self._app.workspace.get_lim(StftIndex)
        self._app.workspace.set_position(start, stop)
        self._app.stft.set_position(start)
        self._cache["get_workspace_signal"] = None
        if old_position[0] != start or old_position[1] != stop:
            self.workspaceChanged.emit(*self._app.workspace.get_lim(StftIndex))

    @require_project_loaded
    def set_selection(
            self,
            x0: ProjectIndex,
            x1: ProjectIndex,
            f0: float,
            f1: float,
            source: Source,
        ):
        self._app.selection.set_selection(x0, x1, f0, f1, source)
        self.selectionChanged.emit()

    @require_project_loaded
    def clear_selection(self):
        self._app.selection.clear()
        self.selectionChanged.emit()

    @require_project_loaded
    def get_selection(self) -> Optional[Selection]:
        if self._app.selection.is_set():
            return self._app.selection.get_selection()
        else:
            return None

    def filter_and_ampenv(
            self,
            signal: np.ndarray,
            f0: float,
            f1: float
        ) -> np.ndarray:
        # TODO: seems a bit weird to include this in api but maybe its okay
        # TODO: also, where should the rectivy lowpass be configured?
        if f0 < 250:
            f0 = 250
        if f1 > 10000:
            f1 = 10000

        return self._app.ampenv.filter_and_ampenv(signal, f0, f1, 200.0)



class SoundsepGuiApi(QObject):
    """Soundsep GUI API exposed to plugins
    """

    def __init__(self, gui: "SoundsepGui"):
        super().__init__()
        self._gui = gui

    def _set_gui(self, gui):
        self._gui = gui

    def show_status(self, msg, timeout=1000):
        if not self._gui:
            return
        self._gui.show_status(msg)

    def get_source_views(self):
        if not self._gui:
            return
        return self._gui.source_views

    def get_preview_plot(self):
        if not self._gui:
            return
        return self._gui.preview_plot_widget
