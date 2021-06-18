"""Thin wrapper around the main app controller
"""
import functools
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import numpy as np

from soundsep.app.app import SoundsepController
from soundsep.app.services import SourceService, Selection, Workspace
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



class Api(QObject):

    # These signals are propogated up to the Loader
    _closeProject = pyqtSignal()
    _switchProject = pyqtSignal(Path)

    # These signals are public
    projectLoaded = pyqtSignal()
    sourcesChanged = pyqtSignal()
    configChanged = pyqtSignal(object)  # Not implemented yet
    workspaceChanged = pyqtSignal()
    selectionChanged = pyqtSignal()

    def __init__(self, app: SoundsepController):
        super().__init__()
        self._app = app
        self._cache = {}
        self._app.configChanged.connect(self.configChanged)

    @property
    def paths(self):
        return self._app.paths

    @property
    def project(self):
        return self._app.project

    @property
    def plugins(self):
        return self._app.plugins

    @property
    def config(self):
        return self._app.config

    def _close(self):
        self._app.close()

    def switch_project(self, project_dir: Path):
        """Attempt to switch project to a new directory

        Arguments
        ---------
        project_dir : pathlib.Path

        Emits
        -----
        _switchProject
        """
        self._switchProject.emit(project_dir)

    def get_mut_datastore(self) -> dict:
        """Return the mutable datastore dictionary

        Returns
        -------
        datastore : dict
        """
        return self._app.datastore

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

    def create_blank_source(self) -> Source:
        """Shorthand for creating a new blank source on the next available channel

        Shorthand for api.create_source("New Source {ch}", ch), where ch is the next
        channel that does not have a source yet, or channel 0 if all channels
        have been used.

        Returns
        -------
        new_source : Source

        Emits
        -----
        sourcesChanged
        """
        new_source = self._app.datastore["sources"].create_template_source()
        self.sourcesChanged.emit()
        return new_source

    def create_source(self, source_name: str, source_channel: int) -> Source:
        """Append a new source

        Arguments
        ---------
        source_name : str
        source_channel : int

        Returns
        -------
        new_source : Source

        Emits
        -----
        sourcesChanged
        """
        new_source = self._app.datastore["sources"].create(source_name, source_channel)
        self.sourcesChanged.emit()
        return new_source

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

        Emits
        -----
        sourcesChanged
        """
        modified_source = self._app.datastore["sources"].edit(source_index, source_name, source_channel)
        self.sourcesChanged.emit()
        return modified_source

    def delete_source(self, source_index: int):
        """Delete an existing source by index

        Arguments
        ---------
        source_index : int

        Emits
        -----
        sourcesChanged
        """
        self._app.datastore["sources"].delete(source_index)
        self.sourcesChanged.emit()

    def get_sources(self) -> List[Source]:
        """Return a list of sources

        Returns
        -------
        sources : List[Source]
        """
        return list(self._app.datastore["sources"])

    def convert_project_index_to_stft_index(self, idx: ProjectIndex) -> StftIndex:
        """Converts a ProjectIndex to StftIndex based on the current Stft config

        Arguments
        ---------
        idx : ProjectIndex

        Returns
        -------
        stft_idx : StftIndex
        """
        return StftIndex(
            self._app.project,
            self._app.services["stft"].config.step,
            idx // self._app.services["stft"].config.step
        )

    def workspace_move_to(self, start: StftIndex, alignment: Workspace.Alignment = None):
        """Move the current workspace to start at the given index in StftIndex

        Arguments
        ---------
        start : StftIndex
        alignment : Workspace.Alignment (default Workspace.Alignment.Left)

        Emits
        -----
        workspaceChanged
        """
        prev_position = self._app.state["workspace"].start
        self._app.state["workspace"].move_to(start)
        self._app.services["stft"].set_position(self._app.state["workspace"].start)
        self._cache["get_workspace_signal"] = None
        if prev_position != start:
            self.workspaceChanged.emit()

    def workspace_move_by(self, dx: int):
        """Move the current workspace by the given amount in StftIndex units

        Arguments
        ---------
        dx : int
            Positive numbers move the workspace to the right, negative numbers to
            the left

        Emits
        -----
        workspaceChanged
        """
        prev_position = self._app.state["workspace"].start
        self._app.state["workspace"].move_by(dx)
        self._app.services["stft"].set_position(self._app.state["workspace"].start)
        self._cache["get_workspace_signal"] = None
        if dx != 0:
            self.workspaceChanged.emit()

    def workspace_scale(self, n: int):
        """Scale the current workspace by the given number of StftIndex units

        TODO: Add a parameters so scaling can center on a position

        Arguments
        ---------
        n : int

        Emits
        -----
        workspaceChanged
        """
        prev_size = self._app.state["workspace"].size

        max_size = self._app.services["stft"].n_cache_total
        if self._app.state["workspace"].size + n >= max_size:
            n = max_size - self._app.state["workspace"].size

        self._app.state["workspace"].scale(n)
        self._app.services["stft"].set_active_size(self._app.state["workspace"].size)
        self._app.services["stft"].set_position(self._app.state["workspace"].start)
        self._cache["get_workspace_signal"] = None
        if n != 0:
            self.workspaceChanged.emit()

    def workspace_set_position(self, start: StftIndex, stop: StftIndex):
        """Set workspace by epxlicitly setting the start and stop coordinates

        Used if you need more precision than workspace_scale() offers

        Arguments
        ---------
        start : StftIndex
        stop : StftIndex

        Emits
        -----
        workspaceChanged
        """
        old_position = self._app.state["workspace"].get_lim(StftIndex)
        self._app.state["workspace"].set_position(start, stop)
        self._app.service["stft"].set_position(start)
        self._cache["get_workspace_signal"] = None
        if old_position[0] != start or old_position[1] != stop:
            self.workspaceChanged.emit()

    def workspace_get_lim(self) -> Tuple:
        """Get the limits of the current workspace in units of StftIndex

        Returns
        -------
        start : StftIndex
        stop : StftIndex
        """
        return self._app.state["workspace"].get_lim(StftIndex)

    def read_stft(self, start: StftIndex, stop: StftIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the STFT values of the current workspace

        Example
        -------
        >>> start, stop = api.workspace_get_lim()
        >>> stft, stale, freqs = api.read_stft(start, stop)

        Returns
        -------
        result : np.ndarray[float]
            2D array of stft data for the requested range
        stale : np.ndarray[bool]
            Which indices of the requested range have not been filled in
        freqs : np.ndarray[float]
            Frequency axis of data
        """
        data, stale = self._app.services["stft"].read(start, stop)
        return data, stale, self._app.services["stft"].get_freqs()

    def get_workspace_signal(self) -> Tuple[list, np.ndarray]:
        """Return the xrange and signal of the current workspace

        Returns
        -------
        time_axis : List[ProjectIndex]
        result : np.ndarray[float]
        """
        # TODO: clean this up into a service maybe?
        # this is a short quick hacky way to stop reading from disk too much
        if self._cache.get("get_workspace_signal"):
            return self._cache["get_workspace_signal"]

        return self._cache_workspace_signal()

    def _cache_workspace_signal(self) -> Tuple[list, np.ndarray]:
        # TODO: with a smarter caching strategy, we dont't need to load
        # the entire workspace signal at at time, just what we don't already
        # have...
        xlim = self._app.state["workspace"].get_lim(ProjectIndex)
        data = self._app.project[xlim[0]:xlim[1]]
        result = list(ProjectIndex.range(xlim[0], xlim[1])), data
        self._cache["get_workspace_signal"] = result
        return result

    def get_signal(self, x0: ProjectIndex, x1: ProjectIndex) -> Tuple[np.ndarray, np.ndarray]:
        """Return the signal in the given range

        Arguments
        ---------
        x0 : ProjectIndex
        x1 : ProjectIndex

        Returns
        -------
        time_axis : np.ndarray[float]
        result : np.ndarray[float]
        """
        xlim = self._app.state["workspace"].get_lim(ProjectIndex)
        if x0 >= xlim[0] and x1 <= xlim[1]:
            if not self._cache.get("get_workspace_signal"):
                self._cache_workspace_signal()
            cached_t, cached_data = self._cache["get_workspace_signal"]
            i0 = x0 - xlim[0]
            i1 = x1 - xlim[0]
            return cached_t[i0:i1], cached_data[i0:i1]

        data = self._app.project[x0:x1]
        return list(ProjectIndex.range(x0, x1)), data

    def set_selection(
            self,
            x0: ProjectIndex,
            x1: ProjectIndex,
            f0: float,
            f1: float,
            source: Source,
        ):
        """Set the current selection to a bound in time and frequency

        Arguments
        ---------
        x0 : ProjectIndex
        x1 : ProjectIndex
        f0 : float
            Lower frequency bound
        f1 : float
            Upper frequency bound
        source : Source

        Emits
        -----
        selectionChanged
        """
        self._app.state["selection"].set_selection(x0, x1, f0, f1, source)
        self.selectionChanged.emit()

    def clear_selection(self):
        """Clear the current selection

        Emits
        -----
        selectionChanged
        """
        self._app.state["selection"].clear()
        self.selectionChanged.emit()

    def get_selection(self) -> 'Optional[Selection]':
        """Clear the current selection

        Returns
        -------
        selection : Optional[soundsep.app.services.Selection]
        """
        if self._app.state["selection"].is_set():
            return self._app.state["selection"].get_selection()
        else:
            return None

    def needs_saving(self) -> bool:
        """Returns True if sources or any plugins have unsaved changes

        Returns
        -------
        needs_saving : bool
        """
        if self._app.datastore["sources"].needs_saving():
            logger.info("Unsaved changes to sources")
            return True

        for p in self._app.plugins.values():
            if p.needs_saving():
                logger.info("Unsaved changes in plugin {}".format(p.__class__.__name__))
                return True

        return False

    def save(self):
        """Save sources and call save hooks on all plugins

        Returns
        -------
        save_successful : bool
        """
        try:
            self._app.save_sources()
            for name, plugin in self._app.plugins.items():
                plugin.save()
        except Exception as e:
            logger.exception("Exception while saving")
            return False
        else:
            return True

    def filter_and_ampenv(
            self,
            signal: np.ndarray,
            f0: float,
            f1: float
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter and compute ampenv of a signal

        Arguments
        ---------
        signal : np.ndarray
            A 1-D signal of size N samples or a 2-D signal of size (N, K) where
            K is the number of channels in the signal.
        f0 : float
            Lower frequency to filter signal at
        f1 : float
            Upper frequency to filter signal at

        Returns
        -------
        filtered : np.ndarray
            The bandpass filtered signal of the same shape as the input signal
        ampenv : np.ndarray
            The amplitude envelope computed from the filtered signal after
            rectification and lowpass
        """
        # TODO: seems a bit weird to include this in api but maybe its okay
        # TODO: also, where should the rectivy lowpass be configured?
        f0 = max(f0, self._app.config["filter.low"])
        f1 = min(f1, self._app.config["filter.high"])
        if f0 > f1:
            raise ValueError("Cannot filter with f0 > f1")

        return self._app.services["ampenv"].filter_and_ampenv(
                signal, f0, f1, self._app.config["filter.ampenv_rectify"])


class SoundsepControllerApi(QObject):
    """Soundsep Controller API exposed to plugins and GUI
    """

    projectLoaded = pyqtSignal()
    projectClosed = pyqtSignal()
    workspaceChanged = pyqtSignal(StftIndex, StftIndex)
    sourcesChanged = pyqtSignal(SourceService)
    selectionChanged = pyqtSignal()
    projectReady = pyqtSignal()

    def __init__(self, app: SoundsepController):
        super().__init__()
        self._app = app

        self._cache = {}
        self._plugins = {}

        self.projectReady.connect(self._on_project_ready)

    def _on_project_ready(self):
        """Emit signals that apps would have missed
        """
        self.sourcesChanged.emit(self._app.sources)

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
            logger.exception("Failed to open project {}".format(directory))

            self._app.clear_project()
            self.projectClosed.emit()
        else:
            self.projectClosed.emit()
            self.projectLoaded.emit()

    @require_project_loaded
    def paths(self):
        return self._app.paths

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
        modified_source = self._app.sources.edit(source_index, source_name, source_channel)
        self.sourcesChanged.emit(self._app.sources)
        return modified_source

    @require_project_loaded
    def delete_source(self, source_index: int):
        """Delete an existing source by index

        Arguments
        ---------
        source_index : int
        """
        self._app.sources.delete(source_index)
        self.sourcesChanged.emit(self._app.sources)

    @require_project_loaded
    def get_sources(self) -> List[Source]:
        return list(self._app.sources)

    @require_project_loaded
    def read_config(self):
        return self._app.read_config()

    @require_project_loaded
    def convert_project_index_to_stft_index(self, idx: ProjectIndex):
        return StftIndex(self._app.project, self._app.stft.config.step, idx // self._app.stft.config.step)

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

        max_size = self._app.stft.n_cache_total
        if self._app.workspace.size + n >= max_size:
            n = max_size - self._app.workspace.size

        self._app.workspace.scale(n)
        self._app.stft.set_active_size(self._app.workspace.size)
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

    def check_if_sources_need_saving(self) -> bool:
        """Returns True if Sources have unsaved changes
        """
        if any([self._app.sources.needs_saving()] + [p.needs_saving() for p in self._app.plugins.values()]):
            return True
        else:
            return False

    def save(self):
        try:
            self._app.save_sources()
            for name, plugin in self._app.plugins.items():
                plugin.save()
        except Exception as e:
            logger.exception("Exception while saving")
            return False
        else:
            return True


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
