"""Thin wrapper around the main app controller
"""
import functools
import logging
from pathlib import Path
from typing import Tuple

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import numpy as np

from soundsep.app.app import SoundsepController
from soundsep.app.services import SourceService
from soundsep.core.models import StftIndex, Source


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

    def __init__(self, app: SoundsepController):
        super().__init__()
        self._app = app

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

    # def read_config(self):
    #     return self._app.services.Config.read()

    @require_project_loaded
    def workspace_move_to(self, start: StftIndex):
        """Placeholder

        Arguments
        ---------
        start : StftIndex
        """
        prev_position = self._app.workspace.start
        self._app.workspace.move_to(start)
        self._app.stft.set_position(start)
        if prev_position != start:
            self.workspaceChanged.emit(*self._app.workspace.get_lim(StftIndex))

    @require_project_loaded
    def workspace_move_by(self, dx: int):
        """Placeholder

        Arguments
        ---------
        dx : int
        """
        self._app.workspace.move_by(dx)
        self._app.stft.set_position(self._app.workspace.start)
        if dx != 0:
            self.workspaceChanged.emit(*self._app.workspace.get_lim(StftIndex))

    @require_project_loaded
    def workspace_scale(self, n: int):
        """Placeholder

        Arguments
        ---------
        n : int
        """
        self._app.workspace.scale(n)
        self._app.stft.set_position(self._app.workspace.start)
        if n != 0:
            self.workspaceChanged(*self._app.workspace.get_lim(StftIndex))

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
        # self._app.stft.set_position(self._app.workspace.start)
        data, stale = self._app.stft.read(*xlim)
        return data, stale, self._app.stft.get_freqs()

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
        if old_position[0] != start or old_position[1] != stop:
            self.workspaceChanged.emit(*self._app.workspace.get_lim(StftIndex))


class SoundsepGuiApi(QObject):
    """Soundsep GUI API exposed to plugins
    """
