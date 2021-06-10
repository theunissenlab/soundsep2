from enum import Enum
from pathlib import Path
from typing import List, Tuple

import PyQt5.QtWidgets as widgets
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from soundsep.core.app import Workspace
from soundsep.core.models import Source, ProjectIndex


class SourceChange(Enum):
    APPEND = 1
    EDIT = 2
    DELETE = 3
    RESET = 4


class Soundsep(QObject):
    """State, Controller, and API for a Soundsep Project

    Changes typically propogate from user -> gui -> this object -> workspace/project,
    so usually we don't need to propogate events; the GUI elements
    that made the API calls should already know what they
    have done and can update. However, maybe its fine?
    """
    sourcesChanged = pyqtSignal(SourceChange, int)
    selectionChanged = pyqtSignal()
    xrangeChanged = pyqtSignal(ProjectIndex, ProjectIndex)

    def __init__(self, workspace: Workspace):
        super().__init__()

        self._ws = workspace
        self._active_selection = None

        config = self._ws.read_config()
        self._visible_range = (
            ProjectIndex(self._ws.project, 0),
            ProjectIndex(
                self._ws.project, 
                int(self._ws.project.sampling_rate * config["duration"])
            )
        )

        # self.view_state = None
        # self.selection_state = None

    def __getitem__(self, plugin_name):
        return self._ws.plugins[plugin_name]

    @property
    def plugins(self) -> list:
        return self._ws.plugins

    def get_datastore(self) -> dict:
        """Get the *mutable* shared data store dict"""
        return self._ws.datastore

    def read_config(self) -> dict:
        """Read the config file
            
        Returns
        -------
        config : dict
        """
        return self._ws.read_config()
    
    @property
    def paths(self):
        return self._ws.paths

    @property
    def project(self):
        return self._ws.project

    def set_xrange(self, start: ProjectIndex, stop: ProjectIndex):
        """Set the active view range in ProjectIndex coordinates

        Emits Soundsep.xrangeChanged

        Arguments
        ---------
        start : ProjectIndex
        stop : ProjectIndex
        """
        self._visible_range = (start, stop)
        self.xrangeChanged.emit(start, stop)

    def get_xrange(self):
        """Get the active view range in ProjectIndex coordinates

        Returns
        -------
        start : ProjectIndex
        stop : ProjectIndex
        """
        return self._visible_range

    def get_active_selection(self):
        """Get the active selection it it exists

        Returns
        -------
        selection : None or ((ProjectIndex, ProjectIndex), (float, float), Source)
            Returns None if there is no active selection, or a tuple of 
            the active xbounds (a tuple in ProjectIndex coordinates)
            and the active ybounds (a tuple in floating point Hz)
        """
        return self._active_selection

    def create_source(self, name, channel):
        """Creates a new source 

        Emits Soundsep.sourcesChanged(SourceChange.CREATE, i)
        where i is the index of the newly created source

        Arguments
        ---------
        name : str
        channel : int
        """
        self._ws.sources.append(Source(self._ws.project, name, channel))
        self.sourcesChanged.emit(
            SourceChange.APPEND,
            len(self._ws.sources) - 1
        )

    def edit_source(self, i: int, name: str, channel: int):
        """Edit the Source at index i

        Emits Soundsep.sourcesChanged(SourceChange.EDIT, i)

        Arguments
        ---------
        i : int
            Index of source to edit
        name : str
        channel : int
        """
        source = self._ws.sources[i]
        changed = False

        if source.name != name:
            source.name = name
            changed = True

        if source.channel != channel:
            source.channel = channel
            changed = True

        self.sourcesChanged.emit(SourceChange.EDIT, i)

    def delete_source(self, i: int):
        """Delete the source at index i

        Emits Soundsep.sourcesChanged(SourceChange.DELETE, i)

        Arguments
        ---------
        i : int
            Index of source to delete
        """
        del self._ws.sources[i]
        self.sourcesChanged.emit(SourceChange.DELETE, i)

    def get_sources(self) -> List[Source]:
        """Get all the current sources
        """
        return self._ws.sources[:]  # readonly!

    def set_selection(
            self,
            xbounds: Tuple[ProjectIndex, ProjectIndex],
            ybounds: Tuple[float, float],
            source: Source
        ):
        """Set the given time/freq bounds as the currently selected region

        Emits Soundsep.selectionChanged

        Arguments
        ---------
        xbounds : Tuple[ProjectIndex, ProjectIndex]
            A tuple of the start and end-points of the selection,
            given in Project coordinates
        ybounds : Tuple[float, float]
            A tuple of the low and high frequencies of the selection,
            given in Hz
        """
        self._active_selection = (xbounds, ybounds, source)
        self.selectionChanged.emit()

    def clear_selection(self):
        """Clear the currently seleted region

        Emits Soundsep.selectionChanged
        """
        self._active_selection = None
        self.selectionChanged.emit()

    def load_sources(self, save_file: Path):
        """Load previously saved sources from csv

        csv must have columns SourceName and SourceChannel.
        Default location to look Soundsep.ws.paths.default_sources_savefile

        Emits Soundsep.sourcesChanged(SourceChange.RESET, 0)

        Arguments
        ---------
        save_file: 
        """
        self._ws.load_sources(save_file)
        self.sourcesChanged(SourceChange.RESET, 0)

    def save_sources(self, save_file: Path):
        """Save sources to a csv file

        Default location to save is Soundsep.ws.paths.default_sources_savefile
        """
        self._ws.load_sources(save_file)
