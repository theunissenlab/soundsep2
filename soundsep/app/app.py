import functools
import glob
import importlib
import logging
import os
import pkgutil
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from soundsep.app.services import (
    AmpenvService,
    SelectionService,
    SourceService,
    StftCache,
    StftConfig
)
from soundsep.core.base_plugin import BasePlugin
from soundsep.core.models import Project, ProjectIndex, StftIndex, Source
from soundsep.core.io import load_project


logger = logging.getLogger(__name__)


class FileLocations(dict):
    """Path lookup with default values

    Default directory structure::

      project_name/
        project.yaml
        audio/
        export/
        plugins/
        _appdata/
          save/
          recovery/
          logs/
    """
    def __init__(
            self,
            base_dir: Path,
            config_file: Optional[Path] = None,
            audio_dir: Optional[Path] = None,
            export_dir: Optional[Path] = None,
            plugin_dir: Optional[Path] = None,
            save_dir: Optional[Path] = None,
            recovery_dir: Optional[Path] = None,
            log_dir: Optional[Path] = None,
        ):
        if not base_dir.is_dir():
            raise ValueError("Base project directory must exist")

        appdata_dir = base_dir / "_appdata"

        self["base_dir"] = base_dir
        self["_subdirectories"] = {
            "appdata_dir": appdata_dir,
            "audio_dir": audio_dir or base_dir / "audio",
            "export_dir": export_dir or base_dir / "export",
            "plugin_dir": plugin_dir or base_dir / "plugins",
            "save_dir": save_dir or appdata_dir / "save",
            "recovery_dir": recovery_dir or appdata_dir / "recovery_dir",
            "log_dir": log_dir or appdata_dir / "logs",
        }
        self["_files"] = {
            "config_file": config_file or base_dir / "project.yaml",
            "default_sources_savefile": self.save_dir / "sources.csv",
        }

    def __getattr__(self, attr: str):
        try:
            return self["_subdirectories"][attr]
        except (AttributeError, KeyError):
            pass

        try:
            return self["_files"][attr]
        except (AttributeError, KeyError):
            pass

        try:
            return self[attr]
        except (AttributeError, KeyError):
            pass

        return object.__getattribute__(self, attr)

    def create_folders(self):
        """Create all subdirectory folders for the given configuration if they don't exist"""
        for v in self._subdirectories.values():
            if not os.path.commonprefix([self.base_dir, v]) == str(self.base_dir):
                raise ValueError("All project folders must be a subdirectory of the base project")
            v.mkdir(parents=True, exist_ok=True)


# TODO move to core
class Workspace(QObject):
    """Representation of the current working time range in StftIndex units

    The workspace is represented by a start index (inclusive) and end index (exclusive)
    """
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

    def set_position(self, start: StftIndex, stop: StftIndex, preserve_requested_size: bool=False):
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


def require_project_loaded(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self.has_project_loaded():
            warnings.warn("Cannot call method {} on SoundsepController without project loaded".format(fn.__name__))
            return
        return fn(self, *args, **kwargs)
    return wrapper


class SoundsepController(QObject):
    """Soundsep application logic
    """
    def __init__(self):
        super().__init__()
        self.project = None
        self.paths = None
        self.workspace = None
        # self.services = None

        # TODO make these properties with require_project_loaded decorator?
        self.stft = None
        self.ampenv = None
        self.sources = None
        self.selection = None

        self.datastore = None

        self._plugins = {}

    @property
    def plugins(self):
        return self._plugins

    def has_project_loaded(self):
        return self.project is not None

    def _write_default_project_config(self):
        from soundsep.core.default_config import default_config
        with open(self.paths.config_file, "w+") as f:
            return f.write(yaml.dump(default_config))

    def initialize_project(self, base_dir: Path):
        self.paths = FileLocations(base_dir)
        if not self.paths.config_file.exists():
            self._write_default_project_config()
        config = self.read_config(self.paths.config_file)
        self.paths.create_folders()

    def load_project(self, base_dir: Path):
        self.paths = FileLocations(base_dir)
        if not self.paths.config_file.exists():
            self._write_default_project_config()

        config = self.read_config(self.paths.config_file)

        self.project = load_project(
            self.paths.audio_dir,
            config.get("filename_pattern", None),
            config.get("block_keys", None),
            config.get("channel_keys", None),
        )

        step = config.get("stft.step", 44)
        self.workspace = Workspace(
            StftIndex(self.project, step, 0),
            StftIndex(self.project, step, config.get("workspace.default_size", 2000)),
        )

        # TODO: we should consider delaying the instantiation of services
        # because it might be useful to have a project instantiated and then have files
        # imported later...

        # Initialize Services
        self.ampenv = AmpenvService(self.project)
        self.sources = self.load_sources()
        # TODO: not sure if here is the place, but we can adjust stft.window so it
        # aligns well with a scipy.fft.next_fast_len for ~30% speedup in some cases
        self.stft = StftCache(
            self.project,
            self.workspace.size,
            pad=config.get("stft.cache.size", 8 * self.workspace.size),  # TODO whats a good default?
            stft_config=StftConfig(window=config.get("stft.window", 302), step=step)
        )
        self.selection = SelectionService(self.project)

        self.datastore = {}
        # TODO: set a timer job to Watch config file for changes?

    def clear_project(self):
        self.project = None
        self.paths = None
        self.workspace = None

        if self.stft is not None:
            self.stft._worker.cancel()

        self.stft = None
        self.ampenv = None
        self.sources = None
        self.selection = None
        self.datastore = None

        self._plugins = {}

    def read_config(self, path=None) -> dict:
        """Read the configuration file into a dictionary

        If path is not provided, attemps to read from the self.path object.
        read_config() can only be called without a path argument if a project
        is loaded
        """
        if path is None:
            path = self.paths.config_file

        with open(path, "r") as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    @require_project_loaded
    def load_sources(self) -> SourceService:
        """Read sources from a save file"""
        sources = SourceService(self.project)
        if self.paths.default_sources_savefile.exists():
            data = pd.read_csv(self.paths.default_sources_savefile)
            for i in range(len(data)):
                row = data.iloc[i]
                sources.append(Source(
                    self.project,
                    str(row["SourceName"]),
                    int(row["SourceChannel"]),
                    int(row["SourceIndex"]),
                ))
        return sources

    @require_project_loaded
    def save_sources(self):
        """Save sources to a csv file"""
        self.paths.create_folders()
        data = pd.DataFrame([{"SourceName": s.name, "SourceChannel": s.channel, "SourceIndex": s.index} for s in self.sources])
        data.to_csv(self.paths.default_sources_savefile)

    def reload_plugins(self, api, gui):
        local_plugin_modules = self.load_local_plugins()
        app_plugin_modules = self.load_app_plugins()
        all_active_plugin_modules = local_plugin_modules + app_plugin_modules

        # Setup plugins
        self._plugins = {
            Plugin.__name__: Plugin(api, gui)
            for Plugin in BasePlugin.registry
            if any([Plugin.__module__.startswith(m) for m in all_active_plugin_modules])
        }

        logger.info("Loaded Plugins {}".format(self._plugins))

        for name, plugin in self.plugins.items():
            plugin.setup_plugin_shortcuts()
            for w in plugin.plugin_toolbar_items():
                gui.ui.toolbarLayout.addWidget(w)

            panel = plugin.plugin_panel_widget()
            if panel:
                gui.ui.pluginPanelToolbox.addTab(
                    panel,
                    name
                )

            menu = plugin.add_plugin_menu(gui.ui.menuPlugins)

    @require_project_loaded
    def load_local_plugins(self) -> List[str]:
        logger.debug("Searching {} for plugin modules".format(
            self.paths.plugin_dir))

        local_modules = glob.glob(os.path.join(self.paths.plugin_dir, "*"))

        plugin_names = []
        for plugin_file in local_modules:
            name = os.path.splitext(os.path.basename(plugin_file))[0]
            mod_name = "local_plugins.{}".format(name)

            if name.startswith(".") or name.startswith("_"):
                continue

            spec = importlib.util.spec_from_file_location(mod_name, plugin_file)

            if spec is None:
                continue

            try:
                plugin_module = importlib.util.module_from_spec(spec)
            except AttributeError:
                continue
            else:
                spec.loader.exec_module(plugin_module)
            plugin_names.append(mod_name)

        return plugin_names

    def load_app_plugins(self) -> List[str]:
        """Load plugins from three possible locations

        (1) soundsep.plugins, and (2) self.paths.plugin_dir
        """
        import soundsep.plugins

        def iter_namespace(ns_pkg):
            return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

        # Search for builtin plugins in soundsep/plugins
        plugin_names = []
        for finder, name, ispkg in iter_namespace(soundsep.plugins):
            importlib.import_module(name)
            plugin_names.append(name)

        return plugin_names
