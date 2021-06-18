import importlib
import logging
import pickle
import pkgutil
from typing import List

import pandas as pd
import yaml
from PyQt5.QtCore import QObject, pyqtSignal

from soundsep.api import Api
from soundsep.app.services import (
    AmpenvService,
    SelectionService,
    SourceService,
    StftCache,
    StftConfig,
    Workspace,
)
from soundsep.config.defaults import DEFAULTS
from soundsep.config.paths import ProjectPathFinder
from soundsep.core.base_plugin import BasePlugin
from soundsep.core.models import StftIndex, Source
from soundsep.core.io import load_project


logger = logging.getLogger(__name__)


class SoundsepApp(QObject):
    """Soundsep application logic

    Config and project file loaders are not created on instantiation so that
    debugging and analysis of a project folder's contents can be done before
    the data is attempted to be read.

    Example
    -------
    >>> app = SoundsepApp("data/project1")
    >>> app.validate_project_folder()
    >>> app.read_config()
    >>> app.instantiate_plugins(gui)
    >>> app.load_project()
    """

    configChanged = pyqtSignal(object)  # Not implemented yet

    def __init__(self, project_dir: 'pathlib.Path'):
        super().__init__()
        self.project_dir = project_dir
        self.api = Api(self)
        self.paths = ProjectPathFinder(project_dir)
        self.config = SoundsepApp.read_config(self.paths.config)
        self.project = load_project(
            self.paths.audio_dir,
            self.config["filename_pattern"],
            self.config["block_keys"],
            self.config["channel_keys"],
        )

        self.plugins = {}
        self.state = {}
        self.services = {}
        self.datastore = {}

    @staticmethod
    def read_config(path: 'pathlib.Path') -> dict:
        """Read the configuration file into a dictionary

        If path is not provided, attemps to read from the self.path object.
        read_config() can only be called without a path argument if a project
        is loaded
        """
        with open(path, "r") as f:
            local_config = yaml.load(f, Loader=yaml.SafeLoader)
        return {**DEFAULTS, **local_config}

    def instantiate_plugins(self, gui: 'soundsep.app.main_window.SoundsepMainWindow'):
        local_plugin_modules = self.load_local_plugins()
        app_plugin_modules = self.load_app_plugins()
        all_active_plugin_modules = local_plugin_modules + app_plugin_modules

        # Setup plugins
        self.plugins = {
            Plugin.__name__: Plugin(self.api, gui)
            for Plugin in BasePlugin.registry
            if any([Plugin.__module__.startswith(m) for m in all_active_plugin_modules])
        }

        for plugin in self.plugins.values():
            gui.attach_plugin(plugin)

    def setup(self):
        step = self.config["stft.step"]
        self.state["workspace"] = Workspace(
            StftIndex(self.project, step, 0),
            StftIndex(self.project, step, self.config["workspace.default_size"]),
        )
        self.state["selection"] = SelectionService(self.project)

        self.services["ampenv"] = AmpenvService(self.project)
        self.services["stft"] = StftCache(
            self.project,
            self.state["workspace"].size,
            pad=self.config["stft.cache.size"],
            stft_config=StftConfig(window=self.config["stft.window"], step=step)
        )
        self.datastore["sources"] = self.load_sources()

    def close(self):
        self.services["stft"].close()

    def load_sources(self) -> SourceService:
        """Read sources from a save file"""
        sources = SourceService(self.project)
        if self.paths.sources_file.exists():
            data = pd.read_csv(self.paths.sources_file)
            for i in range(len(data)):
                row = data.iloc[i]
                sources.append(Source(
                    self.project,
                    str(row["SourceName"]),
                    int(row["SourceChannel"]),
                    int(row["SourceIndex"]),
                ))
        return sources

    def save_sources(self):
        """Save sources to a csv file"""
        self.paths.create_folders()
        data = pd.DataFrame([
            {"SourceName": s.name, "SourceChannel": s.channel, "SourceIndex": s.index}
            for s in self.datastore["sources"]
        ])
        data.to_csv(self.paths.sources_file)
        self.datastore["sources"].set_needs_saving(False)

    def panic_save(self, e: Exception):
        """Dumps the datastore, app state, etc to a pickle file"""
        self.paths.recovery_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "services": self.services,
            "state": self.state,
            "datastore": self.datastore,
            "exception": e,
        }
        with open(self.paths.recovery_file, "w+") as f:
            pickle.dump(payload, f)

    def load_local_plugins(self) -> List[str]:
        logger.debug("Searching {} for plugin modules".format(
            self.paths.plugin_dir))
        local_modules = self.paths.plugin_dir.glob("*")

        plugin_names = []
        for plugin_file in local_modules:
            name = plugin_file.stem
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
