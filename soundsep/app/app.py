import importlib
import logging
import pickle
import pkgutil
from pathlib import Path
from typing import List

import pandas as pd
import yaml
from PyQt6.QtCore import QObject, QSettings, pyqtSignal

from soundsep.api import Api
from soundsep.app.exceptions import BadConfigFormat, ConfigDoesNotExist
from soundsep.app.services import (
    AmpenvService,
    SelectionService,
    SourceService,
    Workspace,
)
from soundsep.app.stft_service import StftService
from soundsep.core.stft.cache import StftParameters
from soundsep.config.defaults import DEFAULTS
from soundsep.config.paths import ProjectPathFinder
from soundsep.core.base_plugin import BasePlugin
from soundsep.core.models import StftIndex, Source, NWBFile
from soundsep.core.io import load_project
from soundsep.settings import (
    QSETTINGS_APP,
    QSETTINGS_ORG,
    SETTINGS_VARIABLES,
)


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

        # NWB mode flags - set by from_nwb_file()
        self._nwb_mode = False
        self._nwb_path = None

        if not self.paths.config.exists():
            raise ConfigDoesNotExist

        try:
            self.config = SoundsepApp.read_config(self.paths.config)
        except:
            raise BadConfigFormat

        self.project = load_project(
            Path(self.config["audio_directory"]) if self.config["audio_directory"] else self.paths.audio_dir,
            self.config["filename_pattern"],
            self.config["block_keys"],
            self.config["channel_keys"],
            recursive=self.config["recursive_search"]
        )

        self.plugins = {}
        self.state = {}
        self.services = {}
        self.datastore = {}

        self.qsettings = QSettings(
            QSETTINGS_ORG,
            QSETTINGS_APP,
        )

        self.qsettings.setValue(SETTINGS_VARIABLES["REOPEN_PROJECT_PATH"], str(project_dir))

    @classmethod
    def from_nwb_file(cls, nwb_path: 'pathlib.Path') -> 'SoundsepApp':
        """Create a SoundsepApp directly from an NWB file without a config.

        This allows opening NWB files directly without creating a project first.
        App data (sources, etc.) is stored directly in the NWB file's scratch group.

        Arguments
        ---------
        nwb_path : pathlib.Path
            Path to the NWB file

        Returns
        -------
        SoundsepApp
            A configured SoundsepApp instance
        """
        nwb_path = Path(nwb_path)
        if not nwb_path.exists():
            raise FileNotFoundError(f"NWB file not found: {nwb_path}")
        if nwb_path.suffix.lower() != ".nwb":
            raise ValueError(f"Expected .nwb file, got: {nwb_path}")

        # Use the NWB file's parent directory for exports and other path-based operations
        project_dir = nwb_path.parent

        # Create the app instance using __new__ and manual initialization
        # We can't use __init__ because it requires a config file
        app = cls.__new__(cls)
        QObject.__init__(app)

        app.project_dir = project_dir
        app.api = Api(app)
        app.paths = ProjectPathFinder(project_dir)

        # NWB mode - store data in the NWB file itself
        app._nwb_mode = True
        app._nwb_path = nwb_path

        # Create config from defaults, pointing to the NWB file
        app.config = {
            **DEFAULTS,
            "audio_directory": str(nwb_path.parent),
            "filename_pattern": nwb_path.name,
            "block_keys": None,
            "channel_keys": None,
            "recursive_search": False,
        }

        # Load project directly from the NWB file
        app.project = load_project(
            nwb_path,  # Pass the file directly
            filename_pattern=None,
            block_keys=None,
            channel_keys=None,
            recursive=False
        )

        app.plugins = {}
        app.state = {}
        app.services = {}
        app.datastore = {}

        app.qsettings = QSettings(
            QSETTINGS_ORG,
            QSETTINGS_APP,
        )

        # Store the NWB file path for reopening
        app.qsettings.setValue(SETTINGS_VARIABLES["REOPEN_PROJECT_PATH"], str(nwb_path))

        # Create folders for plugins that still need filesystem storage
        # (e.g., segments plugin saves to CSV)
        app.paths.create_folders()

        return app

    @staticmethod
    def read_config(path: 'pathlib.Path') -> dict:
        """Read the configuration file into a dictionary

        If path is not provided, attemps to read from the self.path object.
        read_config() can only be called without a path argument if a project
        is loaded
        """
        with open(path, "r") as f:
            local_config = yaml.load(f, Loader=yaml.SafeLoader)

        if local_config["audio_directory"]:
            audio_dir = Path(local_config["audio_directory"])
            if not audio_dir.is_absolute():
                local_config["audio_directory"] = str(path.parent / audio_dir)

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
        # self.services["stft"] = StftCache(
        #     self.project,
        #     self.state["workspace"].size,
        #     pad=self.config["stft.cache.size"],
        #     stft_config=StftConfig(window=self.config["stft.window"], step=step)
        # )
        self.services["stft"] = StftService(
            self.project,
            n_scales=self.config["stft.cache.n_scales"],
            cache_size=self.config["stft.cache.size"],
            fraction_cached=self.config["stft.cache.fraction_cached"],
            stft_params=StftParameters(hop=step, half_window=self.config["stft.window"]),
        )
        self.services["stft"].set_central_range(self.state["workspace"].start, self.state["workspace"].stop)

        self.datastore["sources"] = self.load_sources()

    def close(self):
        self.project.close_files()
        self.services["stft"].close()

    def load_sources(self) -> SourceService:
        """Read sources from a save file or NWB scratch group"""
        sources = SourceService(self.project)

        if self._nwb_mode and self._nwb_path:
            # Load from NWB file scratch group
            try:
                source_data = NWBFile.read_soundsep_sources(str(self._nwb_path))
                for row in source_data:
                    sources.append(Source(
                        self.project,
                        str(row["SourceName"]),
                        int(row["SourceChannel"]),
                        int(row["SourceIndex"]),
                    ))
            except Exception as e:
                logger.warning(f"Could not load sources from NWB file: {e}")
        elif self.paths.sources_file.exists():
            # Load from CSV file
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
        """Save sources to a csv file or NWB scratch group"""
        source_data = [
            {"SourceName": s.name, "SourceChannel": s.channel, "SourceIndex": s.index}
            for s in self.datastore["sources"]
        ]

        if self._nwb_mode and self._nwb_path:
            # Save to NWB file scratch group
            # Must close the file first since it's open for reading audio data
            # Files will be reopened lazily when audio is needed again
            self.project.close_files()
            try:
                NWBFile.write_soundsep_sources(str(self._nwb_path), source_data)
                logger.info(f"Saved {len(source_data)} sources to NWB file")
            except Exception as e:
                logger.error(f"Could not save sources to NWB file: {e}")
                raise
            # Also create folders for plugins that still need filesystem storage
            self.paths.create_folders()
        else:
            # Save to CSV file
            self.paths.create_folders()
            data = pd.DataFrame(source_data)
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
