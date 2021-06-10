import os
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from soundsep.core.models import Source
from soundsep.core.io import load_project


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
            "recovery_dir": recovery_dir or appdata_dir / "save",
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
            if not os.path.commonprefix([self.base_dir, v]) == self.base_dir:
                raise ValueError("All project folders must be a subdirectory of the base project")
            v.mkdir(parents=True, exist_ok=True)


# Should I call this a Workspace?
# Alternative names?
class Workspace:
    """

    TODO: make all paths configurable instead of arguments

    Attributes
    ----------
    paths : FileLocations
        Holds pointers to important folders and files relevant
        to the current workspace
    project : Optional[Project]
        Project instance for reading data. Is None when a project
        has not been loaded.
    sources : Optional[List[Source]]
        A list of active Source instances in the workspace. Is
        None if sources have not been loaded.
    datastore : dict
        Data storage object. Keys are namespaces so that each
        plugin can write to its own space. The "soundsep"
        namespace exists by default.

    Arguments
    ---------
    base_dir : Path
        Starting directory
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

        self.paths = FileLocations(
            base_dir,
            config_file,
            audio_dir,
            export_dir,
            plugin_dir,
            save_dir,
            recovery_dir,
            log_dir,
        )

        if not self.paths.config_file.exists():
            self._write_default_project_config()

        # TODO: wait why am i having this agian?
        # Why not just force it to load on initialization?
        self.project = None
        self.sources = None
        self.datastore = {"soundsep": {}}

        self.load_project()
 
    def ready(self) -> bool:
        """Return True if a project has been loaded"""
        return self.project is not None

    def _write_default_project_config(self):
        pass

    def read_config(self) -> dict:
        """Read the configuration file into a dictionary"""
        with open(self.paths.config_file, "r") as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    def load_project(self):
        """Load the project with config values"""
        config = self.read_config()
        self.project = load_project(
            self.paths.audio_dir,
            config.get("filename_pattern", None),
            config.get("block_keys", None),
            config.get("channel_keys", None),
        )
        self.sources = []

    def load_sources(self, save_file: Path):
        """Read sources from a save file"""
        if not self.ready():
            raise RuntimeError("Cannot attempt to load sources before project is loaded")

        self.sources = []
        data = pd.read_csv(sources_file)
        for i in range(len(data)):
            row = data.iloc[i]
            self.sources.append(Source(
                self.project,
                str(row["SourceName"]),
                int(row["SourceChannel"])
            ))

    def save_sources(self, save_file: Path):
        """Save sources to a csv file"""
        if not self.ready():
            raise RuntimeError("You really shouldn't be trying to save when nothing is loaded.")

        data = pd.DataFrame([{"SourceName": s.name, "SourceChannel": s.channel} for s in self.sources])
        data.to_csv(data)

