"""Default path locations relative to a project directory
"""
import os


class ProjectPathFinder:
    def __init__(self, project_dir: 'pathlib.Path'):
        self.project_dir = project_dir

    @property
    def appdata_dir(self):
        return self.project_dir / "_appdata"

    @property
    def config(self):
        return self.project_dir / "soundsep.yaml"

    @property
    def audio_dir(self):
        return self.project_dir / "audio"

    @property
    def plugin_dir(self):
        return self.project_dir / "plugins"

    @property
    def save_dir(self):
        return self.appdata_dir / "save"

    @property
    def sources_file(self):
        return self.appdata_dir / "sources.csv"

    @property
    def logs_dir(self):
        return self.appdata_dir / "logs"

    @property
    def recovery_dir(self):
        return self.appdata_dir / "recovery"

    @property
    def recovery_file(self):
        return self.recovery_dir / "recovery.pkl"

    @property
    def _subdirectories(self) -> 'List[pathlib.Path]':
        return [
            self.appdata_dir,
            self.audio_dir,
            self.plugin_dir,
            self.save_dir,
            self.logs_dir
        ]

    def create_folders(self):
        """Create all subdirectory folders for the given configuration if they don't exist"""
        for v in self._subdirectories:
            if not os.path.commonprefix([self.project_dir, v]) == str(self.project_dir):
                raise ValueError("All project folders must be a subdirectory of the base project")
            v.mkdir(parents=True, exist_ok=True)
