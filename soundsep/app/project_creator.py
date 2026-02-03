import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import os
import yaml
from pathlib import Path
from string import Formatter

import parse
from PyQt6 import QtWidgets as widgets
from PyQt6.QtCore import Qt, pyqtSignal

from soundsep.config.defaults import DEFAULTS
from soundsep.core.io import group_files_by_pattern, guess_filename_pattern, search_for_audio_files, load_file
from soundsep.core.models import AudioFile, NWBFile, Block
from soundsep.ui.project_creator import Ui_ProjectCreator


from tqdm import tqdm

class ProjectCreator(widgets.QWidget):

    createConfigCanceled = pyqtSignal()
    openProject = pyqtSignal(Path)

    def __init__(self):
        super().__init__()

        # Cache for loaded audio files: path -> file object
        self._file_cache = {}
        self._cached_base_path = None
        self._cached_recursive = None

        self.init_ui()
        self.connect_events()

    def init_ui(self):
        self.ui = Ui_ProjectCreator()
        self.ui.setupUi(self)
        self.ui.errorTable.horizontalHeader().setSectionResizeMode(widgets.QHeaderView.ResizeMode.Stretch)
        self.ui.step2GroupBox.setVisible(False)
        self.ui.step3GroupBox.setVisible(False)
        self.ui.submitButtons.setVisible(False)
        self.ui.treeView.show_columns(["Name", "Id", "Ch", "Dur"])

    def connect_events(self):
        self.ui.basePathEdit.clicked.connect(self.on_choose_audio_folder)
        self.ui.browseButton.clicked.connect(self.on_choose_audio_folder)
        self.ui.basePathEdit.textChanged.connect(self.on_path_selected)
        self.ui.recursiveSearchCheckBox.clicked.connect(self.on_path_selected)
        self.ui.templateEdit.textChanged.connect(self.on_template_changed)
        self.ui.step2Next.clicked.connect(self.on_template_completed)
        self.ui.keySelector.keysChanged.connect(self.on_keys_changed)
        self.ui.templateEdit.returnPressed.connect(self.ui.step2Next.click)
        self.ui.closeButton.clicked.connect(self.on_cancel)
        self.ui.createConfigButton.clicked.connect(self.on_create_config)

    def on_cancel(self):
        self.createConfigCanceled.emit()
        self.close()

    def on_create_config(self):
        config = {}

        base_path = self.ui.basePathEdit.text()
        template_string = self.ui.templateEdit.text()
        recursive = self.ui.recursiveSearchCheckBox.checkState() == Qt.CheckState.Checked

        keys = self.ui.keySelector.get_keys()

        config["audio_directory"] = base_path
        config["filename_pattern"] = template_string
        if keys["block_keys"]:
            config["block_keys"] = keys["block_keys"]
        if keys["channel_keys"]:
            config["channel_keys"] = keys["channel_keys"]
        config["recursive_search"] = recursive

        self.save_config({**DEFAULTS, **config})

    def save_config(self, config: dict):
        self.hide()
        #options = widgets.QFileDialog.options()
        save_target, _ = widgets.QFileDialog.getSaveFileName(
            None,
            "Saving yaml file",
            os.path.join(os.path.dirname(config["audio_directory"]), "soundsep.yaml"),
            "*.yaml",
            #options=options)
        )

        if not save_target:
            self.show()
            return

        if os.path.exists(save_target):
            confirmed = widgets.QMessageBox.question(
                self,
                "Confirm create config",
                "{} already exists. Are you sure you want to overwrite it?".format(save_target),
                widgets.QMessageBox.Yes | widgets.QMessageBox.No
            )
            if confirmed == widgets.QMessageBox.No:
                self.show()
                return

        with open(save_target, "w") as f:
            yaml.dump(config, f)

        self.openProject.emit(Path(os.path.dirname(save_target)))
        self.close()

    def on_choose_audio_folder(self):
        # TODO: figure out what QFileDialog.options() does
        #options = widgets.QFileDialog.options()
        path = widgets.QFileDialog.getExistingDirectory(
            self,
            "Select audio folder containing WAV or NWB files",
            "."
            #options=options
        )

        if path:
            self.ui.treeView.set_base_dir(path)
            self.ui.basePathEdit.setText(path)

    def set_format_variables(self, format_variables):
        self.ui.keySelector.set_variables(format_variables)

    def on_keys_changed(self, new_keys):
        if not self.ui.step3GroupBox.isVisible():
            return

        base_path = self.ui.basePathEdit.text()
        template_string = self.ui.templateEdit.text()

        def _parse_block(path):
            """Gets format variables in """
            result = parse.parse(template_string, str(path))
            if not result:
                return []
            else:
                return [result[v] for v in new_keys["block_keys"]]

        def _parse_channels(path):
            """Gets format variables in """
            result = parse.parse(template_string, str(path))
            if not result:
                return []
            else:
                return [result[v] for v in new_keys["channel_keys"]]

        if not new_keys["block_keys"]:
            self.update_treeview_as_audio_files(_parse_channels or None)

        else:
            self.ui.treeView.clear()

            errors = []
            base_path = Path(base_path)

            # Use cached files instead of reloading
            checked_filelist = [Path(p) for p in self._file_cache.keys()]

            block_groups, grouping_errors = group_files_by_pattern(
                base_path,
                checked_filelist,
                filename_pattern=template_string,
                block_keys=new_keys["block_keys"],
                channel_keys=new_keys["channel_keys"],
                load_files=False,  # Don't load files, we'll use cache
            )
            errors += grouping_errors

            blocks = []
            channel_id_sets = collections.defaultdict(list)
            for key, group in block_groups:
                group = list(group)
                # Get cached file objects
                for g in group:
                    g["wav_file"] = self._file_cache[str(g["path"])]

                try:
                    new_block = Block([g["wav_file"] for g in group], fix_uneven_frame_counts=False)
                except ValueError as e:
                    for g in group:
                        errors.append((g["wav_file"]._path, str(e)))
                    new_block = Block([g["wav_file"] for g in group], fix_uneven_frame_counts=True)

                channel_id_sets[tuple([g["channel_id"] for g in group])].append(new_block)
                blocks.append(new_block)

            # Validate that each block shares the same channel_ids across files
            if new_keys["channel_keys"] and len(channel_id_sets) != 1:
                for k, v in channel_id_sets.items():
                    errors.append((
                        ",".join([str(f.path) for f in v[0]._files]),
                        "Channel id's inconsistent across blocks: {}".format(k)
                    ))

            self.show_errors(errors)
            self.ui.treeView.set_blocks(blocks, _parse_block, _parse_channels)

    def on_path_selected(self):
        self.autofill_filename_pattern()

    def show_errors(self, errors):
        """Takes errors, a list of tuples (filename, msg)
        """
        self.ui.errorTable.setRowCount(len(errors))
        for i, (bad_file, error_msg) in enumerate(errors):
            if error_msg is None:
                error_msg = "Could not parse filename using the given template"
            fileitem = widgets.QTableWidgetItem(str(bad_file))
            fileitem.setToolTip(str(bad_file))
            self.ui.errorTable.setItem(i, 0, fileitem)
            erroritem = widgets.QTableWidgetItem(str(error_msg))
            erroritem.setToolTip(str(error_msg))
            self.ui.errorTable.setItem(i, 1, erroritem)

    def update_treeview_as_audio_files(self, keys_fn=None):
        self.ui.treeView.clear()

        base_path = self.ui.basePathEdit.text()
        recursive = self.ui.recursiveSearchCheckBox.checkState() == Qt.CheckState.Checked

        if base_path:
            base_path = Path(base_path)

            # Check if we need to reload (path or recursive changed)
            cache_valid = (
                self._cached_base_path == base_path and
                self._cached_recursive == recursive and
                len(self._file_cache) > 0
            )

            if not cache_valid:
                # Clear cache and reload
                self._file_cache = {}
                self._cached_base_path = base_path
                self._cached_recursive = recursive

                filelist = list(search_for_audio_files(base_path, recursive=recursive))

                errors = []
                # load files in parallel
                with ThreadPoolExecutor() as executor:
                    future_to_file = {executor.submit(load_file, f): f for f in filelist}
                    prog_bar = tqdm(total=len(filelist), desc="TV Loading audio files")
                    for future in as_completed(future_to_file):
                        f = future_to_file[future]
                        try:
                            audio_file = future.result()
                            self._file_cache[str(f)] = audio_file
                        except Exception as e:
                            errors.append((str(f), str(e)))
                        prog_bar.update(1)
                    prog_bar.close()

                if errors:
                    self.show_errors(errors)

            audio_files = list(self._file_cache.values())

            if len(audio_files):
                self.ui.treeView.set_audio_files(audio_files, keys_fn)
                self.ui.step2GroupBox.setVisible(True)
                return

        self.ui.step2GroupBox.setVisible(False)
        self.ui.step3GroupBox.setVisible(False)

    def autofill_filename_pattern(self):
        base_path = self.ui.basePathEdit.text()

        recursive = self.ui.recursiveSearchCheckBox.checkState() == Qt.CheckState.Checked
        if base_path:
            base_path = Path(base_path)
            filelist = []
            for f in search_for_audio_files(base_path, recursive=recursive):
                filelist.append(f)

        if not len(filelist):
            return

        block_keys, filename_pattern = guess_filename_pattern(base_path, filelist)

        self.ui.templateEdit.setText(filename_pattern)
        self.on_template_completed()

        format_variables = [
            i[1] for i in Formatter().parse(filename_pattern)
            if i[1]
        ]
        self.set_format_variables(format_variables)

        for key in block_keys:
            self.ui.keySelector.var_to_buttons[key]["block"].setChecked(True)
        self.ui.keySelector.check_for_changes()

    def on_template_changed(self):
        base_path = self.ui.basePathEdit.text()
        template_string = self.ui.templateEdit.text()
        try:
            format_variables = [
                i[1] for i in Formatter().parse(template_string)
                if i[1]
            ]
        except:
            return

        self.set_format_variables(format_variables)

        def _parse(path):
            """Gets format variables in """
            result = parse.parse(template_string, str(path))
            if not result:
                return []
            else:
                values = []
                for v in filter(None, format_variables):
                    try:
                        values.append(result[v])
                    except KeyError:
                        pass
                return values

        self.update_treeview_as_audio_files(_parse)

    def on_template_completed(self):
        if not self.ui.templateEdit.text():
            self.ui.templateEdit.setText(self.ui.templateEdit.placeholderText())
        self.ui.step3GroupBox.setVisible(True)
        self.ui.submitButtons.setVisible(True)
