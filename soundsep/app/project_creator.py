import collections
import glob
import os
import yaml
from pathlib import Path
from string import Formatter

import parse
from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import Qt, pyqtSignal

from soundsep.config.defaults import DEFAULTS
from soundsep.core.io import group_files_by_pattern, search_for_wavs
from soundsep.core.models import AudioFile, Block
from soundsep.ui.project_creator import Ui_ProjectCreator


class ProjectCreator(widgets.QWidget):

    createConfigCanceled = pyqtSignal()
    openProject = pyqtSignal(Path)

    def __init__(self):
        super().__init__()

        self.init_ui()
        self.connect_events()

    def init_ui(self):
        self.ui = Ui_ProjectCreator()
        self.ui.setupUi(self)
        self.ui.errorTable.horizontalHeader().setSectionResizeMode(widgets.QHeaderView.Stretch)
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
        options = widgets.QFileDialog.Options()
        save_target, _ = widgets.QFileDialog.getSaveFileName(
            None,
            "Saving yaml file",
            os.path.dirname(config["audio_directory"]),
            "soundsep.yaml",
            options=options)

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
        options = widgets.QFileDialog.Options()
        path = widgets.QFileDialog.getExistingDirectory(
            self,
            "Select audio folder containing WAV files",
            ".",
            options=options
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
        recursive = self.ui.recursiveSearchCheckBox.checkState() == Qt.CheckState.Checked
        
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
            filelist = []
            for f in search_for_wavs(base_path, recursive=recursive):
                filelist.append(str(f))

            # filter out any wav files that didn't work
            checked_filelist = []
            for f in filelist:
                try:
                    AudioFile(f)
                except Exception as e:
                    errors.append((str(f), str(e)))
                else:
                    checked_filelist.append(f)

            block_groups, grouping_errors = group_files_by_pattern(
                base_path,
                checked_filelist,
                filename_pattern=template_string,
                block_keys=new_keys["block_keys"],
                channel_keys=new_keys["channel_keys"],
            )
            errors += grouping_errors

            blocks = []
            channel_id_sets = collections.defaultdict(list)
            for key, group in block_groups:
                group = list(group)
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
                # for k, v in [str(([os.path.basename(f.path) for f in v[0]._files], k))
                for k, v in channel_id_sets.items():
                    errors.append((
                        ",".join([str(f.path) for f in v[0]._files]),
                        "Channel id's inconsistent across blocks: {}".format(k)
                    ))

            self.show_errors(errors)
            self.ui.treeView.set_blocks(blocks, _parse_block, _parse_channels)

    def on_path_selected(self):
        self.update_treeview_as_audio_files()

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
            filelist = []
            for f in search_for_wavs(base_path, recursive=recursive):
                filelist.append(f)

            audio_files = []
            errors = []
            for f in filelist:
                try:
                    audio_files.append(AudioFile(f))
                except Exception as e:
                    errors.append((str(f), str(e)))

            if errors:
                self.show_errors(errors)

            if len(audio_files):
                self.ui.treeView.set_audio_files(audio_files, keys_fn)
                self.ui.step2GroupBox.setVisible(True)
                return

        self.ui.step2GroupBox.setVisible(False)
        self.ui.step3GroupBox.setVisible(False)

    def on_template_changed(self):
        base_path = self.ui.basePathEdit.text()
        template_string = self.ui.templateEdit.text()
        try:
            format_variables = [i[1] for i in Formatter().parse(template_string) if i[1]]
        except:
            return

        self.set_format_variables(format_variables)

        def _parse(path):
            """Gets format variables in """
            result = parse.parse(template_string, str(path))
            if not result:
                return []
            else:
                return [result[v] for v in format_variables if v]

        self.update_treeview_as_audio_files(_parse)

    def on_template_completed(self):
        if not self.ui.templateEdit.text():
            self.ui.templateEdit.setText(self.ui.templateEdit.placeholderText())
        self.ui.step3GroupBox.setVisible(True)
        self.ui.submitButtons.setVisible(True)
