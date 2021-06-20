import glob
import os

from PyQt5 import QtWidgets as widgets

from soundsep.ui.project_creator import Ui_Form
from soundsep.core.io import _load_project_by_blocks


class ProjectDebugView(widgets.QWidget):
    """Shows the block_keys, channel_keys, and template, and how all the files fit in

    Can be read only or editable
    """

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.connect_events()

    def init_ui(self):
        self.ui = Ui_Form()
        self.ui.setupUi(self)

    def connect_events(self):
        self.ui.audioLocationEdit.clicked.connect(self.on_choose_audio_folder)
        self.ui.audioLocationEdit.textChanged.connect(self.on_edit)
        self.ui.channelKeysEdit.textChanged.connect(self.on_edit)
        self.ui.blockKeysEdit.textChanged.connect(self.on_edit)
        self.ui.filenameEdit.editingFinished.connect(self.on_edit)

    def parse_inputs(self):
        valid = True

        audio_path = self.ui.audioLocationEdit.text()
        if not audio_path:
            valid = False

        filename_template = self.ui.filenameEdit.text() or self.ui.filenameEdit.placeholderText()

        block_keys = [k.strip() for k in self.ui.blockKeysEdit.toPlainText().split("\n") if k.strip()]
        channel_keys = [k.strip() for k in self.ui.channelKeysEdit.toPlainText().split("\n") if k.strip()]

        return valid, {
            "audio_path": audio_path,
            "filename_pattern": filename_template,
            "block_keys": block_keys or None,
            "channel_keys": channel_keys or None,
        }

    def on_choose_audio_folder(self):
        options = widgets.QFileDialog.Options()
        path = widgets.QFileDialog.getExistingDirectory(
            self,
            "Select audio folder containing WAV files",
            os.path.expanduser("~"),
            options=options
        )

        if path:
            self.ui.audioLocationEdit.setText(path)

        self.ui.filelistWidget.clear()
        if self.ui.audioLocationEdit.text():
            for f in glob.glob(os.path.join(self.ui.audioLocationEdit.text(), "*.wav")):
                self.ui.filelistWidget.addItem(str(f))

    def on_edit(self):
        """Render the potential project"""
        self.ui.errorsList.clear()
        self.ui.blocksList.clear()
        valid, inputs = self.parse_inputs()
        if not valid:
            return

        wav_files = glob.glob(os.path.join(inputs["audio_path"], "*.wav"))
        dummy_project, errors = _load_project_by_blocks(
            wav_files,
            inputs["filename_pattern"],
            inputs["block_keys"],
            inputs["channel_keys"],
            allow_errors=True
        )

        self.ui.filesLabel.setText("{} WAV files".format(len(wav_files)))
        if len(errors):
            self.ui.errorsLabel.setText("{} errors".format(len(errors)))
        else:
            self.ui.errorsLabel.setText("")

        try:
            self.ui.samplerateLabel.setText("{} Hz".format(dummy_project.sampling_rate))
            self.ui.framesLabel.setText("{} frames".format(dummy_project.frames))
            self.ui.channelsLabel.setText("{} channels".format(dummy_project.channels))
        except:
            pass

        for block in dummy_project.blocks:
            self.ui.blocksList.addItem(",".join([str(f) for f in block._files]))

        for error in errors:
            self.ui.errorsList.addItem(str(error))



