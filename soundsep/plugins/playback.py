import struct

from PyQt5.QtMultimedia import QAudio, QAudioFormat, QAudioOutput
from PyQt5.QtCore import QBuffer, QByteArray, QIODevice
from PyQt5 import QtGui
from PyQt5 import QtWidgets as widgets
import numpy as np

from soundsep.core.base_plugin import BasePlugin


class PlaybackPlugin(BasePlugin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize UI
        self.button = widgets.QPushButton("ᐅ Play")
        self.button.setCheckable(True)
        self.button.clicked.connect(self.play_audio)

        self.playback_action = widgets.QAction("Play Selection")
        self.playback_action.setCheckable(True)
        self.playback_action.triggered.connect(self.toggle_play)
        self.playback_action.setShortcut(QtGui.QKeySequence("Space"))

        # Set up audio playback
        format = QAudioFormat()
        format.setChannelCount(1)
        format.setSampleRate(self.api.project.sampling_rate)
        format.setSampleSize(16)
        format.setCodec("audio/pcm")
        format.setByteOrder(QAudioFormat.LittleEndian)
        format.setSampleType(QAudioFormat.SignedInt)
        self.output = QAudioOutput(format, self)
        self.buffer = QBuffer()
        self.data = QByteArray()

        # Connect events
        self.api.workspaceChanged.connect(self.stop_playback)
        self.api.selectionChanged.connect(self.stop_playback)
        self.api.sourcesChanged.connect(self.stop_playback)
        self.output.stateChanged.connect(self.on_state_changed)

    def on_state_changed(self, state):
        if state == QAudio.IdleState or state == QAudio.StoppedState:
            self.button.setChecked(False)
        elif state == QAudio.ActiveState:
            self.button.setChecked(True)
        self.playback_action.setChecked(self.button.isChecked())

    def toggle_play(self):
        self.button.setChecked(not self.button.isChecked())
        self.playback_action.setChecked(self.button.isChecked())
        self.play_audio()

    def stop_playback(self):
        if self.output.state() == QAudio.ActiveState:
            self.output.stop()
        if self.buffer.isOpen():
            self.buffer.close()
        self.data.clear()

    def _prepare_buffer(self, data):
        data /= np.max(data)
        data *= 0.8
        data *= 32767
        data = data.astype(np.int16)
        self.data.clear()
        for i in range(len(data)):
            self.data.append(struct.pack("<h", data[i]))
        self.buffer.setData(self.data)
        self.buffer.open(QIODevice.ReadOnly)
        self.buffer.seek(0)

    def play_audio(self):
        if self.button.isChecked():
            # Fetch the visible data to play
            _, y_data = self.gui.ui.previewPlot.waveform_plot.getData()

            if self.output.state() == QAudio.ActiveState:
                self.output.stop()
            if self.buffer.isOpen():
                self.buffer.close()

            self._prepare_buffer(y_data)
            self.output.start(self.buffer)
        else:
            self.stop_playback()

    def plugin_toolbar_items(self):
        return [self.button]

    def add_plugin_menu(self, menu_parent):
        menu = menu_parent.addMenu("&Playback")
        menu.addAction(self.playback_action)
        return menu
