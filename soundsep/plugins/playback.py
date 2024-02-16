import struct

from PyQt6.QtMultimedia import QAudio, QAudioFormat, QAudioOutput, QSoundEffect, QMediaPlayer, QAudioDevice, QAudioSink
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice
from PyQt6 import QtGui
from PyQt6 import QtWidgets as widgets
import numpy as np

from soundsep.core.base_plugin import BasePlugin


class PlaybackPlugin(BasePlugin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize UI
        self.button = widgets.QPushButton("ᐅ Play")
        self.button.setCheckable(True)
        self.button.clicked.connect(self.play_audio)

        self.playback_action = QtGui.QAction("Play Selection")
        self.playback_action.setCheckable(True)
        self.playback_action.triggered.connect(self.toggle_play)
        self.playback_action.setShortcut(QtGui.QKeySequence("Space"))

        # Set up audio playback
        format_ = QAudioFormat()
        format_.setChannelCount(1)
        format_.setSampleRate(self.api.project.sampling_rate)
        format_.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        self.qaudiosinkFormat = format_

        self.output = QAudioSink(format_, self)
        self.buffer = QBuffer()
        self.data = QByteArray()

        # Connect events
        self.api.workspaceChanged.connect(self.stop_playback)
        self.api.selectionChanged.connect(self.stop_playback)
        self.api.sourcesChanged.connect(self.stop_playback)
        self.api.closingProgram.connect(self.cleanup)
        self.output.stateChanged.connect(self.on_state_changed)

    def on_state_changed(self, state):
        if state == QAudio.State.IdleState or state == QAudio.State.StoppedState:
            self.button.setChecked(False)
        elif state == QAudio.State.ActiveState:
            self.button.setChecked(True)
        self.playback_action.setChecked(self.button.isChecked())

    def toggle_play(self):
        self.button.setChecked(not self.button.isChecked())
        self.playback_action.setChecked(self.button.isChecked())
        self.play_audio()

    def cleanup(self):
        self.stop_playback()

    def stop_playback(self):
        self.output.stop()
        if self.buffer.isOpen():
            self.buffer.close()
        self.data.clear()
        self._create_output()

    def _prepare_buffer(self, data):
        data /= np.max(data)
        data *= 0.8
        data *= 32767
        data = data.astype(np.int16)
        self.data.clear()
        for i in range(len(data)):
            self.data.append(struct.pack("<h", data[i]))
        self.buffer.setData(self.data)
        self.buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        self.buffer.seek(0)

    def _create_output(self):
        del self.output
        self.output = QAudioSink(self.qaudiosinkFormat, self)
        self.output.stateChanged.connect(self.on_state_changed)


    def play_audio(self):
        if self.button.isChecked():
            # Fetch the visible data to play
            _, y_data = self.gui.ui.previewPlot.waveform_plot.getData()

            if self.output.state() != QAudio.State.StoppedState:
                self.stop_playback()
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
