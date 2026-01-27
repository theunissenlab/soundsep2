from PyQt6.QtMultimedia import QAudio, QAudioFormat, QAudioSink
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
        # Disconnect state handler to prevent callbacks during cleanup
        try:
            self.output.stateChanged.disconnect(self.on_state_changed)
        except (TypeError, RuntimeError):
            pass

        # Use reset() instead of stop() - it immediately halts the audio thread
        # rather than draining buffers first
        self.output.reset()
        self.output.deleteLater()

        # Don't close the buffer here - let it be replaced in _prepare_buffer
        # This avoids race conditions where the audio thread might still be reading
        # The old buffer/data will be garbage collected when replaced

        # Create a fresh output for next playback
        self.output = QAudioSink(self.qaudiosinkFormat, self)
        self.output.stateChanged.connect(self.on_state_changed)

    def _prepare_buffer(self, data):
        # Copy data to avoid modifying the original
        data = data.copy()
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
        data = data * 0.8
        data = (data * 32767).astype(np.int16)

        # Close any existing buffer before creating new one
        # Safe to do here since output was already reset in stop_playback
        if self.buffer.isOpen():
            self.buffer.close()

        # Create fresh buffer and data objects for each playback
        self.data = QByteArray()
        self.data.append(data.tobytes())

        # Create a new buffer each time to avoid state issues
        self.buffer = QBuffer()
        self.buffer.setData(self.data)
        self.buffer.open(QIODevice.OpenModeFlag.ReadOnly)

    def play_audio(self):
        # Check if we're currently playing (before button state change takes effect)
        was_playing = self.output.state() == QAudio.State.ActiveState

        if self.button.isChecked() or was_playing:
            # If was playing, we're restarting - ensure button stays checked
            if was_playing:
                self.button.setChecked(True)
                self.playback_action.setChecked(True)

            # Fetch the visible data to play
            _, y_data = self.gui.ui.previewPlot.waveform_plot.getData()

            if y_data is None or len(y_data) == 0:
                self.button.setChecked(False)
                self.playback_action.setChecked(False)
                return

            # Stop any existing playback (also creates fresh output)
            self.stop_playback()

            # Prepare buffer with audio data
            self._prepare_buffer(y_data)

            # Start playback
            self.output.start(self.buffer)
        else:
            self.stop_playback()

    def plugin_toolbar_items(self):
        return [self.button]

    def add_plugin_menu(self, menu_parent):
        menu = menu_parent.addMenu("&Playback")
        menu.addAction(self.playback_action)
        return menu
