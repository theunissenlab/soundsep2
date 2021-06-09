import pyqtgraph as pg
import PyQt5.QtWidgets as widgets
from PyQt5 import QtGui

from soundsep.core.io import ProjectIndex
from soundsep.gui.stft_view import ScrollableSpectrogram, ScrollableSpectrogramConfig


class ViewState(object):
    """Dummy state object for examples"""
    def __init__(self, t0, width, step):
        self.t0 = t0
        self.width = width
        self.step = step

    def next(self):
        self.t0 += self.step

    def next_page(self):
        self.t0 += self.width

    def prev(self):
        self.t0 -= self.step

    def prev_page(self):
        self.t0 -= self.width

    def slice(self):
        return slice(self.t0, self.t0 + width)


class MainApp(widgets.QWidget):
    def __init__(self, project):
        super().__init__()
        self.title = "SoundSep"
        self.project = project
        self.init_ui()

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        layout.addWidget(STFTWindow(self.project, 0))
        self.setLayout(layout)


class STFTWindow(pg.GraphicsLayoutWidget):
    def __init__(self, project, channel):
        super().__init__()
        self.project = project
        self.channel = channel

        self.view_state = ViewState(
            ProjectIndex(project, 0),
            22050,
            22050 * 3,
        )

        self._init_shortcuts()
        self._init_ui()
        self.spectrogram_panel.scroll_to(0)

    def _init_shortcuts(self):
        self.next_shortcut = widgets.QShortcut(QtGui.QKeySequence("D"), self)
        self.next_shortcut.activated.connect(self.next)

        self.next_page_shortcut = widgets.QShortcut(QtGui.QKeySequence("Shift+D"), self)
        self.next_page_shortcut.activated.connect(self.next_page)

        self.prev_shortcut = widgets.QShortcut(QtGui.QKeySequence("A"), self)
        self.prev_shortcut.activated.connect(self.prev)

        self.prev_page_shortcut = widgets.QShortcut(QtGui.QKeySequence("Shift+A"), self)
        self.prev_page_shortcut.activated.connect(self.prev_page)

    def _init_ui(self):
        self.spectrogram_panel = ScrollableSpectrogram(
            project=self.project,
            channel=self.channel,
            config=ScrollableSpectrogramConfig(
                window_size=500,
                window_step=22,
                spectrogram_size=22050 * 5,
                cmap="turbo",
            )
        )

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.spectrogram_panel)
        self.setLayout(layout)

    def prev(self):
        self.view_state.prev()
        self.spectrogram_panel.scroll_to(self.view_state.t0)

    def prev_page(self):
        self.view_state.prev_page()
        self.spectrogram_panel.scroll_to(self.view_state.t0)

    def next(self):
        self.view_state.next()
        self.spectrogram_panel.scroll_to(self.view_state.t0)

    def next_page(self):
        self.view_state.next_page()
        self.spectrogram_panel.scroll_to(self.view_state.t0)
