"""An example of navigation and paging through a spectrogram view

* Keyboard shortcuts for moving left and right
* Ability to modify the colormap
* Switching between normal and spectral derivative view
* TODO: Correct time and frequency axes
* TODO: Minimap scrolling
"""

import pyqtgraph as pg
import PyQt6.QtWidgets as widgets
from PyQt6 import QtGui

from soundsep.gui.main import run_app
from soundsep.gui.stft_view import ScrollableSpectrogram, ScrollableSpectrogramConfig


example_file = "/home/kevin/Data/from_pepe/2018_02_11 16-31-00/ch0.wav"
example_channel = 0


class ViewState(object):
    """Dummy state object for examples"""
    t0 = 0
    t1 = 22050 * 5  # 5 'seconds' of data
    step = 11025

    @classmethod
    def next(cls):
        cls.t0 += cls.step
        cls.t1 += cls.step

    @classmethod
    def next_page(cls):
        cls.t0 += cls.step * 5
        cls.t1 += cls.step * 5

    @classmethod
    def prev(cls):
        cls.t0 -= cls.step
        cls.t1 -= cls.step

    @classmethod
    def prev_page(cls):
        cls.t0 -= cls.step * 5
        cls.t1 -= cls.step * 5

    @classmethod
    def slice(cls):
        return slice(cls.t0, cls.t1)


class STFTWindowExample(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        self.title = "STFT Window Example"
        self._init_shortcuts()
        self._init_ui()

        self.spectrogram_panel.scroll_to(0)

    def _init_shortcuts(self):
        self.next_shortcut = QtGui.QShortcut(QtGui.QKeySequence("D"), self)
        self.next_shortcut.activated.connect(self.next)

        self.next_page_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Shift+D"), self)
        self.next_page_shortcut.activated.connect(self.next_page)

        self.prev_shortcut = QtGui.QShortcut(QtGui.QKeySequence("A"), self)
        self.prev_shortcut.activated.connect(self.prev)

        self.prev_page_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Shift+A"), self)
        self.prev_page_shortcut.activated.connect(self.prev_page)

    def _init_ui(self):
        self.spectrogram_panel = ScrollableSpectrogram(
            filename=example_file,
            channel=example_channel,
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
        ViewState.prev()
        self.spectrogram_panel.scroll_to(ViewState.t0)

    def prev_page(self):
        ViewState.prev_page()
        self.spectrogram_panel.scroll_to(ViewState.t0)

    def next(self):
        ViewState.next()
        self.spectrogram_panel.scroll_to(ViewState.t0)

    def next_page(self):
        ViewState.next_page()
        self.spectrogram_panel.scroll_to(ViewState.t0)


if __name__ == "__main__":
    run_app(STFTWindowExample)
