"""
An example of drawing an adjustable ROI on a spectrogram view

* Click and drag on the image to select a region
* Adjust region size
* Click away from region to clear selection
* TODO: show selected range on screen
* TODO: show currently hovered frequency and time at all times
"""

import numpy as np
import pyqtgraph as pg
import PyQt6.QtWidgets as widgets
from PyQt6 import QtGui

from soundsep.gui.main import run_app
from soundsep.gui.stft_view import ScrollableSpectrogram, ScrollableSpectrogramConfig
from soundsep.gui.components.selection_box import SelectionBox
from soundsep.gui.components.spectrogram_view_box import SpectrogramViewBox


example_file = "/home/kevin/Data/from_pepe/2018_02_11 16-31-00/ch0.wav"
example_channel = 0


class SelectionBoxExample(widgets.QWidget):
    """Selection box example. Uses the spectrogram example as a background to draw on

    Draws when click and dragged
    """
    def __init__(self):
        super().__init__()
        self.title = "Selection Box Example"

        self.current_selection_roi = None
        self._init_spectrogram_view()

    def _init_spectrogram_view(self):
        spectrogram_viewbox = SpectrogramViewBox()
        self.spectrogram_panel = ScrollableSpectrogram(
            filename=example_file,
            channel=example_channel,
            config=ScrollableSpectrogramConfig(
                window_size=500,
                window_step=22,
                spectrogram_size=22050 * 5,
                cmap="turbo",
            ),
            viewBox=spectrogram_viewbox
        )

        spectrogram_viewbox.dragComplete.connect(self.on_drag_complete)
        spectrogram_viewbox.dragInProgress.connect(self.on_drag_in_progress)
        spectrogram_viewbox.clicked.connect(self.on_click)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.spectrogram_panel)
        self.setLayout(layout)
        self.spectrogram_panel.scroll_to(0)

    def on_drag_complete(self, from_, to):
        pass

    def on_drag_in_progress(self, from_, to):
        if self.current_selection_roi is None:
            self.draw_selection_box(from_, to)
        else:
            line = to - from_
            self.current_selection_roi.setPos([
                min(to.x(), from_.x()),
                min(to.y(), from_.y())
            ])
            self.current_selection_roi.setSize([
                np.abs(line.x()),
                np.abs(line.y())
            ])

    def on_click(self, at):
        self.clear_selection_box()

    def clear_selection_box(self):
        if self.current_selection_roi is not None:
            self.spectrogram_panel.removeItem(self.current_selection_roi)
            self.current_selection_roi.deleteLater()
            self.current_selection_roi = None

    def draw_selection_box(self, from_, to):
        self.current_selection_roi = SelectionBox(
            pos=from_,
            size=to - from_,
            pen=(156, 156, 100),
            rotatable=False,
            removable=False,
            maxBounds=self.spectrogram_panel.image.boundingRect()
        )
        self.current_selection_roi.sigRegionChangeFinished.connect(
            self.on_selection_changed
        )
        self.spectrogram_panel.addItem(self.current_selection_roi)

    def on_selection_changed(self, roi):
        x_bounds, y_bounds = roi.get_scaled_bounds(
            self.spectrogram_panel.image,
            self.spectrogram_panel.xlim(),
            self.spectrogram_panel.ylim(),
        )


if __name__ == "__main__":
    run_app(SelectionBoxExample)
