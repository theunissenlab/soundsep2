from enum import Enum

import pyqtgraph as pg
import numpy as np
from PyQt5 import QtWidgets as widgets
from PyQt5 import QtGui
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal

from soundsep.gui.components.overlays import FloatingButton
from soundsep.gui.components.spectrogram_view_box import SpectrogramViewBox
from soundsep.core.models import ProjectIndex, StftIndex
from soundsep.core.stft import spectral_derivative


class STFTViewMode(Enum):
    NORMAL = 1
    DERIVATIVE = 2


class FrequencyAxis(pg.AxisItem):
    """Frequency axis in kHz for spectrograms
    """
    def tickStrings(self, values, scale, spacing):
        return ["{}k".format(int(value // 1000)) for value in values]


class StftTimeAxis(pg.AxisItem):
    """Time axis converting StftIndex into timestamps
    """

    def __init__(self, *args, project, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = project

    def _format_time(self, t: float):
        """Format time in seconds to form hh:mm:ss"""
        h = int(t / 3600)
        t -= h * 3600
        m = int(t / 60)
        t -= m * 60
        s = t
        return "{}:{:02d}:{:.2f}".format(h, m, s)

    def _to_timestamp(self, x):
        return ProjectIndex(self.project, x).to_timestamp()

    def tickStrings(self, values, scale, spacing):
        return [self._format_time(self._to_timestamp(value)) for value in values]


class SourceView(widgets.QWidget):

    hover = pyqtSignal(ProjectIndex, float)

    def __init__(self, source, parent=None):
        super().__init__(parent=parent)
        self.source = source
        self.init_ui()

    def init_ui(self):
        self.layout = widgets.QVBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.spectrogram = ScrollableSpectrogram()

        self._stft_time_axis = StftTimeAxis(project=self.source.project, orientation="bottom")
        self.spectrogram.setAxisItems({
            "left": FrequencyAxis(orientation="left"),
            "bottom": self._stft_time_axis,
        })

        self.spectrogram.scene().sigMouseMoved.connect(self.on_sig_mouse_moved)

        self.dialog = FloatingButton(
            "▼ {}".format(self.source.name),
            paddingx=80,
            paddingy=20,
            parent=self.spectrogram
        )

        self.layout.addWidget(self.spectrogram)
        self.setLayout(self.layout)

    def on_sig_mouse_moved(self, event):
        pos = self.spectrogram.getViewBox().mapSceneToView(event)
        self.hover.emit(ProjectIndex(self.source.project, int(pos.x())), pos.y())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.dialog.update_position()


class ScrollableSpectrogram(pg.PlotWidget):
    """Scrollable Spectrogram Plot Widget that uses the spectrogram indices as its native coordinate system
    """

    def __init__(self, parent=None):
        super().__init__(viewBox=SpectrogramViewBox(), background=None, parent=parent)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        # self.setCursor(Qt.CrossCursor)
        self.hideButtons()

        self._data = None  # Caches the values of the set data in case the view mode is switched

        self.init_ui()

    def set_view_mode(self, view_mode: STFTViewMode):
        self._view_mode = view_mode
        if self._view_mode == STFTViewMode.DERIVATIVE:
            self.set_cmap("gist_yarg")
            if self._data is not None:
                self.image.setImage(spectral_derivative(self._data))
                self.image.setTransform(self._tr)
        elif self._view_mode == STFTViewMode.NORMAL:
            self.set_cmap("turbo")
            if self._data is not None:
                self.image.setImage(self._data)
                self.image.setTransform(self._tr)
        else:
            raise RuntimeError("Invalid _view_mode {}".format(self._view_mode))

    def init_ui(self):
        self.image = pg.ImageItem()
        # TODO: dynamic cursor changing based on hover and what action would be activated?
        self.image.setCursor(Qt.CrossCursor)
        self.addItem(self.image)
        # TODO hardcoded? make this configurable
        self.set_view_mode(STFTViewMode.NORMAL)

    def get_limits_rect(self) -> QRectF:
        """Return the plot limits (i0: StftIndex, f0: float, width: int, height: float) as a QRect
        """
        (x0, x1), (y0, y1) = self.viewRange()
        return QRectF(x0, y0, x1 - x0, y1 - y0)

    def set_data(self, i0: StftIndex, i1: StftIndex, data: np.ndarray, freqs: np.ndarray):
        positive_freqs = freqs >= 0
        data = data[:, positive_freqs]
        freqs = freqs[positive_freqs]

        df = freqs[1] - freqs[0]
        self._tr = QtGui.QTransform()
        self._tr.translate(i0.to_project_index(), 0)
        self._tr.scale(i0.step, df)
        self._data = data

        if self._view_mode == STFTViewMode.NORMAL:
            self.image.setImage(data)
            self.image.setTransform(self._tr)
        elif self._view_mode == STFTViewMode.DERIVATIVE:
            self.image.setImage(spectral_derivative(data))
            self.image.setTransform(self._tr)
        else:
            raise RuntimeError("Invalid _view_mode {}".format(self._view_mode))

        self.setXRange(int(i0.to_project_index()), int(i1.to_project_index()), padding=0.0)
        self.setYRange(freqs[0], freqs[-1], padding=0.0)

    def set_cmap(self, cmap):
        # TODO this doesnt use config at all
        self._cmap = pg.colormap.get(cmap, source='matplotlib', skipCache=True)
        self.image.setLookupTable(self._cmap.getLookupTable(alpha=True))
