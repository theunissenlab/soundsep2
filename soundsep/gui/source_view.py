from enum import Enum

import pyqtgraph as pg
import numpy as np
from PyQt5 import QtWidgets as widgets
from PyQt5 import QtGui
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal

from soundsep.gui.components.overlays import FloatingButton, FloatingComboBox
from soundsep.gui.components.spectrogram_view_box import SpectrogramViewBox
from soundsep.core.models import ProjectIndex, StftIndex, Source
from soundsep.core.stft import spectral_derivative
from soundsep.gui.components.axes import ProjectIndexTimeAxis, FrequencyAxis


class STFTViewMode(Enum):
    NORMAL = 1
    DERIVATIVE = 2


class EditSourceModal(widgets.QDialog):

    deleteSourceSignal = pyqtSignal(Source)
    editSourceSignal = pyqtSignal(Source)

    def __init__(self, source, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = source

        self.setModal(True)
        self.setWindowTitle("Edit source information")

        layout = widgets.QVBoxLayout(self)
        layout.addWidget(widgets.QLabel("Source Name"))
        self.line_edit = widgets.QLineEdit(self)
        self.line_edit.setText(self.source.name)
        layout.addWidget(self.line_edit)

        button_box = widgets.QDialogButtonBox(
            widgets.QDialogButtonBox.Help
            | widgets.QDialogButtonBox.Ok
            | widgets.QDialogButtonBox.Cancel
        )
        button_box.button(widgets.QDialogButtonBox.Help).setText("Delete Source")

        layout.addWidget(button_box)
        self.setLayout(layout)

        # Im just using helpRequested as the delete signal
        button_box.helpRequested.connect(self.delete)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

    def delete(self):
        """Sends a delete signal"""
        self.deleteSourceSignal.emit(self.source)
        super().accept()

    def accept(self):
        """Sends signal to edit the source"""
        self.source.name = self.line_edit.text()
        self.editSourceSignal.emit(self.source)
        super().accept()


class SourceView(widgets.QWidget):

    editSourceSignal = pyqtSignal(Source)
    deleteSourceSignal = pyqtSignal(Source)
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

        self.spectrogram.setAxisItems({
            "left": FrequencyAxis(orientation="left"),
            "bottom": ProjectIndexTimeAxis(project=self.source.project, orientation="bottom"),
        })

        self.spectrogram.scene().sigMouseMoved.connect(self.on_sig_mouse_moved)

        self.source_channel_dialog = FloatingComboBox(
            paddingx=80,
            paddingy=10,
            parent=self.spectrogram
        )
        self.source_name_dialog = FloatingButton(
            "▼ {}".format(self.source.name),
            paddingx=140,
            paddingy=10,
            parent=self.spectrogram
        )
        self.source_name_dialog.clicked.connect(self.open_edit_source_modal)

        self.source_channel_dialog.addItems([str(ch) for ch in range(self.source.project.channels)])
        self.source_channel_dialog.setCurrentIndex(self.source.channel)
        self.source_channel_dialog.currentIndexChanged.connect(self.on_source_channel_changed)

        self.layout.addWidget(self.spectrogram)
        self.setLayout(self.layout)

    def on_source_channel_changed(self, new_channel):
        self.source.channel = new_channel
        self.editSourceSignal.emit(self.source)

    def open_edit_source_modal(self):
        dialog = EditSourceModal(self.source)
        dialog.editSourceSignal.connect(self.editSourceSignal.emit)
        dialog.deleteSourceSignal.connect(self.deleteSourceSignal.emit)
        dialog.exec()

    def on_sig_mouse_moved(self, event):
        pos = self.spectrogram.getViewBox().mapSceneToView(event)
        self.hover.emit(ProjectIndex(self.source.project, int(pos.x())), pos.y())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.source_name_dialog.update_position()
        self.source_channel_dialog.update_position()


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

    def overlay(self, x, y):
        self._overlay_plot.setData(x, y)

    def clear_overlay(self):
        self._overlay_plot.setData([], [])

    def set_view_mode(self, view_mode: STFTViewMode):
        self._view_mode = view_mode
        if self._view_mode == STFTViewMode.DERIVATIVE:
            self.set_cmap("gist_yarg")
            if self._data is not None:
                self.image.setImage(spectral_derivative(self._data))
                self.image.setTransform(self._tr)
        elif self._view_mode == STFTViewMode.NORMAL:
            # TODO: allow for configuration of the cmap
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

        self._overlay_plot = pg.PlotCurveItem()
        self._overlay_plot.setPen(pg.mkPen("r", width=2))
        self.addItem(self._overlay_plot)

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
        self._cmap = pg.colormap.get(cmap, source='matplotlib', skipCache=True)
        self.image.setLookupTable(self._cmap.getLookupTable(alpha=True))
