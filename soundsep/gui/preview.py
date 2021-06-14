import pyqtgraph as pg
import numpy as np

from PyQt5 import QtWidgets as widgets
from PyQt5 import QtGui


class PreviewPlot(pg.PlotWidget):
    """Preview plot. Has an automatic
    """
    def __init__(self, project, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = project
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()

        self.waveform_plot = pg.PlotCurveItem()
        self.waveform_plot.setPen(pg.mkPen((130, 120, 200), width=1))
        self.addItem(self.waveform_plot)

        self.ampenv_plot = pg.PlotCurveItem()
        self.ampenv_plot.setPen(pg.mkPen((200, 20, 20), width=3))
        self.addItem(self.ampenv_plot)
