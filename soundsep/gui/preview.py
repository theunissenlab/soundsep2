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
        self.disableAutoRange()
        self.hideButtons()

        self.waveform_plot = pg.PlotCurveItem()
        self.waveform_plot.setPen(pg.mkPen((130, 120, 200), width=1))
        self.addItem(self.waveform_plot)

        self.ampenv_plot = pg.PlotCurveItem()
        self.ampenv_plot.setPen(pg.mkPen((200, 20, 20), width=3))
        self.addItem(self.ampenv_plot)

        self.waveform_plot.sigPlotChanged.connect(self.on_plot_change)
        self.ampenv_plot.sigPlotChanged.connect(self.on_plot_change)

    def on_plot_change(self):
        xrange, yrange = self.waveform_plot.getData()
        xrange_ampenv, yrange_ampenv = self.ampenv_plot.getData()

        xmin = xrange[0]
        xmax = xrange[-1]
        ymax = np.max(np.abs(np.concatenate([yrange, yrange_ampenv])))
        self.setXRange(xmin, xmax)
        self.setYRange(-ymax, ymax)
