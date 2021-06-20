import pyqtgraph as pg
import numpy as np
from PyQt5.QtCore import pyqtSignal

from soundsep.core.models import ProjectIndex
from .axes import ProjectIndexTimeAxis
from .mouse_events_view_box import MouseEventsViewBox



class PreviewPlot(pg.PlotWidget):
    """
    """
    fineSelectionCleared = pyqtSignal()
    fineSelectionMade = pyqtSignal(float, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, viewBox=MouseEventsViewBox(), **kwargs)
        self.region_select = None

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
        self.getViewBox().dragInProgress.connect(self.on_drag)
        self.getViewBox().clicked.connect(self.on_click)
        self.getViewBox().zoomEvent.connect(self.on_zoom_event)

    def on_drag(self, from_, to):
        if self.region_select is None:
            self.region_select = pg.LinearRegionItem((from_.x(), to.x()))
            self.addItem(self.region_select)
            self.region_select.sigRegionChanged.connect(self.on_region_changed)
        else:
            self.region_select.setRegion((from_.x(), to.x()))

    def on_region_changed(self):
        self.fineSelectionMade.emit(*self.region_select.getRegion())

    def on_click(self, pos):
        self.clear_region()

    def clear_region(self):
        self.removeItem(self.region_select)
        self.region_select = None
        self.fineSelectionCleared.emit()

    def on_plot_change(self):
        xrange, yrange = self.waveform_plot.getData()
        xrange_ampenv, yrange_ampenv = self.ampenv_plot.getData()

        if len(xrange):
            xmin = xrange[0]
            xmax = xrange[-1]
            ymax = np.max(np.abs(np.concatenate([yrange, yrange_ampenv])))
            self.setXRange(xmin, xmax, padding=0.0)
            self.setYRange(-ymax, ymax, padding=0.0)

        self.clear_region()

    def on_zoom_event(self, direction, position):
        xrange, yrange = self.waveform_plot.getData()
        if len(xrange):
            xmin = xrange[0]
            xmax = xrange[-1]
            (currxmin, currxmax), _ = self.getViewBox().viewRange()
            currsize = currxmax - currxmin
            if direction > 0:
                target_size = currsize * 0.8
            else:
                target_size = currsize * 1.25
            target_size = min(xmax - xmin, target_size)
            cursor_frac = (position.x() - currxmin) / currsize
            target_center = position.x()
            target_bounds = [
                target_center - cursor_frac * target_size,
                target_center + (1 - cursor_frac) * target_size
            ]

            if target_bounds[0] <= xmin:
                target_bounds[0] = xmin
                target_bounds[1] = min(xmin + target_size, xmax)
            elif target_bounds[1] >= xmax:
                target_bounds[1] = xmax
                target_bounds[0] = max(xmax - target_size, xmin)
            self.setXRange(target_bounds[0], target_bounds[1], padding=0.0)
