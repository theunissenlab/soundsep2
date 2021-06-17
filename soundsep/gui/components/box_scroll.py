"""A plot based scrolly rectangle

Could be used like a minimap kind of thing at some point
"""



import pyqtgraph as pg
from PyQt5.QtCore import Qt, QRectF, pyqtSignal

from .axes import ProjectIndexTimeAxis


class ProjectScrollbar(pg.PlotWidget):

    positionChanged = pyqtSignal(float, float)

    def __init__(self, project, parent=None):
        super().__init__(parent=parent)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideAxis("left")
        self.disableAutoRange()
        self.setMaximumHeight(80)
        self.plotItem.setMaximumHeight(80)
        self.setCursor(Qt.SplitHCursor)
        self.hideButtons()

        self.setAxisItems({
            "bottom": ProjectIndexTimeAxis(project=project, orientation="bottom"),
        })
        self.setXRange(0, project.frames, padding=0.0)
        self.setYRange(0, 1, padding=0.0)

        self.rect = pg.RectROI(0, 0, 1, 1,
            movable=True,
            pen=pg.mkPen("r", width=4),
            hoverPen=pg.mkPen("r", width=6),
            resizable=False,
        )
        self.addItem(self.rect)
        for handle in self.rect.getHandles():
            self.rect.removeHandle(handle)
        self.rect.maxBounds = QRectF(0, 0.1, project.frames, 0.8)
        self.rect.sigRegionChanged.connect(self.on_move)

    def on_move(self):
        pos = self.rect.pos()
        size = self.rect.size()
        self.positionChanged.emit(pos.x(), pos.x() + size.x())

    def set_current_range(self, x0, x1):
        self.rect.setSize((x1 - x0, 0.8), update=False)
        self.rect.setPos((x0, 0.1), update=False)
