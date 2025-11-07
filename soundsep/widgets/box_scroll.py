"""A plot based scrolly rectangle

Could be used like a minimap kind of thing at some point
"""
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QRectF, pyqtSignal

from .axes import ProjectIndexTimeAxis


class ProjectScrollbar(pg.PlotWidget):

    positionChanged = pyqtSignal(float, float)

    def __init__(self, project, parent=None):
        super().__init__(parent=parent)
        self.project = project
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideAxis("left")
        self.disableAutoRange()
        self.setMaximumHeight(80)
        self.plotItem.setMaximumHeight(80)
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

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to navigate to clicked position"""
        # Get the mouse position in the scene - convert QPoint to QPointF
        from PyQt6.QtCore import QPointF
        pos = QPointF(event.pos())
        scene_pos = self.plotItem.vb.mapSceneToView(pos)
        clicked_x = scene_pos.x()
        
        # Get current window size
        current_size = self.rect.size().x()
        
        # Calculate new position centered on clicked position
        new_x0 = max(0, clicked_x - current_size / 2)
        new_x1 = new_x0 + current_size
        
        # Make sure we don't go past the end
        if new_x1 > self.project.frames:
            new_x1 = self.project.frames
            new_x0 = max(0, new_x1 - current_size)
        
        # Update the rect position
        self.rect.setPos((new_x0, 0.1), update=True)
        
        # Let the parent handle the event too
        super().mouseDoubleClickEvent(event)
