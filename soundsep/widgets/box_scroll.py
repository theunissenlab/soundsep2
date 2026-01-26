"""A plot based scrolly rectangle

Could be used like a minimap kind of thing at some point
"""
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QRectF, pyqtSignal

from .axes import ProjectIndexTimeAxis


class ProjectScrollbar(pg.PlotWidget):

    positionChanged = pyqtSignal(float, float)
    selectionChanged = pyqtSignal()  # Emitted when user creates/modifies selection

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

        # Store interval rectangles
        self.interval_rects = []

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

        # Selection region for time range selection (Shift+drag)
        self.selection_region = pg.LinearRegionItem(
            values=[0, 0],
            brush=pg.mkBrush(255, 200, 100, 80),  # Orange semi-transparent
            pen=pg.mkPen((255, 150, 50), width=2),
            movable=True,
            bounds=[0, project.frames],
        )
        self.selection_region.setVisible(False)
        self.selection_region.setZValue(10)  # Above interval rects
        self.addItem(self.selection_region)

        # Track selection state
        self._selecting = False
        self._selection_start = None

    def on_move(self):
        pos = self.rect.pos()
        size = self.rect.size()
        self.positionChanged.emit(pos.x(), pos.x() + size.x())

    def set_current_range(self, x0, x1):
        self.rect.setSize((x1 - x0, 0.8), update=False)
        self.rect.setPos((x0, 0.1), update=False)

    def mousePressEvent(self, event):
        """Handle mouse press - Shift+drag starts selection"""
        from PyQt6.QtCore import QPointF
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            pos = self.plotItem.vb.mapSceneToView(QPointF(event.pos()))
            self._selection_start = pos.x()
            self._selecting = True
            self.selection_region.setRegion([self._selection_start, self._selection_start])
            self.selection_region.setVisible(True)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move - update selection region while dragging"""
        from PyQt6.QtCore import QPointF
        if self._selecting:
            pos = self.plotItem.vb.mapSceneToView(QPointF(event.pos()))
            x = max(0, min(pos.x(), self.project.frames))
            self.selection_region.setRegion([
                min(self._selection_start, x),
                max(self._selection_start, x)
            ])
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release - finalize selection"""
        if self._selecting:
            self._selecting = False
            self.selectionChanged.emit()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

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

    def clear_intervals(self):
        """Clear all interval rectangles from the scrollbar"""
        for rect_item in self.interval_rects:
            self.removeItem(rect_item)
        self.interval_rects = []

    def add_intervals(self, intervals_data):
        """Add interval rectangles to the scrollbar
        
        Arguments
        ---------
        intervals_data : list of tuples
            List of (start_time_seconds, stop_time_seconds) tuples
        """
        self.clear_intervals()
        
        sampling_rate = self.project.sampling_rate
        
        for start_time, stop_time in intervals_data:
            # Convert time in seconds to project frames
            start_frame = int(start_time * sampling_rate)
            stop_frame = int(stop_time * sampling_rate)
            width = stop_frame - start_frame
            
            # Create a rectangle for this interval
            # Position it below the main scrollbar rect (y=0.0 to y=0.05)
            rect_item = pg.QtWidgets.QGraphicsRectItem(start_frame, 0.0, width, 0.5)
            rect_item.setBrush(pg.mkBrush(100, 150, 255, 150))  # Semi-transparent blue
            rect_item.setPen(pg.mkPen(None))  # No border
            
            self.addItem(rect_item)
            self.interval_rects.append(rect_item)

    def get_selection(self):
        """Get the current selection region in samples, or None if no selection."""
        if not self.selection_region.isVisible():
            return None
        region = self.selection_region.getRegion()
        return (int(region[0]), int(region[1]))

    def get_selection_seconds(self):
        """Get the current selection region in seconds, or None if no selection."""
        sel = self.get_selection()
        if sel is None:
            return None
        return (sel[0] / self.project.sampling_rate, sel[1] / self.project.sampling_rate)

    def clear_selection(self):
        """Clear the selection region."""
        self.selection_region.setVisible(False)
