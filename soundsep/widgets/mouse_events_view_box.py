"""Viewbox for interactions with spectrogram view
"""

import pyqtgraph as pg

from PyQt5.QtCore import Qt, QPoint, QPointF, pyqtSignal


class MouseEventsViewBox(pg.ViewBox):
    """A ViewBox for the Spectrogram viewer to handle user interactions

    This object manages click, drag, and wheel events on the spectrogram
    view image.
    """
    dragComplete = pyqtSignal(QPointF, QPointF)
    dragInProgress = pyqtSignal(QPointF, QPointF)
    hovered = pyqtSignal(QPointF)
    clicked = pyqtSignal(QPointF)
    contextMenuRequested = pyqtSignal(QPoint)
    zoomEvent = pyqtSignal(int, object)

    def raiseContextMenu(self, ev):
        super().raiseContextMenu(ev)
        pos = ev.screenPos()
        self.contextMenuRequested.emit(pos.toPoint())

    def mouseDragEvent(self, event):
        if event.button() == Qt.LeftButton:
            event.accept()
            start_pos = self.mapSceneToView(event.buttonDownScenePos())
            end_pos = self.mapSceneToView(event.scenePos())
            if event.isFinish():
                self.dragComplete.emit(start_pos, end_pos)
            else:
                self.dragInProgress.emit(start_pos, end_pos)

    def mouseClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            event.accept()
            self.clicked.emit(self.mapSceneToView(event.scenePos()))
        elif event.button() == Qt.RightButton:
            event.accept()
            self.raiseContextMenu(event)
        else:
            super().mouseClickEvent(event)

    def wheelEvent(self, event):
        """Emits the direction of scroll and the location in fractional position"""
        pos = self.mapSceneToView(event.scenePos())
        if event.delta() > 0:
            self.zoomEvent.emit(1, pos)
        elif event.delta() < 0:
            self.zoomEvent.emit(-1, pos)
