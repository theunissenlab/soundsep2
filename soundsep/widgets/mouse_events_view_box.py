"""Viewbox for interactions with spectrogram view
"""

import pyqtgraph as pg

from PyQt5.QtCore import Qt, QPointF, pyqtSignal


class MouseEventsViewBox(pg.ViewBox):
    """A ViewBox for the Spectrogram viewer to handle user interactions

    This object manages click, drag, and wheel events on the spectrogram
    view image.
    """
    dragComplete = pyqtSignal(QPointF, QPointF)
    dragInProgress = pyqtSignal(QPointF, QPointF)
    hovered = pyqtSignal(QPointF)
    clicked = pyqtSignal(QPointF)
    zoomEvent = pyqtSignal(int, object)

    def mouseDragEvent(self, event):
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
        else:
            super().mouseClickEvent(event)

    def wheelEvent(self, event):
        """Emits the direction of scroll and the location in fractional position"""
        pos = self.mapSceneToView(event.scenePos())
        if event.delta() > 0:
            self.zoomEvent.emit(1, pos)
        elif event.delta() < 0:
            self.zoomEvent.emit(-1, pos)
