"""Selection box that can change size with control points at corners and sides"""

import pyqtgraph as pg

from pyqtgraph.graphicsItems.ROI import Handle

# import PyQt5.QtWidgets as widgets


class SelectionBox(pg.RectROI):
    """A wrapper on the pyqtgraph ROI object specifically for our spectrograms

    Emits events when the size changes in plot coordinates?
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # for handle in self.getHandles():
        #     self.removeHandle(handle)

        self.setPen((255, 255, 0), width=2, cosmetic=True)

        for x in (0.0, 0.5, 1.0):
            for y in (0.0, 0.5, 1.0):
                if x == 0.5 and y == 0.5:
                    continue
                handle = Handle(
                    radius=10,
                    typ="t",
                    pen=(200, 200, 220),
                    hoverPen=(255, 255, 0),
                    parent=self,
                )
                handle.pen.setWidth(2)
                self.addScaleHandle(
                    (x, y),
                    (1 - x, 1 - y),
                    item=handle
                )

    def get_scaled_bounds(self, img, xlim, ylim):
        """Return the roi bounds of an image, scaled to the given data coordinates

        :param img: Image the roi is being drawn on
        :type img: pyqtgraph.ImageItem
        :param xlim: The x limits to scale the roi width to
        :type xlim: tuple(float, float)
        :param ylim: The y limits to scale the roi height to
        :type ylim: tuple(float, float)
        """
        tr = self.sceneTransform() * pg.fn.invertQTransform(img.sceneTransform())
        tr.scale(float(xlim[1] - xlim[0]) / img.width(), float(ylim[1] - ylim[0]) / img.height())

        x0 = xlim[0] + tr.m11() * tr.dx()
        x1 = x0 + tr.m11() * self.boundingRect().width()

        y0 = ylim[0] + tr.m22() * tr.dy()
        y1 = y0 + tr.m22() * self.boundingRect().height()

        return (x0, x1), (y0, y1)


