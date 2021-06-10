"""Selection box that can change size with control points at corners and sides"""

from typing import Tuple

import pyqtgraph as pg
from pyqtgraph.graphicsItems.ROI import Handle


class SelectionBox(pg.RectROI):
    """Wrapper of pyqtgraph RectROI object with 8 control points

    Emits events when the size changes in plot coordinates?
    """

    _roi_color = (255, 255, 0)
    _handle_color = (200, 200, 220)
    _handle_hover_color = (255, 255, 0)
    _handle_radius = 12
    _handle_type = "t"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPen(self._roi_color, width=2, cosmetic=True)

        for x in (0.0, 0.5, 1.0):
            for y in (0.0, 0.5, 1.0):
                if x == 0.5 and y == 0.5:
                    continue
                handle = Handle(
                    radius=self._handle_radius,
                    typ=self._handle_type,
                    pen=self._handle_color,
                    hoverPen=self._handle_hover_color,
                    parent=self,
                )
                handle.pen.setWidth(2)
                self.addScaleHandle(
                    (x, y),
                    (1 - x, 1 - y),
                    item=handle
                )

    def get_scaled_bounds(self, img: pg.ImageItem, xlim: Tuple[float, float], ylim: Tuple[float, float]):
        """Return the roi bounds of an image, scaled to the given data coordinates

        Arguments
        ---------
        img : pg.ImageItem
            2D image object that exposes .width() and .height() methods for its pixel dimensions
        xlim : Tuple[float, float]
            x-limits to scale ROI region to, corresponding to the .width() dimension of the image
        ylim : Tuple[float, float]
            y limits to scale ROI region to, corresponding to the .height() dimension of the image

        Returns
        -------
        xbounds : Tuple[float, float]
            ROI bounds in x coordinates given by xlim. The values are within the range
            xlim, inclusive.
        ybounds : Tuple[float, float]
            ROI bounds in y coordinates given by ylim. The values are within the range
            ylim, inclusive.
        """
        tr = self.sceneTransform() * pg.fn.invertQTransform(img.sceneTransform())
        tr.scale(float(xlim[1] - xlim[0]) / img.width(), float(ylim[1] - ylim[0]) / img.height())

        x0 = xlim[0] + tr.m11() * tr.dx()
        x1 = x0 + tr.m11() * self.boundingRect().width()

        y0 = ylim[0] + tr.m22() * tr.dy()
        y1 = y0 + tr.m22() * self.boundingRect().height()

        return (x0, x1), (y0, y1)


