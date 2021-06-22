import pyqtgraph as pg

from soundsep.core.models import ProjectIndex
from soundsep.core.utils import hhmmss


class FrequencyAxis(pg.AxisItem):
    """Frequency axis in kHz for spectrograms

    Drops the tick equalling zero if it is there
    """
    def tickStrings(self, values, scale, spacing):
        return ["{}k".format(int(value // 1000)) for value in values]

    def tickValues(self, *args, **kwargs):
        vals = super().tickValues(*args, **kwargs)
        vals = [
            (_spacing, ticks[1:] if ticks[0] == 0 else ticks)
            for _spacing, ticks in vals
        ]
        return vals


class ProjectIndexTimeAxis(pg.AxisItem):
    """Time axis converting StftIndex into timestamps
    """

    def __init__(self, *args, project, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = project

    def _to_timestamp(self, x):
        return ProjectIndex(self.project, x).to_timestamp()

    def tickStrings(self, values, scale, spacing):
        return [hhmmss(self._to_timestamp(value), dec=2) for value in values]
