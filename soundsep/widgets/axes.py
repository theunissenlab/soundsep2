import pyqtgraph as pg

from soundsep.core.models import ProjectIndex


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

    def _format_time(self, t: float):
        """Format time in seconds to form hh:mm:ss"""
        h = int(t / 3600)
        t -= h * 3600
        m = int(t / 60)
        t -= m * 60
        s = t
        return "{}:{:02d}:{:05.02f}".format(h, m, s)

    def _to_timestamp(self, x):
        return ProjectIndex(self.project, x).to_timestamp()

    def tickStrings(self, values, scale, spacing):
        return [self._format_time(self._to_timestamp(value)) for value in values]
