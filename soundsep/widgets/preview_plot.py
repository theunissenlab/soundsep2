import pyqtgraph as pg
import numpy as np
from PyQt6 import QtWidgets as widgets
from PyQt6.QtCore import pyqtSignal, Qt, QTimer, QEvent, QPointF
from PyQt6.QtGui import QTransform
from scipy.signal import spectrogram

from soundsep.core.models import ProjectIndex
from .axes import ProjectIndexTimeAxis, FrequencyAxis
from .mouse_events_view_box import MouseEventsViewBox



class PreviewPlot(pg.PlotWidget):
    """
    """
    fineSelectionCleared = pyqtSignal()
    fineSelectionMade = pyqtSignal(float, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, viewBox=MouseEventsViewBox(), **kwargs)
        self.region_select = None
        self._advanced_mode = False

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


        # Secondary Y-axis for advanced ampenv (different scale)
        self.advanced_ampenv_viewbox = pg.ViewBox()
        self.plotItem.scene().addItem(self.advanced_ampenv_viewbox)
        self.plotItem.getAxis('right').linkToView(self.advanced_ampenv_viewbox)
        self.advanced_ampenv_viewbox.setXLink(self.plotItem)
        self.plotItem.showAxis('right')
        self.plotItem.getAxis('right').setLabel('Ampenv', color='#20c820')

        self.advanced_ampenv_plot = pg.PlotCurveItem()
        self.advanced_ampenv_plot.setPen(pg.mkPen((20, 200, 20), width=3))
        self.advanced_ampenv_viewbox.addItem(self.advanced_ampenv_plot)


        # Hide right axis by default (only show in advanced mode)
        self.plotItem.getAxis('right').hide()
        self.advanced_ampenv_plot.hide()

        self.waveform_plot.sigPlotChanged.connect(self.on_plot_change)
        self.ampenv_plot.sigPlotChanged.connect(self.on_plot_change)
        self.advanced_ampenv_plot.sigPlotChanged.connect(self._update_advanced_view)
        self.getViewBox().dragInProgress.connect(self.on_drag)
        self.getViewBox().clicked.connect(self.on_click)
        self.getViewBox().zoomEvent.connect(self.on_zoom_event)

        # Update advanced viewbox geometry when main view changes
        self.plotItem.vb.sigResized.connect(self._update_advanced_viewbox_geometry)

    def _update_advanced_viewbox_geometry(self):
        """Keep the advanced viewbox aligned with the main plot"""
        self.advanced_ampenv_viewbox.setGeometry(self.plotItem.vb.sceneBoundingRect())

    def _update_advanced_view(self):
        """Update the Y range for the advanced ampenv viewbox"""
        xdata, ydata = self.advanced_ampenv_plot.getData()
        if xdata is not None and len(xdata) > 0:
            ymax = np.max(np.abs(ydata)) if len(ydata) > 0 else 1
            ymin = np.min(np.abs(ydata)) if len(ydata) > 0 else 0
            # Clip to float32-safe range to avoid overflow warnings
            ymax = np.clip(ymax, 1e-6, 1e30)
            ymin = np.clip(ymin, 0, 1e30)
            self.advanced_ampenv_viewbox.setYRange(max(1e-6, ymin*0.9), ymax * 1.1, padding=0)

    def set_advanced_mode(self, enabled: bool):
        """Toggle between basic and advanced display modes"""
        self._advanced_mode = enabled
        if enabled:
            # Show advanced ampenv on right axis, hide basic ampenv
            self.ampenv_plot.hide()
            self.advanced_ampenv_plot.show()
            self.plotItem.getAxis('right').show()
        else:
            # Show basic ampenv, hide advanced
            self.ampenv_plot.show()
            self.advanced_ampenv_plot.hide()
            self.plotItem.getAxis('right').hide()
            self.advanced_ampenv_plot.setData([], [])

    def set_advanced_ampenv_data(self, t, ampenv):
        """Set the data for the advanced ampenv plot"""
        self.advanced_ampenv_plot.setData(t, ampenv)

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
            # Clip to float32-safe range to avoid overflow warnings
            ymax = np.clip(ymax, 1e-30, 1e30)
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


class AdvancedPreviewWidget(widgets.QWidget):
    """Advanced detection preview with spectrogram and ampenv plots"""

    thresholdChanged = pyqtSignal(float)
    signalBandChanged = pyqtSignal(float, float)  # low, high
    noiseBandChanged = pyqtSignal(float, float)   # low, high
    parametersChanged = pyqtSignal()  # Generic signal for any parameter change

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signal_data = None
        self._sampling_rate = None
        self._time_axis = None
        self._suppress_signals = False
        self._init_ui()

    def _init_ui(self):
        layout = widgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Spectrogram plot
        self.spec_plot = pg.PlotWidget()
        self.spec_plot.setMouseEnabled(x=False, y=False)
        self.spec_plot.setMenuEnabled(False)
        self.spec_plot.hideButtons()
        self.spec_plot.setLabel('left', 'Frequency', units='Hz')
        self.spec_plot.getAxis('bottom').setStyle(showValues=False)  # Hide x-axis labels (shared with ampenv)

        self.spec_image = pg.ImageItem()
        self.spec_plot.addItem(self.spec_image)

        # Signal band selector (green) - horizontal orientation for vertical selection
        self.signal_band_region = pg.LinearRegionItem(
            values=[2000, 10000],
            orientation='horizontal',
            brush=pg.mkBrush(0, 255, 0, 40),
            pen=pg.mkPen('g', width=2),
            movable=True
        )
        self.signal_band_region.setZValue(10)
        self.spec_plot.addItem(self.signal_band_region)

        # Noise band selector (red)
        self.noise_band_region = pg.LinearRegionItem(
            values=[500, 1500],
            orientation='horizontal',
            brush=pg.mkBrush(255, 0, 0, 40),
            pen=pg.mkPen('r', width=2),
            movable=True
        )
        self.noise_band_region.setZValue(10)
        self.spec_plot.addItem(self.noise_band_region)

        # Connect band region signals
        self.signal_band_region.sigRegionChanged.connect(self._on_signal_band_changed)
        self.noise_band_region.sigRegionChanged.connect(self._on_noise_band_changed)

        # Ampenv plot with log Y-axis
        self.ampenv_plot = pg.PlotWidget()
        self.ampenv_plot.setMouseEnabled(x=False, y=False)
        self.ampenv_plot.setMenuEnabled(False)
        self.ampenv_plot.hideButtons()
        self.ampenv_plot.setLogMode(x=False, y=True)
        self.ampenv_plot.setLabel('left', 'Amplitude (log)')
        self.ampenv_plot.setLabel('bottom', 'Time', units='s')
        self.ampenv_plot.enableAutoRange(axis='y', enable=False)

        # self.ampenv_curve = pg.PlotCurveItem()
        # self.ampenv_curve.setPen(pg.mkPen((20, 200, 20), width=2))
        # self.ampenv_plot.addItem(self.ampenv_curve)
        self.ampenv_curve = self.ampenv_plot.plot([], [], pen=pg.mkPen((20, 200, 20), width=2))


        # Threshold line (draggable)
        self.threshold_line = pg.InfiniteLine(
            pos=0.1,
            angle=0,
            movable=True,
            pen=pg.mkPen((255, 200, 0), width=3)
        )
        self.threshold_line.setCursor(Qt.CursorShape.SplitVCursor)
        self.threshold_line.setBounds([-12, 12])  # y from 1e-12 to 1e12

        # self.threshold_line.setBounds([0, None])
        self.threshold_line.sigDragged.connect(self._on_threshold_dragged)
        self.ampenv_plot.addItem(self.threshold_line)

        # Double-click on ampenv plot sets threshold
        self.ampenv_plot.viewport().installEventFilter(self)

        # Link x-axes so they stay aligned
        self.ampenv_plot.setXLink(self.spec_plot)

        # Add plots to layout (2:1 ratio for spectrogram:ampenv)
        layout.addWidget(self.spec_plot, 2)
        layout.addWidget(self.ampenv_plot, 1)

        self.spec_plot.setMinimumHeight(200)
        self.ampenv_plot.setMinimumHeight(120)

        self.setLayout(layout)

    def eventFilter(self, obj, event):
        if obj is self.ampenv_plot.viewport() and event.type() == QEvent.Type.MouseButtonDblClick:
            pos = event.position()
            scene_pos = self.ampenv_plot.mapToScene(int(pos.x()), int(pos.y()))
            vb = self.ampenv_plot.plotItem.vb
            view_pos = vb.mapSceneToView(scene_pos)
            # Y is already in log10 space (because setLogMode y=True maps data to log10)
            self.threshold_line.setValue(view_pos.y())
            self.thresholdChanged.emit(self.get_threshold())
            return True
        return super().eventFilter(obj, event)

    def _on_signal_band_changed(self):
        if self._suppress_signals:
            return
        low, high = self.signal_band_region.getRegion()
        self.signalBandChanged.emit(low, high)
        self.parametersChanged.emit()

    def _on_noise_band_changed(self):
        if self._suppress_signals:
            return
        low, high = self.noise_band_region.getRegion()
        self.noiseBandChanged.emit(low, high)
        self.parametersChanged.emit()

    def _on_threshold_dragged(self):
        self.thresholdChanged.emit(self.threshold_line.value())

    def set_signal_band(self, low: float, high: float):
        """Set the signal band region without emitting signals"""
        self._suppress_signals = True
        self.signal_band_region.setRegion([low, high])
        self._suppress_signals = False

    def set_noise_band(self, low: float, high: float):
        """Set the noise band region without emitting signals"""
        self._suppress_signals = True
        self.noise_band_region.setRegion([low, high])
        self._suppress_signals = False

    def get_signal_band(self):
        """Get the current signal band frequencies"""
        return self.signal_band_region.getRegion()

    def get_noise_band(self):
        """Get the current noise band frequencies"""
        return self.noise_band_region.getRegion()

    def set_threshold(self, value: float):
        value = max(value, 1e-12)
        self.threshold_line.setValue(np.log10(value))

    def get_threshold(self):
        return 10 ** self.threshold_line.value()

    def set_spectrogram_data(self, signal: np.ndarray, fs: int, t_offset: float = 0):
        """Compute and display spectrogram for the given signal"""
        self._signal_data = signal
        self._sampling_rate = fs

        # Compute spectrogram
        nperseg = min(256, len(signal) // 4) if len(signal) > 256 else len(signal) // 2
        if nperseg < 4:
            return

        f, t, Sxx = spectrogram(signal, fs, nperseg=nperseg, noverlap=nperseg // 2)

        # Convert to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Store time axis with offset
        self._time_axis = t + t_offset

        # Set image data (transpose so frequency is Y-axis)
        self.spec_image.setImage(Sxx_db.T)

        # Set transform to map image coordinates to time/frequency
        # Image shape is (n_times, n_freqs) after transpose
        t_min, t_max = self._time_axis[0], self._time_axis[-1]
        f_min, f_max = f[0], f[-1]

        # Calculate scale factors
        t_scale = (t_max - t_min) / Sxx_db.shape[1] if Sxx_db.shape[1] > 1 else 1
        f_scale = (f_max - f_min) / Sxx_db.shape[0] if Sxx_db.shape[0] > 1 else 1

        transform = QTransform()
        transform.translate(t_min, f_min)
        transform.scale(t_scale, f_scale)
        self.spec_image.setTransform(transform)

        # Set view range (0-20kHz for frequency, full time range)
        self.spec_plot.setXRange(t_min, t_max, padding=0)
        self.spec_plot.setYRange(0, min(20000, fs / 2), padding=0)

        # Constrain band selectors to valid frequency range
        max_freq = float(min(20000, fs / 2))
        self.signal_band_region.setBounds([0.0, max_freq])
        self.noise_band_region.setBounds([0.0, max_freq])

    def set_ampenv_data(self, t: np.ndarray, ampenv: np.ndarray):
        x = np.array([xx.to_timestamp() for xx in t], dtype=float)

        y = np.asarray(ampenv, dtype=float)
        y = np.clip(y, 1e-12, None)  # log axis needs > 0

        self.ampenv_curve.setData(x, y)

        if y.size:
            ymin = float(y.min())
            ymax = float(y.max())

            log_min = np.log10(ymin)
            log_max = np.log10(ymax)

            # Clip to float32-safe range to avoid overflow warnings
            log_min = np.clip(log_min, -30, 30)
            log_max = np.clip(log_max, -30, 30)

            # pad in log units
            span = max(1e-6, log_max - log_min)
            pad = 0.05 * span  # 5% headroom in log space

            self.ampenv_plot.setYRange(log_min - pad, log_max + pad, padding=0)
            self.ampenv_plot.setXRange(x[0], x[-1], padding=0)

    def clear(self):
        """Clear all data from the widget"""
        self.spec_image.clear()
        self.ampenv_curve.setData([], [])
        self._signal_data = None
        self._sampling_rate = None
        self._time_axis = None
