from functools import partial

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets as widgets
from PyQt5 import QtGui
from soundsig.signal import bandpass_filter, lowpass_filter

from soundsep.core.api import Soundsep
from soundsep.core.app import Workspace
from soundsep.core.models import ProjectIndex, Source
from soundsep.gui.components.overlays import FloatingButton
from soundsep.gui.components.selection_box import SelectionBox
from soundsep.gui.components.spectrogram_view_box import SpectrogramViewBox
from soundsep.gui.stft_view import ScrollableSpectrogram, ScrollableSpectrogramConfig
from soundsep.gui.ui.main_window import Ui_MainWindow


class PreviewPlot(pg.PlotWidget):
    """Preview of signal and ampenv data"""


class SpectrogramViewWidget(widgets.QWidget):

    def __init__(
            self,
            project,
            source: Source,
            config: ScrollableSpectrogramConfig,
        ):
        super().__init__()
        self.project = project
        self.source = source
        self._config = config
        self.init_ui()

    def init_ui(self):
        self._spectrogram_viewbox = SpectrogramViewBox()
        # self._spectrogram_viewbox.autoRange(padding=0.0)
        self.spectrogram = ScrollableSpectrogram(
            project=self.project,
            channel=self.source.channel,
            config=self._config,
            viewBox=self._spectrogram_viewbox
        )

        self.spectrogram.showAxis("left", False)
        self.spectrogram.showAxis("bottom", False)
    
        layout = widgets.QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.spectrogram)
        self.setLayout(layout)

        self.dialog = FloatingButton("▼ {}".format(self.source.name), parent=self.spectrogram)
        self.spectrogram.image.sigImageChanged.connect(self.on_image_changed)

    def on_image_changed(self):
        self._spectrogram_viewbox.autoRange(padding=0.0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.dialog.update_position()


class MainApp(widgets.QMainWindow):
    def __init__(self, workspace: Workspace):
        super().__init__()
        self.title = "SoundSep"
        self.api = Soundsep(workspace)
        self._source_views = []
        self.plugins = []
        self._roi = None  # Tuple of (source, xbounds, ybounds)

        i0, i1 = self.api.get_xrange()
        self.page_size = i1 - i0

        self.init_ui()
        self.setup_shortcuts()
        self.init_plugins()
        self.init_plugins_ui()

        # Connect signals
        self.api.xrangeChanged.connect(self.on_xrange_changed)
        self.api.sourcesChanged.connect(self.draw_sources)
        self.api.selectionChanged.connect(self.update_preview)

        # If there are no saved sources, make one
        if self.api.paths.default_sources_savefile.exists():
            self.api.load_sources(self.api.paths.default_sources_savefile)
        else:
            self._create_source()

    def init_ui(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.mainSplitter.setSizes([1000, 600])

        self.spectrogram_layout = self.ui.mainScrollArea.widget().layout()
        self.spectrogram_layout.setSpacing(0)
        self.spectrogram_layout.setContentsMargins(0, 0, 0, 0)

        pg.setConfigOption('background', None)
        pg.setConfigOption('foreground', 'k')

        self.preview_plot_widget = PreviewPlot()
        self.preview_plot_widget.plotItem.setMouseEnabled(x=False, y=False)
        self.preview_plot = self.preview_plot_widget.plot([], [])
        self.preview_plot.setPen(pg.mkPen((130, 120, 200), width=1))
        self.ui.previewBox.layout().addWidget(self.preview_plot_widget)

        self.toolbar = widgets.QToolBar()
        self.ui.toolbarDock.setWidget(self.toolbar)
        self.addSourceButtonToolbar = widgets.QPushButton("+Source")
        self.toolbar.addWidget(self.addSourceButtonToolbar)

        self.addSourceButtonToolbar.clicked.connect(self._create_source)

    def init_plugins(self):
        for P in self.api.plugins:
            self.plugins.append(P(self.api, self))

    def init_plugins_ui(self):
        for item in reversed(range(self.ui.pluginPanelToolbox.count())):
            self.ui.pluginPanelToolbox.removeItem(item)

        for plugin in self.plugins:
            for widget in plugin.plugin_toolbar_items():
                self.toolbar.addWidget(widget)

            if plugin.plugin_panel_widget():
                self.ui.pluginPanelToolbox.addItem(
                    plugin.plugin_panel_widget(),
                    type(plugin).__name__
                )

    def _create_source(self):
        """Adds a new source at the next empty channel, or channel 0"""
        create_channel = 0
        current_source_channels = [s.channel for s in self.api.get_sources()]
        for i in range(self.api.project.channels):
            if i not in current_source_channels:
                self.api.create_source("Example{}".format(i), i)
                break
        else:
            self.api.create_source("NewSource", 0)

    def show_status(self, message: str, duration: int=1000):
        self.statusBar().showMessage(message, duration)

    def on_xrange_changed(self, i0: ProjectIndex, i1: ProjectIndex):
        for source_view in self._source_views:
            source_view.spectrogram.scroll_to(i0)
            self.show_status("Moved {:.3f}s".format(i0 / i0.project.sampling_rate), 1000)

    def draw_sources(self):
        # TODO: how to preserve the ROI after adding a source?
        if self._roi:
            self._delete_roi()

        for i in reversed(range(self.spectrogram_layout.count())):
            item = self.spectrogram_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
            self._source_views = []

        config = self.api.read_config()
        for source_idx, source in enumerate(self.api.get_sources()):
            source_view = SpectrogramViewWidget(
                project=self.api.project,
                source=source,
                config=ScrollableSpectrogramConfig(
                    window_size=config["window_size"],
                    window_step=config["window_step"],
                    spectrogram_size=self.page_size,
                    cmap="turbo"
                ),
            )
            self._source_views.append(source_view)
            self.spectrogram_layout.addWidget(source_view, 1)
            source_view.spectrogram.scroll_to(self.api.get_xrange()[0])

            source_view._spectrogram_viewbox.dragComplete.connect(
                partial(self.on_drag_complete, source_idx)
            )
            source_view._spectrogram_viewbox.dragInProgress.connect(
                partial(self.on_drag_in_progress, source_idx)
            )
            source_view._spectrogram_viewbox.clicked.connect(
                partial(self.on_click, source_idx)
            )

    def update_preview(self):
        """Plot the selected bandpass signal or use the default bandpass settings
        """
        try:
            (i0, i1), (f0, f1), source = self.api.get_active_selection()
        except TypeError:
            xbounds = self.api.get_xrange()
            sig = self.api.project[xbounds[0]:xbounds[1], 0]
            filtered = bandpass_filter(
                sig,
                self.api.project.sampling_rate,
                250.0,  # TODO: Don't hardcode this!read from config?
                10000.0, # TODO: ditto
            )
            self.preview_plot.setData(np.arange(len(sig)), sig)
        else:
            f0 = max(f0, 250.0)
            f1 = min(f1, 10000)
            if f1 <= f0 or i1 - i0 < 33:
                return None
            filtered = bandpass_filter(self.api.project[i0:i1, source.channel], self.api.project.sampling_rate, f0, f1)
            self.preview_plot.setData(np.arange(len(filtered)), filtered)

    def on_drag_complete(self, source_idx: int, from_, to):
        if self._roi:
            old_source_idx = self.api.get_sources().index(self._roi[0])
            if old_source_idx != source_idx:
                self.api.clear_selection()

        selection_data = self._get_selection_data(source_idx)
        self.api.set_selection(*selection_data)

    def on_drag_in_progress(self, source_idx: int, from_, to):
        if self._roi is None:
            self._draw_selection_box(source_idx, from_, to)
        elif self.api.get_sources().index(self._roi[0]) != source_idx:
            self._delete_roi()
            self._draw_selection_box(source_idx, from_, to)
        else:
            line = to - from_
            self._roi[1].setPos([
                min(to.x(), from_.x()),
                min(to.y(), from_.y())
            ])
            self._roi[1].setSize([
                np.abs(line.x()),
                np.abs(line.y())
            ])
            selection_data = self._get_selection_data(source_idx)
            self.api.set_selection(*selection_data)

    def on_click(self, source_idx: int, at):
        self._clear_selection_box()

    def _delete_roi(self):
        if self._roi is not None:
            for source_view in self._source_views:
                try:
                    source_view.spectrogram.removeItem(self._roi[1])
                except:
                    pass
            self._roi[1].deleteLater()
            self._roi = None

    def _clear_selection_box(self):
        if self._roi is not None:
            self._delete_roi()
            self.api.clear_selection()

    def _draw_selection_box(self, source_idx: int, from_, to):
        source_view = self._source_views[source_idx]
        self._roi = (
            self.api.get_sources()[source_idx],
            SelectionBox(
                pos=from_,
                size=to - from_,
                pen=(156, 156, 100),
                rotatable=False,
                removable=False,
                maxBounds=source_view.spectrogram.image.boundingRect()
            )
        )
        self._roi[1].sigRegionChanged.connect(
            partial(self.on_selection_changed, source_idx)
        )
        source_view.spectrogram.addItem(self._roi[1])

    def _get_selection_data(self, source_idx):
        """Get selection in terms of ProjectIndex, float freqs, nad Source object"""
        if self._roi is not None:
            source_view = self._source_views[source_idx]
            xbounds, ybounds = self._roi[1].get_scaled_bounds(
                source_view.spectrogram.image,
                source_view.spectrogram.xlim(),
                source_view.spectrogram.ylim(),
            )
            xbounds = (
                ProjectIndex(self.api.project, int(xbounds[0])),
                ProjectIndex(self.api.project, int(xbounds[1])),
            )

            return xbounds, ybounds, self.api.get_sources()[source_idx]
        else:
            return None

    def on_selection_changed(self, source_idx: int):
        selection_data = self._get_selection_data(source_idx)
        if selection_data is None:
            self.api.clear_selection()
        else:
            self.api.set_selection(*selection_data)

    def page_left(self):
        i0, i1 = self.api.get_xrange()
        dx = self.page_size // 2
        self.api.set_xrange(
            i0 - dx,
            i1 - dx
        )

        # update the active selection with update coordinates
        if self._roi:
            source_idx = self.api.get_sources().index(self._roi[0])
            xbounds, ybounds, source = self._get_selection_data(source_idx)

            self.api.set_selection(
                xbounds,
                ybounds,
                source
            )

        self.update_preview()

    def page_right(self):
        i0, i1 = self.api.get_xrange()
        dx = self.page_size // 2
        self.api.set_xrange(
            i0 + self.page_size // 2,
            i1 + self.page_size // 2
        )

        # update the active selection with update coordinates
        if self._roi:
            source_idx = self.api.get_sources().index(self._roi[0])
            xbounds, ybounds, source = self._get_selection_data(source_idx)
            self.api.set_selection(
                xbounds,
                ybounds,
                source
            )

        self.update_preview()

    def setup_shortcuts(self):
        self.next_shortcut = widgets.QShortcut(QtGui.QKeySequence("D"), self)
        self.next_shortcut.activated.connect(self.page_right)

        self.next_page_shortcut = widgets.QShortcut(QtGui.QKeySequence("Shift+D"), self)
        self.next_page_shortcut.activated.connect(self.page_right)

        self.prev_shortcut = widgets.QShortcut(QtGui.QKeySequence("A"), self)
        self.prev_shortcut.activated.connect(self.page_left)

        self.prev_page_shortcut = widgets.QShortcut(QtGui.QKeySequence("Shift+A"), self)
        self.prev_page_shortcut.activated.connect(self.page_left)

