from functools import partial
from pathlib import Path
from collections import namedtuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets as widgets
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer

from soundsep.api import SoundsepControllerApi, SoundsepGuiApi
from soundsep.app.services import SourceService
from soundsep.core.models import ProjectIndex, StftIndex
from soundsep.gui.ui.main_window import Ui_MainWindow
from soundsep.gui.source_view import SourceView
from soundsep.gui.preview import PreviewPlot
from soundsep.gui.components.selection_box import SelectionBox


pg.setConfigOption('background', None)
pg.setConfigOption('foreground', 'k')


Roi = namedtuple("Roi", ["roi", "source"])


class SoundsepGui(widgets.QMainWindow):
    def __init__(self, api: SoundsepControllerApi):
        super().__init__()
        self.api = api
        self.gui_api = SoundsepGuiApi(self)

        self.source_views = []

        self.title = "SoundSep"
        self.init_ui()
        self.connect_events()
        self.setup_shortcuts()

    def init_ui(self):
        # Initialize main window
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.mainSplitter.setSizes([1000, 600])

        self.workspace_layout = self.ui.mainScrollArea.widget().layout()
        self.workspace_layout.setSpacing(0)
        self.workspace_layout.setContentsMargins(0, 0, 0, 0)

        self.toolbar = widgets.QToolBar()
        self.ui.toolbarDock.setWidget(self.toolbar)

        self.preview_plot_widget = PreviewPlot(self.api._app.project)
        self.ui.previewBox.layout().addWidget(self.preview_plot_widget)

        # TODO: replace this with an Icon
        self.add_source_button = widgets.QPushButton("+Source")
        self.toolbar.addWidget(self.add_source_button)
        self.add_source_button.clicked.connect(self.on_add_source)

        self.roi = None

    def connect_events(self):
        # API Events
        self.api.workspaceChanged.connect(self.on_workspace_changed)
        self.api.sourcesChanged.connect(self.on_sources_changed)

        # User events
        self.ui.actionLoad_project.triggered.connect(self.run_directory_loader)

        # Rate limited events
        # TODO dynamically adjust the timeout based on the time it takes and or event backlog?
        self._accumulated_movement = 0
        self._move_timer = QTimer(self)
        self._move_timer.timeout.connect(self._move)

    def show_status(self, message: str, duration: int=1000):
        self.statusBar().showMessage(message, duration)

    def setup_shortcuts(self):
        self.next_shortcut = widgets.QShortcut(QtGui.QKeySequence("D"), self)
        self.next_shortcut.activated.connect(self.next)
        self.prev_shortcut = widgets.QShortcut(QtGui.QKeySequence("A"), self)
        self.prev_shortcut.activated.connect(self.prev)

    def run_directory_loader(self):
        """Dialog to read in a directory of wav files and intervals """
        options = widgets.QFileDialog.Options()
        selected_file = widgets.QFileDialog.getExistingDirectory(
            self,
            "Load directory",
            ".",
            options=options
        )

        if selected_file:
            self.api.load_project(Path(selected_file))

    def next(self):
        # TODO: If we plot the full extent of the spec cache and manipulate xrange we can update
        # the plot here even as we accumulate move increments for a snappier response
        self._accumulated_movement += 100
        if not self._move_timer.isActive():
            self._move_timer.start(20)

    def prev(self):
        """Move the workspace by a fixed amount. Accumulates over 200ms windows"""
        self._accumulated_movement -= 100
        if not self._move_timer.isActive():
            self._move_timer.start(20)

    def _move(self):
        self.api.workspace_move_by(self._accumulated_movement)
        self._accumulated_movement = 0

    def on_add_source(self):
        self.api.create_blank_source()

    def on_workspace_changed(self, x: StftIndex, y: StftIndex):
        self.show_status("{:.2f}-{:.2f}".format(x.to_timestamp(), y.to_timestamp()))
        self.draw_sources()

        # TODO: it is too slow to read/write this whole thing every time we move.
        # Should do something similar to the StftCache...
        # t_arr, data = self.api.get_workspace_signal()

        # self.preview_plot_widget.waveform_plot.setData(t_arr, data[:, 0])
        # self.preview_plot_widget.setXRange(int(x.to_project_index()), int(y.to_project_index()))

    def draw_sources(self):
        x0, x1 = self.api.workspace_get_lim()
        stft_data, _stale, freqs = self.api.get_workspace_stft()
        for source_view in self.source_views:
            source_view.spectrogram.set_data(x0, x1, stft_data[:, source_view.source.channel, :], freqs)

        if np.any(_stale):
            QTimer.singleShot(200, self.draw_sources)

    def on_sources_changed(self, sources: SourceService):
        for i in reversed(range(self.workspace_layout.count())):
            widget = self.workspace_layout.itemAt(i).widget()
            if isinstance(widget, SourceView):
                widget.deleteLater()

        self.source_views = []
        for source in sources:
            source_view = SourceView(source)
            # TODO: have a class for all the sources manage these?
            source_view.spectrogram.getViewBox().dragComplete.connect(partial(self.on_drag_complete, source))
            source_view.spectrogram.getViewBox().dragInProgress.connect(partial(self.on_drag_in_progress, source))
            source_view.spectrogram.getViewBox().clicked.connect(partial(self.on_spectrogram_clicked, source))
            source_view.hover.connect(partial(self.on_spectrogram_hover, source))
            source_view.spectrogram.getViewBox().zoomEvent.connect(partial(self.on_spectrogram_zoom, source))

            self.workspace_layout.addWidget(source_view)
            self.source_views.append(source_view)

        self.draw_sources()

    def on_drag_complete(self, source):
        pass

    def on_drag_in_progress(self, source, from_, to):
        # Draw selection box
        if self.roi is None:
            self.draw_selection_box(source, from_, to)
        elif source != self.roi.source:
            self.delete_roi()
            self.draw_selection_box(source, from_, to)
        else:
            line = to - from_
            self.roi.roi.setPos([
                min(to.x(), from_.x()),
                min(to.y(), from_.y())
            ])
            self.roi.roi.setSize([
                np.abs(line.x()),
                np.abs(line.y())
            ])
            # self.api.set_selection(
            #     ProjectIndex(self.api._app.project, int(round(from_.x()))),
            #     ProjectIndex(self.api._app.project, int(round(to.x()))),
            #     from_.y(),
            #     to.y()
            # )

    def draw_selection_box(self, source, from_, to):
        source_view = self.source_views[source.index]
        self.roi = Roi(
            source=source,
            roi=SelectionBox(
                pos=from_,
                size=to - from_,
                pen=(156, 156, 100),
                rotatable=False,
                removable=False,
                # maxBounds=source_view.spectrogram.viewRange(),
            )
        )
        # self.roi.roi.sigRegionChanged.connect((self.on_selection_changed, source))
        source_view.spectrogram.addItem(self.roi.roi)

    def on_spectrogram_clicked(self, source):
        # Clear roi
        pass

    def on_spectrogram_hover(self, source, x: ProjectIndex, y: float):
        self.show_status("t={:.2f}s, freq={:.2f}Hz".format(x.to_timestamp(), y))

    def on_spectrogram_zoom(self, source, direction: int, pos):
        # TODO: there are weird artifats when using the zoom functionality in the spectrogram
        self.api.workspace_scale(direction * -50)

    # def on_selection_changed(self, source):
    #     if selection_data is None:
    #         self.api.clear_selection()
    #     else:
    #         self.api.set_selection(*selection_data)
