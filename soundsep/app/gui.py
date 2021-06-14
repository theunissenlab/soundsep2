from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets as widgets
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer

from soundsep.api import SoundsepControllerApi, SoundsepGuiApi
from soundsep.app.services import SourceService
from soundsep.core.models import StftIndex
from soundsep.gui.ui.main_window import Ui_MainWindow
from soundsep.gui.source_view import SourceView


pg.setConfigOption('background', None)
pg.setConfigOption('foreground', 'k')


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

        # self.preview_plot_widget = PreviewPlot()
        # self.preview_plot_widget.plotItem.setMouseEnabled(x=False, y=False)
        # self.preview_plot = self.preview_plot_widget.plot([], [])
        # self.preview_plot.setPen(pg.mkPen((130, 120, 200), width=1))
        # self.ui.previewBox.layout().addWidget(self.preview_plot_widget)

        # TODO: replace this with an Icon
        self.add_source_button = widgets.QPushButton("+Source")
        self.toolbar.addWidget(self.add_source_button)
        self.add_source_button.clicked.connect(self.on_add_source)

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
            self.workspace_layout.addWidget(source_view)
            self.source_views.append(source_view)

        self.draw_sources()
