from functools import partial
from pathlib import Path
from collections import namedtuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets as widgets
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer, QPointF

from soundsep.api import SoundsepControllerApi, SoundsepGuiApi
from soundsep.app.services import SourceService
from soundsep.core.models import ProjectIndex, StftIndex
from soundsep.gui.ui.main_window import Ui_MainWindow
from soundsep.gui.source_view import SourceView, STFTViewMode
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
        self.setup_actions()
        self.connect_events()
        self.setup_shortcuts()

    def init_ui(self):
        # Initialize main window
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

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

        self.spectrogram_view_mode_button = widgets.QPushButton("SD")
        self.spectrogram_view_mode_button.setCheckable(True)
        self.spectrogram_view_mode_button.setChecked(False)
        self.toolbar.addWidget(self.spectrogram_view_mode_button)
        self.spectrogram_view_mode_button.clicked.connect(self.on_toggle_view_mode)

        self.setMinimumSize(1000, 500)
        self.ui.mainSplitter.setSizes([1000, 600])

        self.roi = None

    def connect_events(self):
        # API Events
        self.api.workspaceChanged.connect(self.on_workspace_changed)
        self.api.sourcesChanged.connect(self.on_sources_changed)
        self.api.selectionChanged.connect(self.on_selection_changed)

        # Rate limited events
        # TODO dynamically adjust the timeout based on the time it takes and or event backlog?
        self._accumulated_movement = 0
        self._move_timer = QTimer(self)
        self._move_timer.timeout.connect(self._move)

        self._accumulated_zoom = 0
        self._zoom_timer = QTimer(self)
        self._zoom_timer.timeout.connect(self._zoom)

    def show_status(self, message: str, duration: int=1000):
        self.statusBar().showMessage(message, duration)

    def setup_actions(self):
        self.open_action = widgets.QAction("&Open Project")
        self.open_action.triggered.connect(self.run_directory_loader)

        self.import_action = widgets.QAction("&Import WAV files...")

        self.close_action = widgets.QAction("&Close")
        self.close_action.triggered.connect(self.close)

        self.save_action = widgets.QAction("&Save")
        self.save_action.setToolTip("Save the current segements")
        self.save_action.triggered.connect(self.on_save)

        self.create_source_action = widgets.QAction("&Add source...")
        self.create_source_action.setToolTip("Create a new source")
        self.create_source_action.triggered.connect(self.on_add_source)

        # self.save_as_action = widgets.QAction("Save &As")
        # self.save_as_action.setToolTip("Save the current segments")
        # self.save_as_action.triggered.connect(self.on_save_as)

        # User events
        self.ui.menuFile.addAction(self.open_action)
        self.ui.menuFile.addAction(self.import_action)
        self.ui.menuFile.addSeparator()
        self.ui.menuFile.addAction(self.save_action)
        # self.ui.menuFile.addAction(self.save_as_action)
        self.ui.menuFile.addSeparator()
        self.ui.menuFile.addAction(self.close_action)

        self.ui.menuSources.addAction(self.create_source_action)

    def setup_shortcuts(self):
        self.next_shortcut = widgets.QShortcut(QtGui.QKeySequence("D"), self)
        self.next_shortcut.activated.connect(self.next)
        self.next_shortcut_arrow = widgets.QShortcut(QtGui.QKeySequence("right"), self)
        self.next_shortcut_arrow.activated.connect(self.next)
        self.prev_shortcut = widgets.QShortcut(QtGui.QKeySequence("A"), self)
        self.prev_shortcut.activated.connect(self.prev)
        self.prev_shortcut_arrow = widgets.QShortcut(QtGui.QKeySequence("left"), self)
        self.prev_shortcut_arrow.activated.connect(self.prev)

        self.open_action.setShortcut(QtGui.QKeySequence.Open)
        self.close_action.setShortcut(QtGui.QKeySequence("Ctrl+W"))
        self.save_action.setShortcut(QtGui.QKeySequence.Save)
        self.create_source_action.setShortcut(QtGui.QKeySequence("Ctrl+N"))
        # self.save_as_action.setShortcut(QtGui.QKeySequence.SaveAs)

    def closeEvent(self, event):
        if not self.api.check_if_sources_need_saving():
            event.accept()
            return

        reply = widgets.QMessageBox.question(
            self,
            "Close confirmation",
            "Are you sure you want to quit? There are unsaved changes.",
            widgets.QMessageBox.Save | widgets.QMessageBox.Close | widgets.QMessageBox.Cancel,
            widgets.QMessageBox.Save
        )
        if reply == widgets.QMessageBox.Save:
            if self.on_save():
                event.accept()
            else:
                event.ignore()
            return
        elif reply == widgets.QMessageBox.Cancel:
            event.ignore()
            return
        elif reply == widgets.QMessageBox.Close:
            event.accept()
            return

        event.ignore()

    def on_save(self):
        """Attempt save. Returns True if successful"""
        return self.api.save_sources()

    # def on_save_as(self):
    #     self.api.save_sources()

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
        self._accumulated_movement += 1
        if not self._move_timer.isActive():
            self._move_timer.start(1)

    def prev(self):
        """Move the workspace by a fixed amount. Accumulates over 200ms windows"""
        self._accumulated_movement -= 1
        if not self._move_timer.isActive():
            self._move_timer.start(1)

    def _move(self):
        x0, x1 = self.api.workspace_get_lim()
        dx = x1 - x0
        page = dx // 5
        self.api.workspace_move_by(page * self._accumulated_movement)
        self._accumulated_movement = 0
        self._move_timer.stop()

    def on_toggle_view_mode(self):
        if self.spectrogram_view_mode_button.isChecked():
            for source_view in self.source_views:
                source_view.spectrogram.set_view_mode(STFTViewMode.DERIVATIVE)
        else:
            for source_view in self.source_views:
                source_view.spectrogram.set_view_mode(STFTViewMode.NORMAL)

    def on_add_source(self):
        self.api.create_blank_source()

    def on_selection_changed(self):
        # TODO: this function is called very ferquently when the box is dragged around
        # get_signal() is already cached but filter_and_ampevn can/should be cached as
        # well. This could be done as an overall ampenv service refactor and/or
        # having the amplitude envelope be computed in a background thread.

        # Update the preview plot
        selection = self.api.get_selection()
        if selection:
            t, signal = self.api.get_signal(selection.x0, selection.x1)
            signal = signal[:, selection.source.channel]

            filtered, ampenv = self.api.filter_and_ampenv(signal, selection.f0, selection.f1)

            # filtered = filtered[:, selection.source.channel]
            self.preview_plot_widget.waveform_plot.setData(t, filtered)
            self.preview_plot_widget.ampenv_plot.setData(t, ampenv)

    # TODO: Some handlers "on_X()" refer to handlers of api events,
    # while others respond to internal user events and interactions. Should
    # I change the naming conventions?
    def on_workspace_changed(self, x: StftIndex, y: StftIndex):
        self.show_status("{:.2f}-{:.2f}".format(x.to_timestamp(), y.to_timestamp()))
        self.draw_sources()

        if self.roi:
            source_view = self.source_views[self.roi.source.index]
            self.roi.roi.maxBounds = source_view.spectrogram.get_limits_rect()

        # TODO: it is too slow to read the whole signal whole thing every time we move.
        # Should do something similar to the StftCache...
        # Also TODO: have a concept of source focus so for displaying information
        # t_arr, data = self.api.get_workspace_signal()
        # self.preview_plot_widget.waveform_plot.setData(t_arr, data[:, 0])
        # self.preview_plot_widget.setXRange(int(x.to_project_index()), int(y.to_project_index()))

    def draw_sources(self):
        x0, x1 = self.api.workspace_get_lim()
        stft_data, _stale, freqs = self.api.get_workspace_stft()
        for source_view in self.source_views:
            source_view.spectrogram.set_data(x0, x1, stft_data[:, source_view.source.channel, :], freqs)

        if np.any(_stale):
            QTimer.singleShot(100, self.draw_sources)

    def on_sources_changed(self, sources: SourceService):
        for i in reversed(range(self.workspace_layout.count())):
            widget = self.workspace_layout.itemAt(i).widget()
            if isinstance(widget, SourceView):
                widget.deleteLater()
        self.roi = None

        self.source_views = []
        for source in sources:
            source_view = SourceView(source)
            # for ch in range(self.api.get_current_project().channels):
            source_view.editSourceSignal.connect(partial(self.on_edit_source_signal, source))
            source_view.deleteSourceSignal.connect(partial(self.on_delete_source_signal, source))

            source_view.spectrogram.set_view_mode(STFTViewMode.DERIVATIVE if self.spectrogram_view_mode_button.isChecked() else STFTViewMode.NORMAL)

            # TODO: have a class for all the sources manage these?
            source_view.spectrogram.getViewBox().dragComplete.connect(partial(self.on_drag_complete, source))
            source_view.spectrogram.getViewBox().dragInProgress.connect(partial(self.on_drag_in_progress, source))
            source_view.spectrogram.getViewBox().clicked.connect(partial(self.on_spectrogram_clicked, source))
            source_view.hover.connect(partial(self.on_spectrogram_hover, source))
            source_view.spectrogram.getViewBox().zoomEvent.connect(partial(self.on_spectrogram_zoom, source))

            self.workspace_layout.addWidget(source_view)
            self.source_views.append(source_view)

        self.draw_sources()

    def on_delete_source_signal(self, source):
        self.api.delete_source(source.index)

    def on_edit_source_signal(self, source):
        self.api.edit_source(source.index, source.name, source.channel)

    def on_drag_complete(self, source):
        pass

    def on_drag_in_progress(self, source, from_, to):
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
            pos = self.roi.roi.pos()
            size = self.roi.roi.size()
            self.api.set_selection(
                self.api.make_project_index(pos.x()),
                self.api.make_project_index(pos.x() + size.x()),
                pos.y(),
                pos.y() + size.y(),
                source,
            )

    def delete_roi(self):
        self.source_views[self.roi.source.index].spectrogram.removeItem(self.roi.roi)
        self.roi = None

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
                maxBounds=source_view.spectrogram.get_limits_rect()
            )
        )
        self.roi.roi.sigRegionChanged.connect(partial(self.on_roi_changed, source))
        source_view.spectrogram.addItem(self.roi.roi)

    def on_spectrogram_clicked(self, source):
        self.api.clear_selection()
        if self.roi:
            self.delete_roi()

    def on_spectrogram_hover(self, source, x: ProjectIndex, y: float):
        self.show_status("t={:.2f}s, freq={:.2f}Hz, {} ch{}".format(
            x.to_timestamp(),
            y,
            *self.api.get_current_project().to_block_index(x).block.get_channel_info(source.channel)
        ))
        # TODO: show a indicator on all spectrograms of cursor position

    def on_spectrogram_zoom(self, source, direction: int, pos):
        """Buffer zoom events"""
        self._accumulated_zoom += direction

        if not self._zoom_timer.isActive():
            self._zoom_timer.start(10)

    def _zoom(self):
        if self._accumulated_zoom == 0:
            return

        x0, x1 = self.api.workspace_get_lim()
        if self._accumulated_zoom > 0:
            # Zooming in to 2/3 of the current workspace
            scale = 1 + (x1 - x0) // 3
        else:
            # Zooming out to 3/2 of the current workspace
            # 1 is added for when the workspace size is == 1, you can still zoom out
            scale = 1 + (x1 - x0) // 2

        self.api.workspace_scale(np.sign(self._accumulated_zoom) * -1 * scale)
        self._accumulated_zoom = 0
        self._zoom_timer.stop()

    def on_roi_changed(self, source):
        """Handles a dragged change to the rectangular selection ROI"""
        if not self.roi:
            self.api.clear_selection()
        else:
            source_view = self.source_views[source.index]
            pos = self.roi.roi.pos()
            size = self.roi.roi.size()
            self.api.set_selection(
                self.api.make_project_index(pos.x()),
                self.api.make_project_index(pos.x() + size.x()),
                pos.y(),
                pos.y() + size.y(),
                source,
            )
