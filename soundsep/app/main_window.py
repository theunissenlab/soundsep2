import logging
from collections import namedtuple
from functools import partial

import numpy as np
from PyQt5.QtCore import QTimer, QUrl
from PyQt5 import QtWidgets as widgets
from PyQt5 import QtGui

from soundsep import settings
from soundsep.api import SignalTooShort
from soundsep.app.project_loader import ProjectLoader
from soundsep.core.models import StftIndex, ProjectIndex
from soundsep.ui.main_window import Ui_MainWindow
from soundsep.widgets.axes import ProjectIndexTimeAxis
from soundsep.widgets.box_scroll import ProjectScrollbar
from soundsep.widgets.selection_box import SelectionBox
from soundsep.widgets.source_view import SourceView, STFTViewMode


Roi = namedtuple("Roi", ["roi", "source"])


logger = logging.getLogger(__name__)


class SoundsepMainWindow(widgets.QMainWindow):

    def __init__(self, api):
        super().__init__()
        self.api = api
        self.setWindowTitle("SoundSep")
        self.source_views = []
        self.roi = None
        self._last_drag_start = None

        self.setup_actions()
        self.init_ui()
        self.connect_events()
        self.setup_shortcuts()

    def setup_actions(self):
        self.open_action = widgets.QAction("&Open Project")
        self.close_action = widgets.QAction("&Close")
        self.save_action = widgets.QAction("&Save")
        self.save_action.setToolTip("Save the current segements")
        self.clear_selection_action = widgets.QAction("&Clear Current Selection")
        self.clear_selection_action.setToolTip("Clear the currently selected data ranges")
        self.create_source_action = widgets.QAction("&Add source...")
        self.create_source_action.setToolTip("Create a new source")
        self.toggle_ampenv_action = widgets.QAction("Show &Amplitude Envelope")
        self.toggle_ampenv_action.setToolTip("Show or hide amplitude envelope. It is slow right now!")
        self.toggle_ampenv_action.setCheckable(True)

    def init_ui(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.verticalLayout_3.setSpacing(0)
        self.ui.verticalLayout_3.setContentsMargins(0, 0, 0, 0)

        # Set plot axes
        self.ui.previewPlot.setAxisItems({
            "bottom": ProjectIndexTimeAxis(project=self.api.project, orientation="bottom")
        })

        # Cleanup plugin panel
        self.ui.pluginPanelToolbox.clear()

        # Add a scrollbar
        self.scrollbar = ProjectScrollbar(self.api.project, self)
        self.scrollbar.positionChanged.connect(self.on_scrollbar_position_changed)

        scrollbarToolbar = widgets.QToolBar()
        # self.ui.leftVLayout.setSpacing(0)
        # self.ui.leftVLayout.setContentsMargins(0, 0, 0, 0)
        scrollbarToolbar.addWidget(self.scrollbar)
        self.ui.leftVLayout.addWidget(scrollbarToolbar)

        # Add to toolbar
        # TODO: replace this with an Icon
        self.add_source_button = widgets.QPushButton("+Source")
        self.ui.toolbarLayout.addWidget(self.add_source_button)
        self.add_source_button.clicked.connect(self.on_add_source)

        self.spectrogram_view_mode_button = widgets.QPushButton("SD")
        self.spectrogram_view_mode_button.setCheckable(True)
        self.spectrogram_view_mode_button.setChecked(False)
        self.ui.toolbarLayout.addWidget(self.spectrogram_view_mode_button)
        self.spectrogram_view_mode_button.clicked.connect(self.on_toggle_view_mode)

        # Add to menu
        self.ui.menuFile.addAction(self.open_action)
        self.ui.menuFile.addSeparator()
        self.ui.menuFile.addAction(self.save_action)
        self.ui.menuFile.addSeparator()
        self.ui.menuFile.addAction(self.close_action)
        self.ui.menuSources.addAction(self.create_source_action)
        self.ui.menuView.addAction(self.toggle_ampenv_action)
        self.ui.menuView.addAction(self.clear_selection_action)

        # Set screen sizes
        self.setMinimumSize(1400, 800)
        self.ui.mainSplitter.setSizes([1200, 600])
        self.ui.rightSplitter.setSizes([500, 500])

    def connect_events(self):
        # User triggered events
        self.open_action.triggered.connect(self.run_directory_loader)
        self.close_action.triggered.connect(self.close)
        self.save_action.triggered.connect(self.on_save_requested)
        self.create_source_action.triggered.connect(self.on_add_source)
        self.toggle_ampenv_action.triggered.connect(self.on_toggle_ampenv)
        self.clear_selection_action.triggered.connect(self.on_clear_selection_requested)

        self.ui.actionGithub.triggered.connect(self.on_click_help)
        self.ui.previewPlot.fineSelectionMade.connect(self.on_fine_selection)
        self.ui.previewPlot.fineSelectionCleared.connect(self.on_fine_selection_cleared)

        # API Events
        self.api.projectLoaded.connect(self.on_api_project_ready)
        self.api.workspaceChanged.connect(self.on_api_workspace_changed)
        self.api.sourcesChanged.connect(self.on_api_sources_changed)
        self.api.selectionChanged.connect(self.on_api_selection_changed)

        # Rate limited events
        self._accumulated_movement = 0
        self._move_timer = QTimer(self)
        self._move_timer.timeout.connect(self._move)

        self._accumulated_zoom = 0
        self._zoom_timer = QTimer(self)
        self._zoom_timer.timeout.connect(self._zoom)

        self._draw_sources_timer = QTimer(self)
        self._draw_sources_timer.timeout.connect(self.draw_sources)

    def setup_shortcuts(self):
        self.next_shortcut = widgets.QShortcut(QtGui.QKeySequence("D"), self)
        self.next_shortcut.activated.connect(self.on_next)
        self.next_shortcut_arrow = widgets.QShortcut(QtGui.QKeySequence("right"), self)
        self.next_shortcut_arrow.activated.connect(self.on_next)
        self.prev_shortcut = widgets.QShortcut(QtGui.QKeySequence("A"), self)
        self.prev_shortcut.activated.connect(self.on_prev)
        self.prev_shortcut_arrow = widgets.QShortcut(QtGui.QKeySequence("left"), self)
        self.prev_shortcut_arrow.activated.connect(self.on_prev)
        self.toggle_ampenv_action.setShortcut(QtGui.QKeySequence("Ctrl+A"))
        self.clear_selection_action.setShortcut(QtGui.QKeySequence("Escape"))
        self.open_action.setShortcut(QtGui.QKeySequence.Open)
        self.close_action.setShortcut(QtGui.QKeySequence("Ctrl+W"))
        self.save_action.setShortcut(QtGui.QKeySequence.Save)
        self.create_source_action.setShortcut(QtGui.QKeySequence("Ctrl+N"))

    #########################
    ### Utility functions ###
    #########################
    def show_status(self, message: str, duration: int = 1000):
        self.statusBar().showMessage(message, duration)

    def run_directory_loader(self):
        loader = ProjectLoader()
        loader.openProject.connect(self.on_request_switch_projects)
        loader.show()

    ##################
    ### Close hook ###
    ##################
    def confirm_close(self):
        """Returns True if a close action should be completed"""
        if not self.api.needs_saving():
            return True

        reply = widgets.QMessageBox.question(
            self,
            "Close confirmation",
            "Are you sure you want to quit? There are unsaved changes.",
            widgets.QMessageBox.Save | widgets.QMessageBox.Close | widgets.QMessageBox.Cancel,
            widgets.QMessageBox.Save
        )

        if reply == widgets.QMessageBox.Save:
            if self.on_save_requested():
                return True
            else:
                return False
        elif reply == widgets.QMessageBox.Cancel:
            return False
        elif reply == widgets.QMessageBox.Close:
            return True
        return False

    def closeEvent(self, event):
        if self.confirm_close():
            event.accept()
            self.api._close()
        else:
            event.ignore()

    ################
    ## Api events ##
    ################
    def on_api_project_ready(self):
        self.on_api_sources_changed()
        self.draw_sources()
        x0, x1 = self.api.workspace_get_lim()
        self.scrollbar.set_current_range(x0.to_project_index(), x1.to_project_index())

    def on_api_workspace_changed(self):
        x0, x1 = self.api.workspace_get_lim()
        self.show_status("{:.2f}-{:.2f}".format(x0.to_timestamp(), x1.to_timestamp()))
        self.draw_sources()
        self.scrollbar.set_current_range(x0.to_project_index(), x1.to_project_index())

        if self.roi:
            source_view = self.source_views[self.roi.source.index]
            self.roi.roi.maxBounds = source_view.spectrogram.get_limits_rect()

    def on_api_sources_changed(self):
        """Renders SourceView for each Source"""
        for i in reversed(range(self.ui.workspaceLayout.count())):
            widget = self.ui.workspaceLayout.itemAt(i).widget()
            if isinstance(widget, SourceView):
                widget.deleteLater()

        self.roi = None
        self.source_views = []

        for source in self.api.get_sources():
            source_view = SourceView(source)
            source_view.setMinimumHeight(self.api.config["source_view.minimum_height"])
            source_view.editSourceSignal.connect(partial(self.on_edit_source_signal, source))
            source_view.deleteSourceSignal.connect(partial(self.on_delete_source_signal, source))

            # TODO: this will be moved when config loading is improved
            # source_view.spectrogram.set_view_mode(STFTViewMode.DERIVATIVE if self.spectrogram_view_mode_button.isChecked() else STFTViewMode.NORMAL)

            # TODO: have a class for all the sources manage these?
            source_view.spectrogram.getViewBox().dragInProgress.connect(partial(self.on_source_drag_in_progress, source, finish=False))
            source_view.spectrogram.getViewBox().dragComplete.connect(partial(self.on_source_drag_in_progress, source, finish=True))
            source_view.spectrogram.getViewBox().clicked.connect(partial(self.on_source_spectrogram_clicked, source))
            source_view.hover.connect(partial(self.on_source_spectrogram_hover, source))
            source_view.spectrogram.getViewBox().zoomEvent.connect(partial(self.on_source_spectrogram_zoom, source))

            source_view.spectrogram.getViewBox().contextMenuRequested.connect(partial(self.on_context_menu, source))
            source_view.spectrogram.hideAxis("bottom")

            self.ui.workspaceLayout.addWidget(source_view)
            self.source_views.append(source_view)

        if len(self.source_views):
            self.source_views[-1].spectrogram.showAxis("bottom")

        self.draw_sources()

    def on_api_selection_changed(self):
        selection = self.api.get_selection()
        if selection:
            t, signal = self.api.get_signal(selection.x0, selection.x1)
            signal = signal[:, selection.source.channel]
            try:
                filtered, ampenv = self.api.filter_and_ampenv(signal, selection.f0, selection.f1)
            except SignalTooShort:
                logger.debug("Signal was too short for ampenv: {}".format(signal.size))
                return
            self.ui.previewPlot.waveform_plot.setData(t, filtered)
            self.ui.previewPlot.ampenv_plot.setData(t, ampenv)
        else:
            self.ui.previewPlot.waveform_plot.setData([], [])
            self.ui.previewPlot.ampenv_plot.setData([], [])

    #################
    ## User events ##
    #################
    # TODO: So many source releated events. Consider moving into its own special class
    def on_edit_source_signal(self, source):
        self.api.edit_source(source.index, source.name, source.channel)

    def on_delete_source_signal(self, source):
        self.api.delete_source(source.index)

    def on_context_menu(self, source, pos):
        self.context_menu = QtGui.QMenu()
        menu = self.context_menu.addMenu("Apply Tag")
        _, actions = self.api.plugins["TagPlugin"].get_tag_menu(menu)
        for tag, action in actions.items():
            action.triggered.connect(partial(self.api.plugins["TagPlugin"].on_apply_tag, tag))

        self.jump_to_selection_action = widgets.QAction("Jump to selection")
        self.jump_to_selection_action.triggered.connect(partial(self.api.plugins["SegmentPlugin"].jump_to_selection))
        self.jump_to_selection_action.setToolTip("Jump to this position in the segments panel")

        self.context_menu.addAction(self.jump_to_selection_action)

        self.context_menu.popup(pos)

    def on_source_drag_in_progress(self, source, from_, to, finish=False):
        if self.roi is None:
            self._last_drag_start = from_
            self.draw_roi(source, from_, to)
        elif source != self.roi.source or self._last_drag_start != from_:
            self._last_drag_start = from_
            self.delete_roi()
            self.draw_roi(source, from_, to)
        else:
            line = to - from_
            self.roi.roi.setPos([
                min(to.x(), from_.x()),
                min(to.y(), from_.y())
            ], update=False)
            self.roi.roi.setSize([
                np.abs(line.x()),
                np.abs(line.y())
            ], update=True)
            pos = self.roi.roi.pos()
            size = self.roi.roi.size()

            if finish and self.api.config["workspace.constant_refresh"]:
                self.roi.roi.sigRegionChanged.connect(partial(self.on_roi_changed, source))
            elif finish and not self.api.config["workspace.constant_refresh"]:
                self.roi.roi.sigRegionChangeFinished.connect(partial(self.on_roi_changed, source))
            if self.api.config["workspace.constant_refresh"] or finish:
                self.api.set_selection(
                    self.api.make_project_index(pos.x()),
                    self.api.make_project_index(pos.x() + size.x()),
                    pos.y(),
                    pos.y() + size.y(),
                    source,
                )

    def on_source_spectrogram_clicked(self, source):
        self.api.clear_selection()
        if self.roi:
            self.delete_roi()

    def on_source_spectrogram_hover(self, source, x: ProjectIndex, y: float):
        self.show_status("t={:.2f}s, freq={:.2f}Hz, {} ch{}".format(
            x.to_timestamp(),
            y,
            *self.api.project.to_block_index(x).block.get_channel_info(source.channel)
        ))

    def on_source_spectrogram_zoom(self, source, direction: int, pos):
        """Buffer zoom events"""
        self._accumulated_zoom += direction
        if not self._zoom_timer.isActive():
            self._zoom_timer.start(10)

    ###############################
    ## Re-rendering source views ##
    ###############################
    def draw_sources(self):
        """Draw the current spectrogram position on all source views"""
        self._draw_sources_timer.stop()
        x0, x1 = self.api.workspace_get_lim()
        t, stft_data, _stale, freqs = self.api.read_stft(x0, x1)

        any_stale = np.any(_stale)

        for source_view in self.source_views:
            source_view.spectrogram.set_data(x0, x1, t, stft_data[:, source_view.source.channel], freqs)

        if self.toggle_ampenv_action.isChecked() and not any_stale:
            summed_over_freqs = np.sum(stft_data, axis=2)
            max_value = np.max(summed_over_freqs)
            if max_value != 0:
                # Only draw overlaps if spectrograms have not loaded yet
                for source_view in self.source_views:
                    source_view.spectrogram.overlay(
                        t,
                        summed_over_freqs[:, source_view.source.channel] / max_value
                    )
        else:
            for source_view in self.source_views:
                source_view.spectrogram.clear_overlay()

        if np.any(_stale):
            self._draw_sources_timer.start(200)

    ##########################
    ### Drawing selections ###
    ##########################
    def draw_roi(self, source, from_, to):
        source_view = self.source_views[source.index]

        disp = to - from_

        self.roi = Roi(
            source=source,
            roi=SelectionBox(
                pos=(min(to.x(), from_.x()), min(to.y(), from_.y())),
                size=(np.abs(disp.x()), np.abs(disp.y())),
                pen=(156, 156, 100),
                rotatable=False,
                removable=False,
                maxBounds=source_view.spectrogram.get_limits_rect()
            )
        )
        source_view.spectrogram.addItem(self.roi.roi)

    def delete_roi(self):
        self.source_views[self.roi.source.index].spectrogram.removeItem(self.roi.roi)
        self.roi = None

    def on_roi_changed(self, source):
        """Handles a dragged change to the rectangular selection ROI"""
        if not self.roi:
            self.api.clear_selection()
        else:
            pos = self.roi.roi.pos()
            size = self.roi.roi.size()
            self.api.set_selection(
                self.api.make_project_index(pos.x()),
                self.api.make_project_index(pos.x() + size.x()),
                pos.y(),
                pos.y() + size.y(),
                source,
            )

    def on_fine_selection(self, x0: float, x1: float):
        if self.api.get_selection() is not None:
            self.api.set_fine_selection(
                self.api.make_project_index(x0),
                self.api.make_project_index(x1)
            )

    def on_fine_selection_cleared(self):
        self.api.clear_fine_selection()

    ##########################
    ### Navigation / Other ###
    ##########################
    def on_request_switch_projects(self, new_project_dir: 'pathlib.Path'):
        if self.confirm_close():
            self.api.switch_project(new_project_dir)

    def on_scrollbar_position_changed(self, x: float, y: float):
        stft_index = self.api.convert_project_index_to_stft_index(
            ProjectIndex(self.api.project, int(x))
        )
        self.api.workspace_move_to(stft_index)

    def on_next(self):
        self._accumulated_movement += 1
        if not self._move_timer.isActive():
            self._move_timer.start(10)

    def on_prev(self):
        self._accumulated_movement -= 1
        if not self._move_timer.isActive():
            self._move_timer.start(10)

    def _move(self):
        x0, x1 = self.api.workspace_get_lim()
        dx = x1 - x0
        page = dx // 5
        self.api.workspace_move_by(page * self._accumulated_movement)
        self._accumulated_movement = 0
        self._move_timer.stop()

    def _zoom(self):
        if self._accumulated_zoom == 0:
            self._zoom_timer.stop()
            return

        x0, x1 = self.api.workspace_get_lim()
        if self._accumulated_zoom > 0:
            # Zooming in to 4/5 of the current workspace
            scale = 1 + (x1 - x0) // 5
        else:
            # Zooming out to 5/4 of the current workspace
            # 1 is added for when the workspace size is == 1, you can still zoom out
            scale = 1 + (x1 - x0) // 4

        self.api.workspace_scale(np.sign(self._accumulated_zoom) * -1 * scale)
        self._accumulated_zoom = 0
        self._zoom_timer.stop()

    def on_add_source(self):
        self.api.create_blank_source()

    def on_save_requested(self) -> bool:
        """Attempt save. Return True if successful"""
        result = self.api.save()
        if result:
            self.show_status("Save successful", 2000)
        else:
            widgets.QMessageBox.error("Save failed. See logs.")

    def on_toggle_view_mode(self):
        if self.spectrogram_view_mode_button.isChecked():
            for source_view in self.source_views:
                source_view.spectrogram.set_view_mode(STFTViewMode.DERIVATIVE)
        else:
            for source_view in self.source_views:
                source_view.spectrogram.set_view_mode(STFTViewMode.NORMAL)

    def on_toggle_ampenv(self):
        self.draw_sources()

    def on_clear_selection_requested(self):
        self.api.clear_selection()

    def on_click_help(self):
        url = QUrl(settings.GITHUB_URL)
        if not QtGui.QDesktopServices.openUrl(url):
            QtGui.QMessageBox.warning(self, "", "Could not open link to {}".format(settings.GITHUB_URL))

    def attach_plugin(self, plugin: 'soundsep.core.base_plugin.BasePlugin'):
        """Put plugin widgets in their places

        Plugins should implement these methods to draw in predefined locations
            setup_plugin_shortcuts()
            plugin_toolbar_items()
            plugin_panel_widget()
            add_plugin_menu(mainMenu)
        """
        plugin.setup_plugin_shortcuts()
        for w in plugin.plugin_toolbar_items():
            self.ui.toolbarLayout.addWidget(w)

        panel = plugin.plugin_panel_widget()
        if panel:
            self.ui.pluginPanelToolbox.addTab(panel, panel.__class__.__name__)

        plugin.add_plugin_menu(self.ui.menuPlugins)
