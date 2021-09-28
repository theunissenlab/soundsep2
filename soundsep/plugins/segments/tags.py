import logging
from dataclasses import dataclass
from functools import partial

import PyQt5.QtWidgets as widgets
import pandas as pd
from PyQt5 import QtGui

from soundsep.core.base_plugin import BasePlugin


logger = logging.getLogger(__name__)


class TagsPanel(widgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = widgets.QVBoxLayout()
        self.table = widgets.QTableWidget(0, 1)
        self.table.setEditTriggers(widgets.QTableWidget.NoEditTriggers)
        header = self.table.horizontalHeader()
        self.table.setHorizontalHeaderLabels([
            "TagName",
        ])
        header.setSectionResizeMode(0, widgets.QHeaderView.Stretch)
        layout.addWidget(self.table)

        tag_edit_layout = widgets.QHBoxLayout()
        self.new_tag_text = widgets.QLineEdit()
        self.add_tag_button = widgets.QPushButton("Create Tag")
        self.delete_tag_button = widgets.QPushButton("Delete Tag")
        tag_edit_layout.addWidget(self.new_tag_text)
        tag_edit_layout.addWidget(self.add_tag_button)
        tag_edit_layout.addWidget(self.delete_tag_button)

        layout.addLayout(tag_edit_layout)

        self.setLayout(layout)

    def set_data(self, tags):
        self.table.setRowCount(len(tags))
        for row, tag in enumerate(tags):
            self.table.setItem(row, 0, widgets.QTableWidgetItem(tag))


class TagPlugin(BasePlugin):
    TAG_FILENAME = "tags.csv"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.panel = TagsPanel()

        self.init_actions()
        self.connect_events()

        self._needs_saving = False

    def init_actions(self):
        self.apply_tags_action = widgets.QAction("&Tag selection", self)
        self.apply_tags_action.triggered.connect(self.on_apply_tag)

    def on_apply_tag(self, tag):
        selection = self.api.get_fine_selection()
        if selection:
            self.apply_tag(selection.x0, selection.x1, selection.source, tag)

    def on_toggle_selection_tag(self, tag: str, selection: 'List[int]', toggle: bool):
        for i in selection:
            segment = self._datastore["segments"][i]
            if toggle:
                segment.data["tags"].add(tag)
            else:
                segment.data["tags"].remove(tag)

        self.api.plugins["SegmentPlugin"].panel.set_data(self.api.plugins["SegmentPlugin"]._segmentation_datastore)
        self._needs_saving = True

    def apply_tag(self, start: 'ProjectIndex', stop: 'ProjectIndex', source: 'Source', tag: 'str'):
        to_tag = [
            segment for segment in self._datastore["segments"]
            if (
                (segment.start <= start and segment.stop >= start) or
                (segment.start <= stop and segment.stop >= stop) or
                (segment.start >= start and segment.stop <= stop)
            ) and source == segment.source
        ]
        if not len(to_tag):
            return

        logger.debug("Tagging {} segments as {} from {} to {}".format(len(to_tag), tag, start, stop))

        for segment in to_tag:
            segment.data["tags"].add(tag)

        self.api.plugins["SegmentPlugin"].panel.set_data(self.api.plugins["SegmentPlugin"]._segmentation_datastore)
        self._needs_saving = True

    def connect_events(self):
        self.api.projectLoaded.connect(self.on_project_ready)
        self.api.projectDataLoaded.connect(self.on_project_data_loaded)

        self.panel.add_tag_button.clicked.connect(self.on_add_tag)
        self.panel.new_tag_text.textChanged.connect(self.on_text_changed)
        self.panel.new_tag_text.returnPressed.connect(self.on_add_tag)
        self.panel.delete_tag_button.clicked.connect(self.on_delete_tag)

    def on_text_changed(self):
        tag_name = self.panel.new_tag_text.text()
        if not tag_name or ("tags" in self._datastore and tag_name in self._datastore["tags"]):
            self.panel.add_tag_button.setEnabled(False)
        else:
            self.panel.add_tag_button.setEnabled(True)

    def on_add_tag(self):
        tag_name = self.panel.new_tag_text.text()
        tag_name = tag_name.replace(",", "")
        if tag_name and tag_name not in self._datastore["tags"]:
            self._datastore["tags"].append(tag_name)
            self.panel.set_data(self._datastore["tags"])
            self.update_menu(self._datastore["tags"])
        self._needs_saving = True

    def on_delete_tag(self):
        selected_indexes = self.panel.table.selectedIndexes()
        rows = [index.row() for index in selected_indexes]
        to_delete = [self._datastore["tags"][i] for i in rows]

        # for each tag to delete, delete it from all segments and then
        # delete the element in the datastore...
        for tag_name in to_delete:
            for segment in self._datastore["segments"]:
                if tag_name in segment.data["tags"]:
                    segment.data["tags"].remove(tag_name)
            self._datastore["tags"].remove(tag_name)

        self.api.plugins["SegmentPlugin"].panel.set_data(self.api.plugins["SegmentPlugin"]._segmentation_datastore)
        self.panel.set_data(self._datastore["tags"])
        self.update_menu(self._datastore["tags"])
        self._needs_saving = True

    def get_tag_menu(self, menu_parent):
        """Create a menu to select a tag and a dict of actions to respond to selections
        """
        actions = {}
        for tag in self._datastore.get("tags", []):
            actions[tag] = widgets.QAction(tag, self)
            menu_parent.addAction(actions[tag])
        return menu_parent, actions

    def update_menu(self, tags):
        self.menu.clear()
        menu, actions = self.get_tag_menu(self.menu)
        for tag, action in actions.items():
            action.triggered.connect(partial(self.on_apply_tag, tag))

        # self.api.tags_changed(self, 
        # self.api.workspaceChanged.connect(self.on_workspace_changed)
        # self.api.selectionChanged.connect(self.on_selection_changed)
        # self.api.sourcesChanged.connect(self.on_sources_changed)

    @property
    def _datastore(self):
        return self.api.get_mut_datastore()

    def on_project_ready(self):
        tags_file = self.api.paths.save_dir / self.TAG_FILENAME

        if not tags_file.exists():
            self._datastore["tags"] = []
        else:
            tags_dataframe = pd.read_csv(tags_file)
            self._datastore["tags"] = list(tags_dataframe["TagName"])

    def on_project_data_loaded(self):
        self.panel.set_data(self._datastore["tags"])
        self.update_menu(self._datastore["tags"])

    def save(self):
        if "tags" in self._datastore:
            df = pd.DataFrame([{"TagName": tag} for tag in self._datastore["tags"]])
            df.to_csv(self.api.paths.save_dir / self.TAG_FILENAME)
            self._needs_saving = False

    def plugin_panel_widget(self):
        return self.panel

    def add_plugin_menu(self, menu_parent):
        self.menu = menu_parent.addMenu("&Tags")
        return self.menu

    def setup_plugin_shortcuts(self):
        self.apply_tags_action.setShortcut(QtGui.QKeySequence("T"))
