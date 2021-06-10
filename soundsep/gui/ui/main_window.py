# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'soundsep/gui/ui/main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1643, 1077)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.mainSplitter = QtWidgets.QSplitter(self.centralwidget)
        self.mainSplitter.setOrientation(QtCore.Qt.Horizontal)
        self.mainSplitter.setObjectName("mainSplitter")
        self.mainScrollArea = QtWidgets.QScrollArea(self.mainSplitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mainScrollArea.sizePolicy().hasHeightForWidth())
        self.mainScrollArea.setSizePolicy(sizePolicy)
        self.mainScrollArea.setWidgetResizable(True)
        self.mainScrollArea.setObjectName("mainScrollArea")
        self.mainScrollAreaContents = QtWidgets.QWidget()
        self.mainScrollAreaContents.setGeometry(QtCore.QRect(0, 0, 1245, 974))
        self.mainScrollAreaContents.setObjectName("mainScrollAreaContents")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.mainScrollAreaContents)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.mainScrollArea.setWidget(self.mainScrollAreaContents)
        self.rightWidget = QtWidgets.QWidget(self.mainSplitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rightWidget.sizePolicy().hasHeightForWidth())
        self.rightWidget.setSizePolicy(sizePolicy)
        self.rightWidget.setObjectName("rightWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.rightWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pluginPanelToolbox = QtWidgets.QToolBox(self.rightWidget)
        self.pluginPanelToolbox.setObjectName("pluginPanelToolbox")
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setGeometry(QtCore.QRect(0, 0, 356, 881))
        self.page_4.setObjectName("page_4")
        self.pluginPanelToolbox.addItem(self.page_4, "")
        self.verticalLayout_2.addWidget(self.pluginPanelToolbox)
        self.previewBox = QtWidgets.QGroupBox(self.rightWidget)
        self.previewBox.setObjectName("previewBox")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.previewBox)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout_2.addWidget(self.previewBox)
        self.verticalLayout.addWidget(self.mainSplitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1643, 20))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuSources = QtWidgets.QMenu(self.menubar)
        self.menuSources.setObjectName("menuSources")
        self.menuPlugins = QtWidgets.QMenu(self.menubar)
        self.menuPlugins.setObjectName("menuPlugins")
        self.menuSimpleDetection = QtWidgets.QMenu(self.menuPlugins)
        self.menuSimpleDetection.setObjectName("menuSimpleDetection")
        self.menuLabeling = QtWidgets.QMenu(self.menuPlugins)
        self.menuLabeling.setObjectName("menuLabeling")
        self.menuTemplateMatching = QtWidgets.QMenu(self.menuPlugins)
        self.menuTemplateMatching.setObjectName("menuTemplateMatching")
        self.menuRavenExport = QtWidgets.QMenu(self.menuPlugins)
        self.menuRavenExport.setObjectName("menuRavenExport")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuSettings = QtWidgets.QMenu(self.menubar)
        self.menuSettings.setObjectName("menuSettings")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolbarDock = QtWidgets.QDockWidget(MainWindow)
        self.toolbarDock.setObjectName("toolbarDock")
        self.dockWidgetContents_3 = QtWidgets.QWidget()
        self.dockWidgetContents_3.setObjectName("dockWidgetContents_3")
        self.toolbarDock.setWidget(self.dockWidgetContents_3)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(4), self.toolbarDock)
        self.actionLoad_project = QtWidgets.QAction(MainWindow)
        self.actionLoad_project.setObjectName("actionLoad_project")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionExport = QtWidgets.QAction(MainWindow)
        self.actionExport.setObjectName("actionExport")
        self.actionKeymappings = QtWidgets.QAction(MainWindow)
        self.actionKeymappings.setObjectName("actionKeymappings")
        self.actionClose = QtWidgets.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.actionAdd_source = QtWidgets.QAction(MainWindow)
        self.actionAdd_source.setObjectName("actionAdd_source")
        self.actionDetect = QtWidgets.QAction(MainWindow)
        self.actionDetect.setObjectName("actionDetect")
        self.actionLower_Threshold = QtWidgets.QAction(MainWindow)
        self.actionLower_Threshold.setObjectName("actionLower_Threshold")
        self.actionIncrease_Threshold = QtWidgets.QAction(MainWindow)
        self.actionIncrease_Threshold.setObjectName("actionIncrease_Threshold")
        self.actionCreate_Segment = QtWidgets.QAction(MainWindow)
        self.actionCreate_Segment.setObjectName("actionCreate_Segment")
        self.actionClear_selection = QtWidgets.QAction(MainWindow)
        self.actionClear_selection.setObjectName("actionClear_selection")
        self.actionDelete_Segment = QtWidgets.QAction(MainWindow)
        self.actionDelete_Segment.setObjectName("actionDelete_Segment")
        self.actionAdd_label = QtWidgets.QAction(MainWindow)
        self.actionAdd_label.setObjectName("actionAdd_label")
        self.actionEdit_label = QtWidgets.QAction(MainWindow)
        self.actionEdit_label.setObjectName("actionEdit_label")
        self.actionNew_template_from_label = QtWidgets.QAction(MainWindow)
        self.actionNew_template_from_label.setObjectName("actionNew_template_from_label")
        self.actionMatch_to_selection = QtWidgets.QAction(MainWindow)
        self.actionMatch_to_selection.setObjectName("actionMatch_to_selection")
        self.actionExport_to_Raven = QtWidgets.QAction(MainWindow)
        self.actionExport_to_Raven.setObjectName("actionExport_to_Raven")
        self.actionGithub = QtWidgets.QAction(MainWindow)
        self.actionGithub.setObjectName("actionGithub")
        self.menuFile.addAction(self.actionLoad_project)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionExport)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionClose)
        self.menuSources.addAction(self.actionAdd_source)
        self.menuSources.addSeparator()
        self.menuSources.addAction(self.actionClear_selection)
        self.menuSources.addSeparator()
        self.menuSources.addAction(self.actionCreate_Segment)
        self.menuSources.addAction(self.actionDelete_Segment)
        self.menuSimpleDetection.addSeparator()
        self.menuSimpleDetection.addAction(self.actionDetect)
        self.menuSimpleDetection.addAction(self.actionLower_Threshold)
        self.menuSimpleDetection.addAction(self.actionIncrease_Threshold)
        self.menuLabeling.addAction(self.actionAdd_label)
        self.menuLabeling.addAction(self.actionEdit_label)
        self.menuTemplateMatching.addAction(self.actionNew_template_from_label)
        self.menuTemplateMatching.addAction(self.actionMatch_to_selection)
        self.menuRavenExport.addAction(self.actionExport_to_Raven)
        self.menuPlugins.addAction(self.menuSimpleDetection.menuAction())
        self.menuPlugins.addAction(self.menuLabeling.menuAction())
        self.menuPlugins.addAction(self.menuTemplateMatching.menuAction())
        self.menuPlugins.addAction(self.menuRavenExport.menuAction())
        self.menuHelp.addAction(self.actionGithub)
        self.menuSettings.addAction(self.actionKeymappings)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuSources.menuAction())
        self.menubar.addAction(self.menuPlugins.menuAction())
        self.menubar.addAction(self.menuSettings.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.pluginPanelToolbox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pluginPanelToolbox.setItemText(self.pluginPanelToolbox.indexOf(self.page_4), _translate("MainWindow", "Plugins"))
        self.previewBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuSources.setTitle(_translate("MainWindow", "Sources"))
        self.menuPlugins.setTitle(_translate("MainWindow", "Plugins"))
        self.menuSimpleDetection.setTitle(_translate("MainWindow", "SimpleDetection"))
        self.menuLabeling.setTitle(_translate("MainWindow", "Labeling"))
        self.menuTemplateMatching.setTitle(_translate("MainWindow", "TemplateMatching"))
        self.menuRavenExport.setTitle(_translate("MainWindow", "RavenExport"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.menuSettings.setTitle(_translate("MainWindow", "Settings"))
        self.actionLoad_project.setText(_translate("MainWindow", "Load project"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionExport.setText(_translate("MainWindow", "Export"))
        self.actionKeymappings.setText(_translate("MainWindow", "Keymappings"))
        self.actionClose.setText(_translate("MainWindow", "Close"))
        self.actionAdd_source.setText(_translate("MainWindow", "Add source..."))
        self.actionDetect.setText(_translate("MainWindow", "Detect"))
        self.actionLower_Threshold.setText(_translate("MainWindow", "Lower Threshold"))
        self.actionIncrease_Threshold.setText(_translate("MainWindow", "Increase Threshold"))
        self.actionCreate_Segment.setText(_translate("MainWindow", "Create Segment"))
        self.actionClear_selection.setText(_translate("MainWindow", "Clear selection"))
        self.actionDelete_Segment.setText(_translate("MainWindow", "Delete Segment"))
        self.actionAdd_label.setText(_translate("MainWindow", "Add label..."))
        self.actionEdit_label.setText(_translate("MainWindow", "Edit label"))
        self.actionNew_template_from_label.setText(_translate("MainWindow", "New template from label"))
        self.actionMatch_to_selection.setText(_translate("MainWindow", "Match to selection"))
        self.actionExport_to_Raven.setText(_translate("MainWindow", "Export to Raven TSV"))
        self.actionGithub.setText(_translate("MainWindow", "Source code"))
