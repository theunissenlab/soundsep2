# Form implementation generated from reading ui file '/Users/kevin/Projects/soundsep2/soundsep/ui/project_creator.ui'
#
# Created by: PyQt6 UI code generator 6.2.3
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_ProjectCreator(object):
    def setupUi(self, ProjectCreator):
        ProjectCreator.setObjectName("ProjectCreator")
        ProjectCreator.resize(1704, 900)
        ProjectCreator.setMinimumSize(QtCore.QSize(1400, 900))
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(ProjectCreator)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.widget = QtWidgets.QWidget(ProjectCreator)
        self.widget.setObjectName("widget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.step1GroupBox = QtWidgets.QGroupBox(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.step1GroupBox.sizePolicy().hasHeightForWidth())
        self.step1GroupBox.setSizePolicy(sizePolicy)
        self.step1GroupBox.setObjectName("step1GroupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.step1GroupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label = QtWidgets.QLabel(self.step1GroupBox)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.basePathEdit = QClickableLineEdit(self.step1GroupBox)
        self.basePathEdit.setText("")
        self.basePathEdit.setReadOnly(True)
        self.basePathEdit.setObjectName("basePathEdit")
        self.horizontalLayout_2.addWidget(self.basePathEdit)
        self.browseButton = QtWidgets.QPushButton(self.step1GroupBox)
        self.browseButton.setObjectName("browseButton")
        self.horizontalLayout_2.addWidget(self.browseButton)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.recursiveSearchCheckBox = QtWidgets.QCheckBox(self.step1GroupBox)
        self.recursiveSearchCheckBox.setObjectName("recursiveSearchCheckBox")
        self.verticalLayout_3.addWidget(self.recursiveSearchCheckBox)
        self.widget_2 = QtWidgets.QWidget(self.step1GroupBox)
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.step1Next = QtWidgets.QPushButton(self.widget_2)
        self.step1Next.setObjectName("step1Next")
        self.horizontalLayout_4.addWidget(self.step1Next)
        self.verticalLayout_3.addWidget(self.widget_2)
        self.verticalLayout_5.addWidget(self.step1GroupBox, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        self.step2GroupBox = QtWidgets.QGroupBox(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.step2GroupBox.sizePolicy().hasHeightForWidth())
        self.step2GroupBox.setSizePolicy(sizePolicy)
        self.step2GroupBox.setObjectName("step2GroupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.step2GroupBox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.step2GroupBox)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.templateEdit = QtWidgets.QLineEdit(self.step2GroupBox)
        self.templateEdit.setObjectName("templateEdit")
        self.verticalLayout_2.addWidget(self.templateEdit)
        self.widget1 = QtWidgets.QWidget(self.step2GroupBox)
        self.widget1.setObjectName("widget1")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.step2Next = QtWidgets.QPushButton(self.widget1)
        self.step2Next.setObjectName("step2Next")
        self.horizontalLayout_3.addWidget(self.step2Next)
        self.verticalLayout_2.addWidget(self.widget1)
        self.verticalLayout_5.addWidget(self.step2GroupBox, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        self.step3GroupBox = QtWidgets.QGroupBox(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.step3GroupBox.sizePolicy().hasHeightForWidth())
        self.step3GroupBox.setSizePolicy(sizePolicy)
        self.step3GroupBox.setObjectName("step3GroupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.step3GroupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_3 = QtWidgets.QLabel(self.step3GroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.keysScrollArea = QtWidgets.QScrollArea(self.step3GroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.keysScrollArea.sizePolicy().hasHeightForWidth())
        self.keysScrollArea.setSizePolicy(sizePolicy)
        self.keysScrollArea.setWidgetResizable(True)
        self.keysScrollArea.setObjectName("keysScrollArea")
        self.keySelector = KeysSelector()
        self.keySelector.setGeometry(QtCore.QRect(0, 0, 795, 336))
        self.keySelector.setObjectName("keySelector")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.keySelector)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.keysScrollArea.setWidget(self.keySelector)
        self.verticalLayout.addWidget(self.keysScrollArea)
        self.verticalLayout_5.addWidget(self.step3GroupBox)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_5.addItem(spacerItem2)
        self.submitButtons = QtWidgets.QWidget(self.widget)
        self.submitButtons.setObjectName("submitButtons")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.submitButtons)
        self.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem3 = QtWidgets.QSpacerItem(1185, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.closeButton = QtWidgets.QPushButton(self.submitButtons)
        self.closeButton.setObjectName("closeButton")
        self.horizontalLayout.addWidget(self.closeButton)
        self.createConfigButton = QtWidgets.QPushButton(self.submitButtons)
        self.createConfigButton.setObjectName("createConfigButton")
        self.horizontalLayout.addWidget(self.createConfigButton)
        self.verticalLayout_5.addWidget(self.submitButtons)
        self.gridLayout.addWidget(self.widget, 0, 0, 1, 1)
        self.widget2 = QtWidgets.QWidget(ProjectCreator)
        self.widget2.setObjectName("widget2")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.widget2)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_4 = QtWidgets.QLabel(self.widget2)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_6.addWidget(self.label_4)
        self.treeView = AudioFileView(self.widget2)
        self.treeView.setObjectName("treeView")
        self.treeView.headerItem().setTextAlignment(0, QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.treeView.header().setStretchLastSection(False)
        self.verticalLayout_6.addWidget(self.treeView)
        self.verticalLayout_8.addLayout(self.verticalLayout_6)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_5 = QtWidgets.QLabel(self.widget2)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_7.addWidget(self.label_5)
        self.errorTable = QtWidgets.QTableWidget(self.widget2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.errorTable.sizePolicy().hasHeightForWidth())
        self.errorTable.setSizePolicy(sizePolicy)
        self.errorTable.setObjectName("errorTable")
        self.errorTable.setColumnCount(2)
        self.errorTable.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.errorTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.errorTable.setHorizontalHeaderItem(1, item)
        self.errorTable.horizontalHeader().setStretchLastSection(False)
        self.verticalLayout_7.addWidget(self.errorTable)
        self.verticalLayout_8.addLayout(self.verticalLayout_7)
        self.gridLayout.addWidget(self.widget2, 0, 1, 1, 1)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        self.horizontalLayout_5.addLayout(self.gridLayout)

        self.retranslateUi(ProjectCreator)
        QtCore.QMetaObject.connectSlotsByName(ProjectCreator)

    def retranslateUi(self, ProjectCreator):
        _translate = QtCore.QCoreApplication.translate
        ProjectCreator.setWindowTitle(_translate("ProjectCreator", "Form"))
        self.step1GroupBox.setTitle(_translate("ProjectCreator", "Step 1"))
        self.label.setText(_translate("ProjectCreator", "Select folder containing WAV files"))
        self.basePathEdit.setPlaceholderText(_translate("ProjectCreator", "Select base audio file directory"))
        self.browseButton.setText(_translate("ProjectCreator", "Browse..."))
        self.recursiveSearchCheckBox.setText(_translate("ProjectCreator", "Search recursively"))
        self.step1Next.setText(_translate("ProjectCreator", "Next"))
        self.step2GroupBox.setTitle(_translate("ProjectCreator", "Step 2"))
        self.label_2.setText(_translate("ProjectCreator", "Define a filename template string that matches your filenaming convention. Use {curly} {braces} to capture variables used for grouping."))
        self.templateEdit.setPlaceholderText(_translate("ProjectCreator", "{filename}.wav"))
        self.step2Next.setText(_translate("ProjectCreator", "Next"))
        self.step3GroupBox.setTitle(_translate("ProjectCreator", "Step 3"))
        self.label_3.setText(_translate("ProjectCreator", "Choose the variables from (2) to use as \"block keys\" or as \"channel keys\".\n"
"\n"
"Files that share the same \"block keys\" are be assumed to be simultaneously recorded and grouped accordingly\n"
"\n"
"Within a block, files are ordered by \"channel keys\""))
        self.closeButton.setText(_translate("ProjectCreator", "Close"))
        self.createConfigButton.setText(_translate("ProjectCreator", "Create config"))
        self.label_4.setText(_translate("ProjectCreator", "Project Files"))
        self.treeView.headerItem().setText(0, _translate("ProjectCreator", "Name"))
        self.treeView.headerItem().setText(1, _translate("ProjectCreator", "Id"))
        self.treeView.headerItem().setText(2, _translate("ProjectCreator", "Rate"))
        self.treeView.headerItem().setText(3, _translate("ProjectCreator", "Ch"))
        self.treeView.headerItem().setText(4, _translate("ProjectCreator", "Len"))
        self.label_5.setText(_translate("ProjectCreator", "Errors"))
        item = self.errorTable.horizontalHeaderItem(0)
        item.setText(_translate("ProjectCreator", "Filename"))
        item = self.errorTable.horizontalHeaderItem(1)
        item.setText(_translate("ProjectCreator", "Error"))
from soundsep.widgets.block_view import AudioFileView, KeysSelector
from soundsep.widgets.utils import QClickableLineEdit
