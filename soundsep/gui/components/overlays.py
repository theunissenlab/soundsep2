import PyQt5.QtWidgets as widgets
from PyQt5.QtCore import Qt, QSize


class FloatingButton(widgets.QPushButton):

    def __init__(self, *args, paddingx=0, paddingy=0, parent=None, **kwargs):
        super().__init__(*args, parent=parent, **kwargs)
        self.setStyleSheet("""
            QPushButton {
                background-color: rgba(150, 50, 50, 20%);
                color: rgba(255, 255, 255, 50%);
                padding: 0px;
                border: 0px;
                font-size: 36;
                text-align: left;
            }

            QPushButton:hover {
                background-color: rgba(20, 20, 20, 60%);
                color: rgba(255, 255, 255, 100%);
            }

            QPushButton:pressed {
                background-color: rgba(50, 50, 50, 100%);
                color: rgba(255, 255, 255, 100%);
            }
        """)
        self.padding = (paddingx, paddingy)

    def update_position(self):
        if hasattr(self.parent(), 'viewport'):
            parent_rect = self.parent().viewport().rect()
        else:
            parent_rect = self.parent().rect()
        
        self.adjustSize()

        if not parent_rect:
            return

        x = self.padding[0]
        y = self.padding[1]
        self.setGeometry(x, y, self.width(), self.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_position()


class FloaingComboBox(widgets.QComboBox):

    def __init__(self, *args, padding=10, parent=None, **kwargs):
        super().__init__(*args, parent=parent, **kwargs)
        self.setStyleSheet("""
            QComboBox {
                background-color: rgba(150, 50, 50, 20%);
                padding: 0px;
                border: 0px;
                color: rgba(255, 255, 255, 1);
                font-size: 24;
                text-align: left;
            }

            QComboBox:hover {
                background-color: rgba(20, 20, 20, 60%);
            }

            QComboBox:pressed {
                background-color: rgba(50, 50, 50, 100%);
            }
        """)
        self.padding = padding

    def update_position(self):
        if hasattr(self.parent(), 'viewport'):
            parent_rect = self.parent().viewport().rect()
        else:
            parent_rect = self.parent().rect()
        
        self.adjustSize()

        if not parent_rect:
            return

        x = self.padding
        y = self.padding
        self.setGeometry(x, y, self.width(), self.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_position()


