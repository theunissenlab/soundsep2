import asyncio

from PyQt5 import QtWidgets as widgets

from qasync import QEventLoop


def run_app(MainWindow=None):
    """Run an app using asyncio event loop
    """
    import sys

    app = widgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    if MainWindow is None:
        mainWindow = widgets.QMainWindow()
    else:
        mainWindow = MainWindow()

    mainWindow.show()

    with loop:
        loop.run_forever()

