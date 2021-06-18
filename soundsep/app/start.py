import asyncio
import sys

import pyqtgraph as pg
import logging
from PyQt5 import QtWidgets as widgets
from qasync import QEventLoop


def run_app(*args, MainWindow=None, **kwargs):
    """Run an app using asyncio event loop
    """

    pg.setConfigOption('background', None)
    pg.setConfigOption('foreground', 'k')

    # Set up logging
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)

    rootLogger = logging.getLogger()
    rootLogger.addHandler(console)
    rootLogger.setLevel(level=logging.DEBUG)

    app = widgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    if MainWindow is None:
        mainWindow = widgets.QMainWindow(*args, **kwargs)
    else:
        mainWindow = MainWindow(*args, **kwargs)

    mainWindow.show()
    with loop:
        loop.run_forever()


if __name__ == "__main__":
    from soundsep.app.launcher import Launcher

    run_app(MainWindow=Launcher)
