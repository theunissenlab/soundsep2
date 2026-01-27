import asyncio
import sys

import pyqtgraph as pg
import logging
from PyQt6 import QtWidgets as widgets
from PyQt6.QtCore import Qt
from qasync import QEventLoop

import warnings
import numpy as np

# Suppress pyqtgraph ViewBox overflow warning (cosmetic, doesn't affect functionality)
warnings.filterwarnings(
    "ignore",
    message="overflow encountered in cast",
    category=RuntimeWarning,
    module="pyqtgraph.graphicsItems.ViewBox.ViewBox"
)


def _activate_window(window):
    """Bring a window to the foreground, especially needed on macOS."""
    if hasattr(window, 'splash') and window.splash is not None:
        # For Launcher which uses a splash widget
        window.splash.raise_()
        window.splash.activateWindow()
    elif hasattr(window, 'raise_'):
        window.raise_()
        window.activateWindow()


def run_app(*args, MainWindow=None, debug=False, **kwargs):
    """Run an app using asyncio event loop
    """
    # Handle high resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        widgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        widgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    pg.setConfigOption('background', None)
    pg.setConfigOption('foreground', 'k')

    # Set up logging
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if debug else logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)

    rootLogger = logging.getLogger()
    rootLogger.addHandler(console)
    rootLogger.setLevel(level=logging.DEBUG if debug else logging.INFO)

    app = widgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    if MainWindow is None:
        mainWindow = widgets.QMainWindow(*args, **kwargs)
    else:
        mainWindow = MainWindow(*args, **kwargs)

    mainWindow.show()
    _activate_window(mainWindow)
    with loop:
        loop.run_forever()


if __name__ == "__main__":
    from soundsep.app.launcher import Launcher

    run_app(MainWindow=Launcher)
