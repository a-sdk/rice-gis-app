import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ONNX must be imported before PyQt5/GDAL and other native dependencies.
# Prevents Windows native-library loading conflict when controller imports ORT
import onnxruntime
import rasterio 

from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    if os.name == "nt":
        # Windows App ID
        import ctypes
        myappid = 'unpad.ricegis.beta-0.0.1'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    import multiprocessing
    multiprocessing.freeze_support()
    # Start app 
    from path_config import AppPaths
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    pixmap = QPixmap(str(AppPaths.assets("defaults/textures/splash_loading.png")))
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.show()
    splash.showMessage("Starting app...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
    app.processEvents()
    # Init runtime folder
    splash.showMessage("Initializing runtime folder...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
    app.processEvents()
    from core.runtime_initializer import initialize_runtime
    initialize_runtime()
    # Init main window
    splash.showMessage("Loading python modules...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
    app.processEvents()
    from gui.main_window import MainWindow
    splash.showMessage("Loading main window...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
    app.processEvents()
    window = MainWindow()
    window.setGeometry(100, 100, 800, 600) 
    window.setMinimumSize(400, 300)
    window.show()
    splash.finish(window)
    # Logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(str(AppPaths.LOGS / "app_debug.log"), mode='w'), 
            logging.StreamHandler(sys.stdout) 
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("App opened!")
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.error("Uncaught exception.", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception

    sys.exit(app.exec_())


