from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5 import uic
from path_config import AppPaths
import logging

logger = logging.getLogger(__name__)

class InputDialog(QDialog):
    def __init__(self, parent=None, title="", icon=None, metadata=None):
        super().__init__(parent)
        uic.loadUi(str(AppPaths.ui("input_dialog.ui")), self)

        self.temp_metadata = metadata
        self.setWindowTitle(title)
        self.setWindowIcon(icon)
        self._init_layer_comboBox()
        self.btn_browse_tif.clicked.connect(self._handle_browse_tif)
        self.btn_browse_shp.clicked.connect(self._handle_browse_shp)
        self.btn_browse_out.clicked.connect(self._handle_browse_out)
        self.tifcomboBox.currentIndexChanged.connect(self._handle_layer_comboBox)
        self.shpcomboBox.currentIndexChanged.connect(self._handle_layer_comboBox)

    def _handle_browse_tif(self):
        # logger.info("btn_browse_tif pressed")
        path, _ = QFileDialog.getOpenFileName(self, "Open Supported Raster", "", "GeoTiff (*.tif *.tiff)")
        if path: self.lineEdit_tif.setText(path)

    def _handle_browse_shp(self):
        # logger.info("btn_browse_shp pressed")
        path, _ = QFileDialog.getOpenFileName(self, "Open Supported Vector", "", "ESRI Shapefile (*.shp)")
        if path: self.lineEdit_shp.setText(path)

    def _init_layer_comboBox(self):
        # logger.info("User clicked drop-down combo box")
        if not self.temp_metadata:
            self.tifcomboBox.clear()
            self.shpcomboBox.clear()
            return

        for lid, info in self.temp_metadata.items():
            name = info.get("Name")
            if name.lower().endswith(".tif"):
                self.tifcomboBox.addItem(name, lid)
            if name.lower().endswith(".shp"):
                self.shpcomboBox.addItem(name, lid)

        self._handle_layer_comboBox()

    def _handle_layer_comboBox(self):
        id_tif = self.tifcomboBox.currentData()
        id_shp = self.shpcomboBox.currentData()
        if id_tif != None and id_shp != None:
            logger.info(f"User memilih tif:{id_tif}, shp: {id_shp}")
            self.lineEdit_tif.setText(self.temp_metadata[id_tif]["Source"])
            self.lineEdit_shp.setText(self.temp_metadata[id_shp]["Source"])
        else:
            self.lineEdit_tif.setText("")
            self.lineEdit_shp.setText("")

    def _handle_browse_out(self):
        # logger.info("btn_browse_out pressed")
        path = QFileDialog.getExistingDirectory(self, "Save In")
        if path: self.lineEdit_out.setText(path)   

    def get_values(self):
        return (self.lineEdit_tif.text(), 
                self.lineEdit_shp.text(), 
                self.lineEdit_out.text()) 