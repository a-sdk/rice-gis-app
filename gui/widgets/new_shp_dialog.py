from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5 import uic
from path_config import AppPaths
import logging

logger = logging.getLogger(__name__)

class CreateShapefileDialog(QDialog):
    def __init__(self, parent=None, crs=None):
        super().__init__(parent)
        uic.loadUi(str(AppPaths.ui("new_shp_dialog.ui")), self)
        self.viewer_crs = crs
        self.setWindowTitle("New Shapefile Layer")
        self.setWindowIcon(QIcon(str(AppPaths.assets("defaults/textures/icon/poly.png"))))
        self._init_crs()
        self.btn_save_shp.clicked.connect(self._handle_save_shp)

    def _handle_save_shp(self):
        # logger.info("btn_create_shp pressed")
        path, _ = QFileDialog.getSaveFileName(self, "Save Layer As", "", "ESRI Shapefile (*.shp)")
        if path: self.lineEdit_new_shp.setText(path)   

    def get_values(self):
        return (self.lineEdit_new_shp.text(), 
                self.comboBox_geom_type_shp.currentText(), 
                self.lineEdit_crs_new_shp.text()) 
    
    def _init_crs(self):
        if self.viewer_crs:
            self.lineEdit_crs_new_shp.setText(self.viewer_crs)
    