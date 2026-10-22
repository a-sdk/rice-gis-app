from PyQt5 import uic
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, 
    QFileDialog, QMessageBox, QProgressBar,
    QProgressDialog
)
from gui.widgets.viewer import Viewer
from gui.widgets.layer_panel import LayerPanel
from gui.widgets.legend_panel import LegendPanel
from gui.widgets.input_dialog import InputDialog
from gui.widgets.new_shp_dialog import CreateShapefileDialog
from gui.tools.painter import PolygonDrawingTool
from gui.tools.raster_processor import RasterHandler
from gui.tools.vector_processor import VectorHandler
from app.worker import Worker
from gui.layer_manager import LayerManager
from app.controller import (
    NutrientController, WaterController, DiseaseController
)
from app.result_model import AnalysisResult
from path_config import AppPaths
import os
import logging
import shutil

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(str(AppPaths.ui("main_window.ui")), self)
        self.setWindowTitle("RiceGIS")
        self.setWindowIcon(QIcon(str(AppPaths.assets("defaults/textures/icon/ricefield.png"))))
        self._layer_id = 0
        self.elapsed_sec = 0
        self.active_thread = []
        self.scene_extent = ()
        self.queue_files = None
        self.time_str = str(0)
        self.current_crs = "Unknown"
        self.current_msg = "Processing..."
        self.layer_manager = LayerManager()
        self.viewer = Viewer(
            layer_manager=self.layer_manager, 
            parent=self.containerViewer
            )
        viewer_layout = QVBoxLayout(self.containerViewer)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.addWidget(self.viewer)
        # logger.info(f"Viewer panel tipe: {type(self.viewer)}, parent: {self.viewer.parent()}")

        self.layer_panel = LayerPanel(
            layer_manager=self.layer_manager,
            parent=self.containerLayer, 
            viewer=self.viewer
            )
        layer_layout = QVBoxLayout(self.containerLayer)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        layer_layout.addWidget(self.layer_panel)
        # logger.info(f"Layer panel tipe: {type(self.layer_panel)}, parent: {self.layer_panel.parent()}")

        self.legend_panel = LegendPanel(self.containerLegend)
        legend_layout = QVBoxLayout(self.containerLegend)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        legend_layout.addWidget(self.legend_panel)
        # logger.info(f"Legend panel tipe: {type(self.legend_panel)}, parent: {self.legend_panel.parent()}")

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        self.polygon_tool = PolygonDrawingTool(self.viewer)
        self.viewer.active_tool = self.polygon_tool
        self.raster_handler = RasterHandler(layer_manager=self.layer_manager)
        self.vector_handler = VectorHandler(layer_manager=self.layer_manager)
        self.disease_ctrl = DiseaseController()
        self.water_ctrl = WaterController()
        self.nutrient_ctrl = NutrientController()
        self._setup_icons()
        self._setup_statusbar()
        self.statusBar().showMessage("Ready")
        self.statusBar().addPermanentWidget(self.progress_bar)
        self._connect_signals()
        self.viewer.mouseMoved.connect(self.update_coord_label)
        self.raster_handler.crsDetected.connect(self.update_crs_label)
        self.viewer.fit_to_view()

        self.dockLayers.setWindowTitle("Layers")
        self.action_layer_panel.setCheckable(True)
        self.action_layer_panel.toggled.connect(self.dockLayers.setVisible)
        self.dockLayers.visibilityChanged.connect(self.action_layer_panel.setChecked)

        self.dockLegend.setWindowTitle("Legend")
        self.action_legend_panel.setCheckable(True)
        self.action_legend_panel.toggled.connect(self.dockLegend.setVisible)
        self.dockLegend.visibilityChanged.connect(self.action_legend_panel.setChecked)

        self.dockLayers.show()
        self.action_layer_panel.setChecked(True)
        self.dockLegend.hide()
        self.action_legend_panel.setChecked(False)

    def _setup_icons(self):
        icon_tif = QIcon(str(AppPaths.assets("defaults/textures/icon/add_img.png")))
        icon_shp = QIcon(str(AppPaths.assets("defaults/textures/icon/poly.png")))
        icon_open = QIcon(str(AppPaths.assets("defaults/textures/icon/folder.png")))
        icon_exit = QIcon(str(AppPaths.assets("defaults/textures/icon/exit.png")))
        icon_model = QIcon(str(AppPaths.assets("defaults/textures/icon/deep-learning.png")))
        self.icon_disease = QIcon(str(AppPaths.assets("defaults/textures/icon/virus.png")))
        self.icon_mineral = QIcon(str(AppPaths.assets("defaults/textures/icon/fertilizer.png")))
        self.icon_water = QIcon(str(AppPaths.assets("defaults/textures/icon/drop.png")))
        icon_hand = QIcon(str(AppPaths.assets("defaults/textures/icon/hand.png")))
        icon_zoom_in = QIcon(str(AppPaths.assets("defaults/textures/icon/zoom-in.png")))
        icon_zoom_out = QIcon(str(AppPaths.assets("defaults/textures/icon/zoom-out.png")))
        icon_fit_to_view = QIcon(str(AppPaths.assets("defaults/textures/icon/width.png")))
        icon_new_shp = QIcon(str(AppPaths.assets("defaults/textures/icon/edit_poly.png")))
        self.menuOpen.setIcon(icon_open)
        self.action_open_img.setIcon(icon_tif)
        self.action_open_shp.setIcon(icon_shp)
        self.action_add_raster_layer.setIcon(icon_tif)
        self.action_add_vector_layer.setIcon(icon_shp)
        self.action_new_shp_layer.setIcon(icon_new_shp)
        self.action_exit.setIcon(icon_exit)
        self.action_pan.setIcon(icon_hand)
        self.action_zoom_in.setIcon(icon_zoom_in)
        self.action_zoom_out.setIcon(icon_zoom_out)
        self.action_fit_to_view.setIcon(icon_fit_to_view)
        self.menuModel.setIcon(icon_model)
        self.action_nutrient_predict.setIcon(self.icon_mineral)
        self.action_water_predict.setIcon(self.icon_water)
        self.action_disease_predict.setIcon(self.icon_disease)
        self.layer_panel.tree.setIconSize(QSize(24, 24))

    def _setup_statusbar(self):
        self.coord_label = QLabel("X: -, Y: -")
        self.crs_label = QLabel("CRS: -")
        self.statusBar().addPermanentWidget(self.crs_label)
        self.statusBar().addPermanentWidget(self.coord_label)

    def _setup_progress_dialog(self):
        import time
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()

        self.start_time = time.monotonic()
        self.elapsed_sec = 0
        self.mins = 0
        self.secs = 0
        self.time_str = "0s"

        self.pd = QProgressDialog(self.current_msg, "Cancel", 0, 100, self)
        self.pd.setWindowTitle("Please wait")
        self.pd.setWindowModality(Qt.WindowModal)
        self.pd.setMinimumDuration(0)
        self.pd.setValue(0)

        if not hasattr(self, 'timer'):
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._update_timer_label)
        self.timer.start(1000)

    def handle_progress_dialog(self, val, msg):
        self.current_msg = msg
        self.pd.setValue(val)
        self.pd.setLabelText(f"{msg}\nElapsed: {self.time_str}")

    def _update_timer_label(self):
        import time
        start = getattr(self, 'start_time', None)
        if start is not None:
            self.elapsed_sec = int(time.monotonic() - start)
        else:
            self.elapsed_sec += 1

        self.mins = self.elapsed_sec // 60
        self.secs = self.elapsed_sec % 60
        self.time_str = f"{self.mins}m {self.secs}s" if self.mins > 0 else f"{self.secs}s"
        if hasattr(self, 'pd') and self.pd.isVisible():
            self.pd.setLabelText(f"{self.current_msg}\nElapsed: {self.time_str}")

    def reset_elapsed(self):
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        self.elapsed_sec = 0
        self.time_str = "0s"
        self.mins = 0
        self.secs = 0
        
    def update_scene_extent(self):
        # logger.info("Update scene extent...")
        extents = []
        
        for layer_id in self.layer_manager.list_ids():
            layer = self.layer_manager.get_layer(layer_id)

            if layer is None or layer.extent is None:
                continue

            extents.append(layer.extent)

        if not extents:
            return

        xmin = min(e[0] for e in extents)
        ymin = min(e[1] for e in extents)
        xmax = max(e[2] for e in extents)
        ymax = max(e[3] for e in extents)
        self.scene_extent = (xmin, ymin, xmax, ymax)
        # logger.info(f"Global extent, x={xmin}, y={ymax}")
        self.update_scene_origin(xmin, ymax)

    def update_scene_origin(self, xmin, ymax):
        # logger.info("Update origin...")
        # logger.info(f"scene origin x={xmin}")
        # logger.info(f"scene origin y={ymax}")
        self.viewer.update_origin(xmin, ymax)
        self.raster_handler.update_origin(xmin, ymax)
        self.vector_handler.refresh_vector()
        self.polygon_tool.update_draw_origin(xmin, ymax)

    def update_coord_label(self, x, y):
        self.coord_label.setText(f"X: {x:.2f}, Y: {y:.2f}")

    def update_crs_label(self, crs):
        self.crs_label.setText(f"CRS: {crs}")
        self.current_crs = crs 

    def _status_bar_info(self, msg):
        self.statusBar().showMessage(f"{msg}", 3000)

    def start_worker(self, worker):
        self.active_thread.append(worker)
        self._setup_progress_dialog()
        worker.progress_signal.connect(
            self.handle_progress_dialog
        )
        worker.progress_signal.connect(
            self.update_progress_bar
        )
        worker.error_signal.connect(
            self.show_error_msg
        )
        self.pd.canceled.connect(
            worker.requestInterruption
        )
        worker.cancel_signal.connect(
            self.show_cancelled
        )
        worker.finished.connect(
            lambda: self.remove_worker(worker)
        )
        worker.start()

    def remove_worker(self, worker):
        if worker in self.active_thread:
            self.active_thread.remove(worker)

    def _connect_signals(self):
        # logger.info("Action signal connected")
        self.raster_handler.crsDetected.connect(self.update_crs_label)
        self.raster_handler.rasterUpdated.connect(self.viewer.render_geotiff)
        self.vector_handler.crsDetected.connect(self.update_crs_label)
        self.vector_handler.vectorUpdated.connect(self.viewer._render_vector)
        self.layer_panel.layerUpdated.connect(self.viewer.update_z_order)
        self.layer_panel.layerRemoveRequested.connect(self.remove_layer)
        self.layer_panel.layerSelected.connect(self.update_legend_from_layer)
        self.layer_panel.infoMsg.connect(self._status_bar_info)
        self.polygon_tool.drawInfoMsg.connect(self._status_bar_info)
        self.polygon_tool.drawFinished.connect(self.draw_shp_finished)
        self.viewer.viewportChanged.connect(self.on_viewport_changed)
        self.action_open_shp.triggered.connect(self.open_vector_file)
        self.action_open_img.triggered.connect(self.open_img_file)
        self.action_add_raster_layer.triggered.connect(self.open_img_file)
        self.action_add_vector_layer.triggered.connect(self.open_vector_file)
        self.action_new_shp_layer.triggered.connect(self.create_new_shapefile)
        self.action_exit.triggered.connect(self.close)
        self.action_pan.toggled.connect(self.viewer.set_pan_mode)
        self.action_zoom_in.triggered.connect(self.viewer.zoom_in)
        self.action_zoom_out.triggered.connect(self.viewer.zoom_out)
        self.action_fit_to_view.triggered.connect(self.viewer.fit_to_view)
        self.action_disease_predict.triggered.connect(self.run_disease_prediction)
        self.action_water_predict.triggered.connect(self.run_water_prediction)
        self.action_nutrient_predict.triggered.connect(self.run_nutrient_prediction)
        # self.action_debug.triggered.connect(self.simulate_finished)

    def process_next_queue(self):
        if not self.queue_files:
            self.timer.stop()
            # logger.info("All raster loaded succesfully!")
            return

        file = self.queue_files.pop(0)
        self.load_raster_layer(file)      

    def load_raster_layer(self, path):
        file_name = os.path.basename(path)
        self._layer_id += 1
        layer_id = self._layer_id
        raster_worker = Worker(
            self.raster_handler.add_raster,
            file_name,
            layer_id,
            path
        )
        raster_worker.finished_signal.connect(
            self.on_img_loaded
        )
        raster_worker.finished.connect(self.process_next_queue)
        self.start_worker(
            raster_worker
        )

    def load_vector_layer(self, path):
        file_name = os.path.basename(path)
        import fiona
        if path.lower().endswith(".gpkg"):
            gpkg_layers = fiona.listlayers(path)
            for lyr in gpkg_layers:
                if lyr != "app_layer_metadata":
                    self._layer_id += 1
                    layer_id = self._layer_id
                    layer = self.vector_handler.add_gpkg_layer(file_name, layer_id, path, lyr)
                    new_name = f"{file_name} | {lyr}"
                    self.layer_manager.add_layer(layer)
                    self.layer_panel.add_layer_item(layer.sid, new_name)
                    self.update_scene_extent()
                    self.viewer._render_vector(layer.sid)
            return     
        elif path.lower().endswith((".shp")):
            self._layer_id += 1
            layer_id = self._layer_id
            layer = self.vector_handler.add_shapefile(file_name, layer_id, path)
            self.layer_manager.add_layer(layer)
            self.layer_panel.add_layer_item(layer.sid, layer.name)
            self.update_scene_extent()
            self.viewer._render_vector(layer.sid)

    def open_img_file(self):
        # logger.info("Action: open_img pressed")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Supported Raster", "", "GeoTIFF (*.tif *.tiff);;Images (*.jpg *.png)"
        )
        if path:
            self.load_raster_layer(path)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage("Ready")

    def open_vector_file(self):
        # logger.info("Action: open_shp pressed")
        path, _ = QFileDialog.getOpenFileName(self, "Open Supported Vector", "", "ESRI Shapefile (*.shp);;Geopackage (*.gpkg)")
        
        if path:
            self.load_vector_layer(path)
    
    def on_img_loaded(self, result):
        if result is None:
            logger.error("Raster load failed: result is None or thread error")
        else:
            self.layer_manager.add_layer(result)
            self.layer_panel.add_layer_item(result.sid, result.name)
            self.update_scene_extent()
            self.viewer.render_geotiff(
                result.sid,
                init_view=True)

    def on_viewport_changed(self):
        canvas, zoom_ratio = self.viewer.get_viewport_state()
        self.raster_handler.set_viewport_state(
            canvas, 
            zoom_ratio
            )
        self.raster_handler.trigger_update_overview_level()

    def remove_layer(self, layer_id):
        layer = self.layer_manager.remove_layer(layer_id)
        self.dockLegend.hide()
        if not layer:
            return
        self.viewer.remove_item(layer_id)
        self.layer_panel.remove_layer_item(layer_id)
        self.update_scene_extent()
        if layer.layer_type == "raster":
            self.delete_temp_folder(layer.name)
            self.raster_handler.remove_raster(layer_id)
        if layer.layer_type == "vector":
            self.vector_handler.remove_vector(layer_id)
        if self.layer_manager.is_empty:
            self.update_crs_label("Unknown")
            self.legend_panel.clear()
            self.dockLegend.hide()
            return

    def delete_temp_folder(self, layer_name):
        name = layer_name.split(".")[0]
        try:
            if name:
                from path_config import AppPaths
                temp_dir = AppPaths.TEMP / name
                # logger.info(f"Delete tempo folder: {name}")
                if os.path.exists(temp_dir):
                    shutil.rmtree(str(temp_dir))
                    # logger.info(f"Folder temp dihapus: {temp_dir}")
        except Exception as e:
            logger.error(f"Failed to delete: {e}")
            
    def create_new_shapefile(self):
        # logger.info("Action: action_new_shp_layer pressed")
        dialog = CreateShapefileDialog(self, crs=self.current_crs)
        if dialog.exec_():
            path, dtype, crs = dialog.get_values()
            # logger.info(f"Geom type: {dtype}, CRS: {crs}")
            
            if path:
                self.polygon_tool.start_drawing(path, dtype, crs)
                self.statusBar().showMessage("Drawing mode enabled. Left click to add points, right click to finish. Press enter/return to save, esc to cancel", 20000)
    
    def draw_shp_finished(self, path):
        self.polygon_tool.stop_drawing()
        if path:
            self.load_vector_layer(path)

    def get_layers_list(self):
        # logger.info("Update list layer combo box")
        layer_dict = {}
        if self.layer_panel.layer_ids:
            for lid in self.layer_panel.layer_ids:
                layer = self.layer_manager.get_layer(lid)
                info = layer.metadata
                name = info.get("Name")
                path = info.get("Source")
                layer_dict.update({
                    lid: {
                        "Name": name, 
                        "Source": path
                    }
                })
            # logger.info(f"Daftar layer: {layer_dict}")
        return layer_dict 

    def update_legend_from_layer(self, layer_id):
        layer = self.layer_manager.get_layer(layer_id)
        if not layer:
            return
        info = layer.metadata
        if not info:
            self.legend_panel.clear()
            return

        legend = None
        stats = None
        if info.get("Type") == "gpkg":
            legend, stats = self.vector_handler.read_gpkg_metadata(
                info["Source"],
                info["Layer_name"]
            )
        else:
            legend = info.get("Legend")
            stats = info.get("Stats")
        if not layer.is_visible:
            self.legend_panel.clear()
            self.dockLegend.hide()
            return
        elif not legend and not stats:
            self.legend_panel.clear()
            self.dockLegend.hide()
            return
        else:
            self.legend_panel.set_legend(legend)
            self.legend_panel.set_info(stats)
            # logger.info("Legend updated!")
            self.dockLegend.show()

    def update_progress_bar(self, value, msg):
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(value)
            self.progress_bar.setVisible(value > 0 and value < 100)

        self.statusBar().showMessage(msg, 1000*60*10) 
        logger.info(f"Progress {value}% - {msg}")

    def run_analysis(self, controller, tif, shp, out):
        # Inisiasi worker thread
        analysis_worker = Worker(controller.run, tif, shp, out)
        analysis_worker.finished_signal.connect(self.on_analysis_finished)
        self.start_worker(analysis_worker)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("Ready")

    def cleanup_after_worker(self):
        self.timer.stop()
        self.pd.close()

        if hasattr(self, "progress_bar"):
            self.progress_bar.setVisible(False)
        self.statusBar().clearMessage()

    def on_analysis_finished(self, result: AnalysisResult):
        # Menutup timer dan dialog progress
        self.timer.stop()
        self.pd.close()
        # Muncul dialog selesai
        self.show_finished()
        if not result:
            return
                
        # Membuka file hasil prediksi
        if isinstance(result.prediction_path, list):
            if not os.path.exists(result.prediction_path[0]):
                self.show_error_msg(f"File not found.")
                return
            self.queue_files = result.prediction_path
            if self.queue_files:
                try:
                    self.process_next_queue()
                except Exception as e:
                    self.show_error_msg(f"{str(e)}")
        else:
            try:
                if result.prediction_path:
                    self.load_vector_layer(result.prediction_path)
            except Exception as e:
                self.show_error_msg(f"{str(e)}")

    def show_finished(self):
        self.cleanup_after_worker()
        QMessageBox.information(
            self,
            "Done",
            f"All process finished.\nDuration: {self.time_str}"
            )
        self.reset_elapsed()

    def show_cancelled(self):
        self.cleanup_after_worker()
        QMessageBox.warning(
            self,
            "Cancelled",
            f"User aborted.\nDuration: {self.time_str}"
            )
        self.reset_elapsed()

    def show_error_msg(self, error_msg):
        self.cleanup_after_worker()
        QMessageBox.critical(
            self,
            "Error",
            str(error_msg)
            )
        self.reset_elapsed()
        
    def run_disease_prediction(self):
        # logger.info("action_disease_predict pressed")
        metadata = self.get_layers_list()
        dialog = InputDialog(self, title="Disease Detection", icon=self.icon_disease, metadata=metadata)
        if dialog.exec_():
            tif, shp, out = dialog.get_values()
            self.run_analysis(self.disease_ctrl, tif, shp, out)

    def run_water_prediction(self):
        # logger.info("action_water_predict pressed")
        metadata = self.get_layers_list()
        dialog = InputDialog(self, title="Water Availability", icon=self.icon_water, metadata=metadata)
        if dialog.exec_():
            tif, shp, out = dialog.get_values()
            self.run_analysis(self.water_ctrl, tif, shp, out)

    def run_nutrient_prediction(self):
        # logger.info("action_disease_predict pressed")
        metadata = self.get_layers_list()
        dialog = InputDialog(self, title="Nutrient Availability", icon=self.icon_mineral, metadata=metadata)
        if dialog.exec_():
            tif, shp, out = dialog.get_values()
            self.run_analysis(self.nutrient_ctrl, tif, shp, out)

    def closeEvent(self, event):
        thread_running = False
        from path_config import AppPaths
        # logger.info("Attempting to close the app...")
        reply = QMessageBox.question(self, 'Exit', 'Are you sure?',
                                     QMessageBox.StandardButton.Yes |
                                     QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            for worker in self.active_thread:
                if worker.isRunning():
                    thread_running = True
                    # logger.info("Stop worker thread before exit...")
                    worker.requestInterruption()
                    worker.quit()
                    worker.wait()
                    if hasattr(self, 'pd'):
                        self.pd.close()
            if thread_running:
                logger.info("Wait background thread...")
            if AppPaths.TEMP.exists():
                logger.info("Delete temporary folder...")
                import shutil
                shutil.rmtree(AppPaths.TEMP)
            logger.info("Cleanup completed. App closed")
            event.accept()
        else:
            logger.info("Cancel exit.")
            event.ignore()


    