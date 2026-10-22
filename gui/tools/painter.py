from PyQt5.QtWidgets import QGraphicsPolygonItem
from PyQt5.QtGui import (
    QPolygonF, QPen, QColor
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal
import logging
logger = logging.getLogger(__name__)
class PolygonDrawingTool(QObject):
    drawFinished = pyqtSignal(str)
    drawInfoMsg = pyqtSignal(str)
    def __init__(self, viewer):
        super().__init__()
        self.renderer = viewer
        self.temp_shp_type = None
        self.temp_shp_path = None
        self.temp_shp_crs = None
        self.active_poly_item = None
        self.commit_poly_item = []
        self.temp_shp_points = []
        self.geo_coords = []
        self.list_poly = []
        self.is_drawing = False
        self.origin_x = None
        self.origin_y = None

    def update_draw_origin(self, x, y):
        self.origin_x = x
        self.origin_y = y

    def start_drawing(self, path, dtype, crs):
        self.is_drawing = True
        self.temp_shp_crs = crs
        self.temp_shp_path = path
        self.temp_shp_type = dtype
        self.renderer.viewer.setFocus(True)
        self.renderer.viewer.setCursor(Qt.CrossCursor)
        self.renderer.viewer.viewport().setCursor(Qt.CrossCursor)

    def mouse_press(self, pos, button):
        if self.is_drawing:
            if button == Qt.MouseButton.LeftButton:
                # logger.info("Mode gambar: klik kiri")
                # logger.info(f"scene pos={pos}")
                self.temp_shp_points.append(pos)
                if self.origin_x and self.origin_y:
                    world_x = pos.x() + self.origin_x
                    world_y = pos.y() + self.origin_y
                    coords = (world_x, world_y)
                    # logger.info(f"utm={coords}")
                    self.geo_coords.append(coords)
                if self.temp_shp_type == "Point":
                    self.finalize_polygon()
                else:
                    self.update_draw_polygon()
            elif button == Qt.MouseButton.RightButton:
                # logger.info("Mode gambar: klik kanan")
                num_points = len(self.temp_shp_points)
                if self.temp_shp_type == "Polygon" and num_points < 3:
                    self.drawInfoMsg.emit("Polygon require 3 or more points!")
                elif self.temp_shp_type == "LineString" and num_points < 2:
                    self.drawInfoMsg.emit("Line require 2 or more points!")
                elif self.temp_shp_type == "Point" and num_points < 1:
                    self.drawInfoMsg.emit("Point not defined!")
                else:
                    self.finalize_polygon()

    def key_press(self, key):
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.save_polygon()
            if self.is_drawing:
                self.stop_drawing()
        if key == Qt.Key_Escape:
            if self.temp_shp_points:
                self.cancel_drawing()
            else:
                self.stop_drawing()

    def cancel_drawing(self):
        if self.active_poly_item:
            self.renderer.scene.removeItem(self.active_poly_item)
            self.active_poly_item = None
        self.temp_shp_points.clear()
        self.geo_coords.clear()
        # logger.info("Drawing cancelled")
        self.drawInfoMsg.emit("Drawing cancelled")

    def stop_drawing(self):
        self.renderer.viewer.setCursor(Qt.ArrowCursor)
        self.renderer.viewer.viewport().setCursor(Qt.ArrowCursor)
        self.renderer.viewer.setFocus(False)
        # logger.info("Drawing mode disabled")
        self.drawInfoMsg.emit("Drawing mode disabled")
        self.is_drawing = False
        self.cancel_drawing()
        if self.commit_poly_item:
            for item in self.commit_poly_item:
                self.renderer.scene.removeItem(item)
        self.commit_poly_item.clear()
        self.list_poly.clear()
        self.temp_shp_crs = None
        self.temp_shp_path = None
        self.temp_shp_type = None
        self.renderer.viewer.viewport().update() 

    def update_draw_polygon(self):
        if not self.temp_shp_points:
            return
        
        if self.active_poly_item:
            self.renderer.scene.removeItem(self.active_poly_item)

        polygon_data = QPolygonF(self.temp_shp_points)
        self.active_poly_item = QGraphicsPolygonItem(polygon_data)
        pen = QPen(QColor(255, 0, 0, 150))
        pen.setWidth(1)
        pen.setCosmetic(True)
        self.active_poly_item.setPen(pen)
        self.active_poly_item.setBrush(QColor(255, 0, 0, 50))
        self.renderer.scene.addItem(self.active_poly_item)
        self.active_poly_item.setZValue(100)
        self.renderer.viewer.viewport().update()

    def finalize_polygon(self):
        from shapely.geometry import Point, LineString, Polygon
        GEOM_MAPPING = {
            "Point": Point,
            "LineString": LineString,
            "Polygon": Polygon
        }
        dtype = GEOM_MAPPING.get(self.temp_shp_type)
        if not dtype:
            return
        try:
            if dtype == "Point":
                geom = dtype(self.geo_coords[0])
            else:
                geom = dtype(self.geo_coords)
                self.list_poly.append(geom)
                # logger.info(f"Polygon yang dibuat: {self.list_poly}")
            if not geom.is_valid:
                self.drawInfoMsg.emit("Geometry is not valid")
                # logger.info("Geometri is not valid")
                return
            final_poly_item = QGraphicsPolygonItem(QPolygonF(self.temp_shp_points))
            pen = QPen(QColor(153, 245, 39, 150))
            pen.setWidth(1)
            pen.setCosmetic(True)
            final_poly_item.setPen(pen)
            final_poly_item.setBrush(QColor(153, 245, 39, 50))
            self.commit_poly_item.append(final_poly_item)
            # logger.info(f"commit_poly_item={len(self.commit_poly_item)}")
            self.renderer.scene.addItem(final_poly_item)
            final_poly_item.setZValue(100)
            self.renderer.viewer.viewport().update()

            if self.active_poly_item:
                self.renderer.scene.removeItem(self.active_poly_item)
                self.active_poly_item = None
                self.geo_coords.clear()
                self.temp_shp_points.clear()
                
        except Exception as e:
            logger.error(f"ERROR: {type(e).__name__}: {e}", exc_info=True)

    def save_polygon(self):
        import geopandas as gpd
        try:
            gdf = gpd.GeoDataFrame({'id':range(len(self.list_poly))}, crs=self.temp_shp_crs, geometry=self.list_poly)
            gdf.to_file(self.temp_shp_path, driver="ESRI Shapefile")
            self.drawFinished.emit(self.temp_shp_path)

        except Exception as e:
            logger.error(f"ERROR: {type(e).__name__}: {e}", exc_info=True)