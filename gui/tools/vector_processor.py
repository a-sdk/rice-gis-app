import geopandas as gpd
import json
import logging
import psutil
import os
from PyQt5.QtCore import pyqtSignal, QObject
from gui.layer_manager import Layer

logger = logging.getLogger(__name__)
process = psutil.Process(os.getpid())

class VectorHandler(QObject):
    crsDetected = pyqtSignal(str)
    vectorUpdated = pyqtSignal(int)

    def __init__(self, layer_manager):
        super().__init__()
        self.layer_manager = layer_manager
        self.vector_info = {}

    def refresh_vector(self):
        logger.info("Update vector origin...")
        for lid in self.vector_info:
            self.vectorUpdated.emit(lid)

    def add_shapefile(self, name, layer_id, path):
        logger.info(f"==== ADD VECTOR {name} TO LAYER: {layer_id} ====")
        logger.info("Load shapefile")
        gdf = gpd.read_file(path)
        bounds = gdf.total_bounds
        xmin, ymin, xmax, ymax = bounds
        # logger.info(f"Vector extent x={xmin}, y={ymax}")
        # logger.info(f"{self.layer_items}")
        info = {
            "Name": name,
            "Source": path,
            "Type": "shp",
            "CRS": gdf.crs.to_string() if gdf.crs else "Non-Georeferenced",
            "Geom_type": str(gdf.geom_type.iloc[0]),
            "Count": len(gdf),
            "Legend": None,
            "Stats": None,
            "class_col": None
        }
        self.vector_info[layer_id] = info
        self.crsDetected.emit(info["CRS"])
        layer = Layer(
            sid=layer_id,
            name=name,
            item=gdf,
            layer_type="vector",
            metadata=info,
            extent=(xmin, ymin, xmax, ymax),
            qtransform=None
        )
        return layer

    def add_gpkg_layer(self, name, layer_id, path, layer_name):
        logger.info(F"==== ADD VEKTOR {layer_name} TO LAYER: {layer_id} ====")
        logger.info("Load geopackage...")
        gdf = gpd.read_file(path, layer=layer_name)
        bounds = gdf.total_bounds
        xmin, ymin, xmax, ymax = bounds
        legend, stats = self.read_gpkg_metadata(path, layer_name)
        # logger.info(f"{self.layer_items}")
        info = {
            "Name": name, 
            "Source": path,
            "Type": "gpkg",
            "Layer_name": layer_name,
            "CRS": gdf.crs.to_string() if gdf.crs else "Non-Georeferenced",
            "Geom_type": str(gdf.geom_type.iloc[0]),
            "Count": len(gdf),
            "Legend": legend,
            "Stats": stats,
            "class_col": "preds"
        }
        self.vector_info[layer_id] = info
        self.crsDetected.emit(info["CRS"])
        layer = Layer(
            sid=layer_id,
            name=layer_name,
            item=gdf,
            layer_type="vector",
            metadata=info,
            extent=(xmin, ymin, xmax, ymax),
            qtransform=None
        )
        return layer
    
    def read_gpkg_metadata(self, path, layer_name):
        import sqlite3
        conn = sqlite3.connect(path)
        cur = conn.cursor()

        cur.execute("""
        SELECT legend_json, stats_json
        FROM app_layer_metadata
        WHERE layer_name = ?
        """, (layer_name,))

        row = cur.fetchone()
        conn.close()

        if row:
            return json.loads(row[0]), json.loads(row[1])

        return None, None

    def remove_vector(self, layer_id):
        self.vector_info.pop(layer_id, None)