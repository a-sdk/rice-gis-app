import numpy as np
import logging
import psutil
import os
from PyQt5.QtGui import QTransform
from PyQt5.QtCore import pyqtSignal, QObject, QTimer
from app.worker import WorkerHelper
from gui.layer_manager import Layer
from path_config import AppPaths

logger = logging.getLogger(__name__)
process = psutil.Process(os.getpid())

class RasterHandler(QObject):
    crsDetected = pyqtSignal(str)
    rasterUpdated = pyqtSignal(int, bool)

    def __init__(self, layer_manager):
        super().__init__()
        self.OVERVIEW_FACTORS = [2,4,8,16]
        self.layer_manager = layer_manager
        self.raster_info = {}
        self.scene_origin_x = None
        self.scene_origin_y = None
        self.viewport_canvas = None
        self.viewport_zoom_ratio = None
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.update_overview_level)
        self.debounce_delay = 350 #ms
        self.raster_legend = None

    def get_raster_extent(self, transform, width, height):
        xmin = transform.c
        ymax = transform.f
        xmax = xmin + width * transform.a
        ymin = ymax + height * transform.e
        logger.info(f"Raster extent x={xmin}, y={ymax}")
        return xmin, ymin, xmax, ymax
    
    def update_origin(self, xmin, ymax):
        logger.info("Update origin raster...")
        self.scene_origin_x = xmin
        self.scene_origin_y = ymax
        
        for lid in self.raster_info:
            layer = self.layer_manager.get_layer(lid)
            if not layer or layer.item is None:
                continue
            info = layer.metadata
            layer.qtransform = self.build_qtransform(
                info.get("display_transform"),
                info.get("display_width"),
                info.get("display_height"),
                layer.item.shape[1],
                layer.item.shape[0],
                self.scene_origin_x,
                self.scene_origin_y
            )
            self.rasterUpdated.emit(lid, True)

    def choose_display_factor(self, width):
        if width >= 10000:
            factor = 16
        elif width > 8000:
            factor = 8
        elif width > 5000:
            factor = 4
        else:
            factor = 2
        return factor
    
    def read_overview(self, src, factor): 
        from rasterio.enums import Resampling
        h, w = src.shape
        out_h = max(1, h // factor)
        out_w = max(1, w // factor)
        if src.count >= 3:
            data = src.read(
                [1, 2, 3],
                out_shape=(3, out_h, out_w)
            )
            # logger.info(f"RAM: {process.memory_info().rss / 1024**2:.1f} MB")
        else:
            data = src.read(
                out_shape=(src.count, out_h, out_w),
                resampling=Resampling.nearest
            )
            # logger.info(f"RAM: {process.memory_info().rss / 1024**2:.1f} MB")
        
        mask = src.read_masks(
            1, 
            out_shape=(out_h, out_w)
        )

        return data, mask
    
    def build_qtransform(self, transform, w, h, out_w, out_h, origin_x, origin_y):
        from rasterio.transform import Affine
        t1 = transform * Affine.scale(
            w / out_w,
            h / out_h   
        ) 
        final_transform = QTransform(
            t1.a, t1.b, 
            t1.d, t1.e, 
            t1.c - origin_x, t1.f - origin_y
        )
        return final_transform


    def prepare_raster(self, bands, mask, count, nodata, isPrediction, legend=None):
        # Handle native alpha 4-channel RGBA
        if count == 4 and mask is None:
            alpha = bands[3].astype(np.uint8)
            bands = bands[:3]
            count = 3
        elif mask is not None:
            alpha = mask.astype(np.uint8)
        else:
            h, w = bands.shape[1:]
            alpha = np.full((h, w), 255, dtype=np.uint8)
        if legend:
            legend_new = {int(k): v for k, v in legend.items()}
            self.raster_legend = legend_new
        # logger.info(f"bands.nbytes={bands.nbytes/1024**2:.1f} MB")
        # logger.info(f"alpha.nbytes={alpha.nbytes/1024**2:.1f} MB")
        if isPrediction and self.raster_legend:
            h, w = bands.shape[1:]
            img = np.zeros((h, w, 4), dtype=np.uint8)
            img_data = bands[0]
            colors = {0: np.array([0,0,0,0], dtype=np.uint8)}
            for key, val in self.raster_legend.items():
                colors[key] = np.array(val["color"] + [255], dtype=np.uint8)

            for val, color in colors.items():
                img[img_data == val] = color
                
        # If not prediction result 
        else: 
            if count >= 3:
                # Ambil 3 band pertama untuk visualisasi RGB
                img_data = bands[:3]
                if np.issubdtype(img_data.dtype, np.floating):
                    rgb = np.transpose(
                        (img_data.astype(np.uint16) >> 8), 
                        (1, 2, 0)
                        ).astype(np.uint8) 
                elif img_data.dtype == np.uint16:
                    rgb = np.transpose(
                        (img_data >> 8), 
                        (1, 2, 0)
                        ).astype(np.uint8)  
                else:
                    rgb = np.transpose(
                        img_data,
                        (1, 2, 0)
                        ).astype(np.uint8)
                # logger.info(f"rgb.nbytes={rgb.nbytes/1024**2:.1f} MB")
                # logger.info(f"rgb:{rgb.flags['C_CONTIGUOUS']}")
                # logger.info(f"rgb:{rgb.flags['OWNDATA']}")
                img = np.dstack((rgb, alpha))
                del bands
                del rgb
                # logger.info(f"RAM: {process.memory_info().rss / 1024**2:.1f} MB")
            else:
                # Jika hanya 1 atau 2 band, tampilkan sebagai grayscale
                if np.issubdtype(bands.dtype, np.floating):
                    gray = (bands[0] * 255).astype(np.uint8)
                elif bands.dtype == np.uint16:
                    gray = (bands[0] >> 8).astype(np.uint8)
                else:
                    gray = bands[0].astype(np.uint8)
                # logger.info(f"gray.nbytes={gray.nbytes/1024**2:.1f} MB")
                if nodata is not None:
                    alpha = np.where(bands[0] == nodata, 0, alpha).astype(np.uint8)
                img = np.dstack((gray, gray, gray, alpha))
                del bands
                del gray
        # logger.info(f"img.nbytes={img.nbytes/1024**2:.1f} MB")
        img = np.ascontiguousarray(img)
        return img
            
    def add_raster(self, name, layer_id, path, hooks=None):
        import rasterio as rio
        import json
        from rasterio.transform import Affine
        from app.worker import OperationCancelledError
        helper = WorkerHelper(hooks)
        filename = name.split(".")[0]
        isGeotiff = not path.lower().endswith((".png", ".jpg", ".jpeg")) # Geotiff flag
        temp_dir = AppPaths.TEMP / filename
        temp_dir.mkdir(parents=True, exist_ok=True)
        # logger.info(f"RAM: {process.memory_info().rss / 1024**2:.1f} MB")
        logger.info(f"==== ADD RASTER {name} TO LAYER: {layer_id} ====")
        # logger.info("Load raster...")
        helper.progress(20, "Loading raster...")
        if helper.cancelled(): raise OperationCancelledError("User cancelled")
        helper.progress(45, "Fetching raster metadata...")
        if helper.cancelled(): raise OperationCancelledError("User cancelled")
        with rio.open(path) as src:
            # Metadata raster
            dtype = src.dtypes[0]
            count = src.count
            crs = src.crs
            nodata = src.nodata
            tag = src.tags()
            h, w = src.shape
            if isGeotiff:
                t = src.transform
                pixel_width = abs(t.a)
                xmin, ymin, xmax, ymax = self.get_raster_extent(t, w, h)

                if self.scene_origin_x is None:
                    self.scene_origin_x = xmin
                    self.scene_origin_y = ymax

                display_factor = self.choose_display_factor(w)
                logger.info(f"Display factor: {display_factor}")
                if dtype == 'uint8' and 1 not in self.OVERVIEW_FACTORS:
                    self.OVERVIEW_FACTORS.insert(0, 1)
                helper.progress(55, "Generating overview...")
                # Buat overview
                current_build = 1
                for build_factor in list(self.OVERVIEW_FACTORS):
                    if helper.cancelled(): raise OperationCancelledError("User cancelled")
                    total_build = len(self.OVERVIEW_FACTORS)
                    rel_progress = 60 + int((current_build/total_build) * 25)
                    helper.progress(rel_progress, f"Generating overview {current_build}/{total_build}")
                    preview, mask = self.read_overview(src, build_factor)
                    np.savez_compressed(
                        str(temp_dir/f"f{build_factor}.npz"),
                        bands=preview,
                        mask=mask
                )
                    current_build += 1
                # Load overview yang dibuat
                helper.progress(85, "Loading overview...")
                if 1 in self.OVERVIEW_FACTORS:
                    self.OVERVIEW_FACTORS.remove(1)
                if helper.cancelled(): raise OperationCancelledError("User cancelled")
                arr = np.load(str(temp_dir/f"f{display_factor}.npz"))
                bands = arr.get("bands")
                mask = arr.get("mask")
                # Transform overview
                out_h = bands.shape[1]
                out_w = bands.shape[2]
            else:   
                # Standard Image (PNG/JPG/JPEG)
                helper.progress(70, "Reading standard image...")
                t = Affine.translation(0, 0)
                pixel_width = 1.0
                xmin, ymin, xmax, ymax = 0, -h, w, 0
                display_factor = 1
                
                if self.scene_origin_x is None:
                    self.scene_origin_x = xmin
                    self.scene_origin_y = ymax
                
                bands = src.read()
                
                if count == 4:
                    mask = bands[3]
                else:
                    try:
                        mask = src.dataset_mask()
                    except Exception:
                        mask = np.full((h, w), 255, dtype=np.uint8)

                out_h, out_w = h, w

            if helper.cancelled(): raise OperationCancelledError("User cancelled")

            # Build QTransform
            qt_transform = self.build_qtransform(
                t,
                w,
                h,
                out_w,
                out_h,
                self.scene_origin_x,
                self.scene_origin_y
            )
            # Check if geotiff is prediction result
            isPrediction = "PREDICTION" in tag and dtype == 'uint8'
            legend_dict = {}
            stats_dict = {}
            if isPrediction:
                if "LEGEND" in tag:
                    legend_dict = json.loads(tag.get("LEGEND", "{}"))
                if "STATS" in tag:
                    stats_dict = json.loads(tag.get("STATS", "{}"))
        img = self.prepare_raster(bands, mask, count, nodata, isPrediction, legend_dict)
        helper.progress(95, "Saving raster metadata...")
        if helper.cancelled(): raise OperationCancelledError("User cancelled")
        # Simpan info raster
        info = {
            "Name": name,
            "Source": path,
            "Legend": legend_dict,
            "Stats": stats_dict,
            "Dtype": str(dtype),
            "CRS": crs.to_string() if crs and isGeotiff else "Non-Georeferenced",
            "Count": count,
            "Nodata": nodata,
            "transform": t,
            "height": h,
            "width": w,
            "display_transform": t,
            "display_width": w,
            "display_height": h,
            "Res": f"{pixel_width:.4f} m ({pixel_width*100:.1f} cm/px)" if crs and isGeotiff else f"1 px",
            "current_factor": display_factor,
            "base_factor": display_factor,
            "cache_dir": str(temp_dir) if isGeotiff else None,
            "is_prediction": isPrediction
        }
        self.raster_info[layer_id] = info
        self.crsDetected.emit(crs.to_string() if crs else "Non-Georeferenced")
        helper.progress(100, "Done")
        # Assign layer
        layer = Layer(
            sid=layer_id,
            name=name,
            item=img,
            layer_type="raster",
            metadata=info,
            extent=(xmin, ymin, xmax, ymax),
            qtransform=qt_transform
        )
        return layer

    def choose_factor(self, base_factor, ratio):
            import math
            if ratio >=1:
                zoom_steps = math.ceil(math.log2(ratio))
            else:
                zoom_steps = -math.ceil(math.log2(1/ratio))   
            steps = {
                1: 0,
                2: 1,
                4: 2,
                8: 3,
                16: 4
            }
            level = steps[base_factor] - zoom_steps
            level = max(0, min(4, level))
            return [1, 2, 4, 8, 16][level]

    def update_viewport_raster(self, layer_id):
        logger.info("Get viewport size...")
        layer = self.layer_manager.get_layer(layer_id)
        if not layer:
            return
        import time
        start = time.perf_counter()
        canvas = self.viewport_canvas
        info = layer.metadata
        count = info.get("Count")
        nodata = info.get("Nodata")
        prediction = info.get("is_prediction")
        world_left  = canvas.left() + self.scene_origin_x
        world_top   = canvas.top() + self.scene_origin_y
        world_right = canvas.right() + self.scene_origin_x
        world_bot   = canvas.bottom() + self.scene_origin_y
        # logger.info(f"Viewport rect={canvas}")
        # logger.info(f"Scene viewport:"
        #             f"x={round(canvas.x(),2)},"
        #             f"y={round(canvas.y(),2)},"
        #             f"w={round(canvas.width(),2)},"
        #             f"h={round(canvas.height(),2)}"
        #             )
        logger.info(f"Raster size: w={info['width']}, h={info['height']}")
        import rasterio as rio
        from rasterio.windows import Window
        logger.info("Window_reading...")
        with rio.open(info.get("Source")) as src:
            row0, col0 = src.index(world_left, world_top)
            row1, col1 = src.index(world_right, world_bot)
            x = min(col0, col1)
            y = min(row0, row1)
            w = abs(col1 - col0)
            h = abs(row1 - row0)
            window = Window(x, y, w, h)
            if src.count >= 3:
                bands = src.read([1, 2, 3], window=window)
            else:
                bands = src.read(window=window)
            mask = src.read_masks(1, window=window)
            window_transform = rio.windows.transform(window, src.transform)
            
            info["display_transform"] = window_transform
            info["display_width"] = w
            info["display_height"] = h

            # logger.info(f"Window x={x}, y={y}, w={w}, h={h}")
            # logger.info(f"Pixel size={round(window_transform.a,2)}")

        logger.info(f"Window read shape: {bands.shape}")
        logger.info(f"Window mask shape: {mask.shape}")
        elapsed = time.perf_counter() - start
        logger.info(f"window reading time -> {elapsed:.3f}s")
        img = self.prepare_raster(
            bands,
            mask,
            count,
            nodata,
            prediction
        )
        final_transform = self.build_qtransform(
            window_transform,
            w,
            h,
            bands.shape[2],
            bands.shape[1],
            self.scene_origin_x,
            self.scene_origin_y
        )
        # logger.info(f"img shape={img.shape if hasattr(img,'shape') else 'unknown'}")
        # logger.info(f"Window transform={window_transform}")
        # logger.info(f"Final window transform={final_transform}")
        # logger.info("Show window_reading result...")
        # Update layer img + transform
        layer.item = img
        layer.qtransform = final_transform
        self.rasterUpdated.emit(
            layer_id,
            False
        )

    def reload_overview(self, layer_id, factor):
        from pathlib import Path
        layer = self.layer_manager.get_layer(layer_id)
        info = layer.metadata
        prediction = info.get("is_prediction")
        count = info.get("Count")
        nodata = info.get("Nodata")
        dtype = info.get("Dtype")
        current_transform = info.get("transform")
        current_w = info.get("width")
        current_h = info.get("height")
        info["display_transform"] = current_transform
        info["display_width"] = current_w
        info["display_height"] = current_h
        if factor == 1 and dtype == 'uint16':
            self.update_viewport_raster(layer_id)
            logger.info(f"Reload layer {layer_id} -> f{factor} -> window_reading")
        elif factor != info.get("current_factor"):
            cache_dir = Path(info.get("cache_dir"))
            arr = np.load(str(cache_dir/f"f{factor}.npz"))
            bands = arr.get("bands")
            mask = arr.get("mask")

            img = self.prepare_raster(
                bands,
                mask,
                count,
                nodata,
                prediction
            )

            img_transform = self.build_qtransform(
                current_transform,
                current_w,
                current_h,
                bands.shape[2],
                bands.shape[1],
                self.scene_origin_x,
                self.scene_origin_y
            )
                
            # Update layer img + transform 
            layer.item = img
            layer.qtransform = img_transform
            self.rasterUpdated.emit(
                layer_id,
                False
            )
            logger.info(f"Reload layer {layer_id} -> f{factor} {bands.shape}")
        info["current_factor"] = factor     

    def update_overview_level(self):
        zoom_ratio = self.viewport_zoom_ratio
        # logger.info("UPDATE OVERVIEW LEVEL!")  
        for layer_id, info in self.raster_info.items():
            # logger.info(f"==== INFO LAYER {layer_id} ====")
            base_factor = info.get("base_factor")
            current_factor = self.choose_factor(base_factor, zoom_ratio)
            # logger.info(f"layer={layer_id}, factor={current_factor}")
            self.reload_overview(layer_id, current_factor)

    def trigger_update_overview_level(self):
        self.debounce_timer.stop() # Stop timer
        self.debounce_timer.start(self.debounce_delay) # Mulai ulang

    def set_viewport_state(self, canvas, zoom_ratio):
        self.viewport_canvas = canvas
        self.viewport_zoom_ratio = zoom_ratio

    def remove_raster(self, layer_id):
        self.raster_info.pop(layer_id, None)
        if not self.raster_info:
            self.raster_legend = None
            self.scene_origin_x = None
            self.scene_origin_y = None

    