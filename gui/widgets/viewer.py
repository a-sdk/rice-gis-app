from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, 
    QGraphicsItemGroup, QFrame
)
from PyQt5.QtGui import (
    QPixmap, QPainter, QImage,
    QPolygonF, QPen, QColor,
    QPainterPath, QPixmapCache
)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF, QEvent
import logging

logger = logging.getLogger(__name__)
class Viewer(QWidget):
    mouseMoved = pyqtSignal(float, float)
    viewportChanged = pyqtSignal()
    def __init__(self, layer_manager, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer = QGraphicsView()
        self.scene = QGraphicsScene()
        self.layer_manager = layer_manager
        self.layer_items = {}
        self.setMouseTracking(True)
        self.viewer.setMouseTracking(True)
        self.viewer.viewport().setMouseTracking(True)
        self.viewer.viewport().installEventFilter(self)
        self.viewer.setScene(self.scene)
        self.viewer.setRenderHint(QPainter.Antialiasing)
        self.viewer.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.viewer.setFrameShape(QFrame.NoFrame)
        self.viewer.setLineWidth(0)
        self.viewer.setStyleSheet("border: none;")
        self.viewer.setFocusPolicy(Qt.StrongFocus)
        self.viewer.horizontalScrollBar().valueChanged.connect(self.viewportChanged.emit)
        self.viewer.verticalScrollBar().valueChanged.connect(self.viewportChanged.emit)
        self.main_layout.addWidget(self.viewer)
        self.scene_origin_x = None
        self.scene_origin_y = None
        self.base_view_scale = None
        self.base_raster_size = {}
        self.list_ids = []
        self._zoom = 0
        self.active_tool = None

    def update_origin(self, x, y):
        self.scene_origin_x = x
        self.scene_origin_y = y

    def set_z_order(self, ordered_ids):
        for i, lid in enumerate(ordered_ids):
            if lid in self.layer_items:
                z = len(ordered_ids) - i
                self.layer_items[lid].setZValue(z)
                # logger.info(
                #     f"Set z-order: "
                #     f"id={lid} -> z-order={z}"
                #     )
        self.viewer.viewport().update()

    def update_z_order(self, ids):
        self.list_ids = ids
        self.set_z_order(self.list_ids)

    def get_viewport_state(self):
        # logger.info(
        #     f"EMIT viewport state: "
        #     f"base={self.base_view_scale} "
        # )
        scale = self.viewer.transform().m11()
        canvas = self.viewer.mapToScene(self.viewer.viewport().rect()).boundingRect()
        if self.base_view_scale is None:
            self.base_view_scale = scale 
        if self.base_view_scale == 1:
            self.base_view_scale *= scale
        if self.base_view_scale < 3:
            self.base_view_scale = scale * 1
        zoom_ratio = scale / self.base_view_scale
        # logger.info(
        #     f"scale={round(scale, 2)}, "
        #     f"base={round(self.base_view_scale, 2)}, "
        #     f"ratio={round(zoom_ratio, 2)}"
        # ) 
        return canvas, zoom_ratio
    
    def render_geotiff(self, layer_id, init_view=False):
        layer = self.layer_manager.get_layer(layer_id)
        info = layer.metadata
        img = layer.item
        transform = layer.qtransform
        w = info.get("width")
        h = info.get("height")
        dtype = info.get("Dtype")
        # Map top left 0, 0
        top_left = transform.map(QPointF(0, 0))
        bot_right = transform.map(QPointF(w, h))
        from PyQt5 import sip
        ptr = sip.voidptr(img.ctypes.data)
        qimg = QImage(ptr, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGBA8888)
        # logger.info(f"QImage generated: w={qimg.width()}, h={qimg.height()}")
        # logger.info(f"qimg.inbytes={qimg.sizeInBytes()/1024**2:.1f} MB")
        # logger.info(f"RAM: {process.memory_info().rss / 1024**2:.1f} MB")
        pixmap = QPixmap.fromImage(qimg)
        # logger.info(f"QPixmap generated: w={pixmap.width()}, h={pixmap.height()}")
        # logger.info(f"RAM sebelum del: {process.memory_info().rss / 1024**2:.1f} MB")
        del img
        del qimg
        import gc
        gc.collect()
        # logger.info(f"RAM setelah del: {process.memory_info().rss / 1024**2:.1f} MB")
        if layer_id in self.layer_items and self.layer_items[layer_id] != None:
            # logger.info(f"Layer {layer_id} found!")
            item = self.layer_items[layer_id]
            item.setPixmap(pixmap)
            # logger.info(f"Update pixmap scene...")
        else:
            # logger.info(f"Layer {layer_id} not found!")
            self.base_raster_size[layer_id] = (w, h)
            # Set bounding scene
            if len(self.base_raster_size) == 1:
                # logger.info("Mengatur sceneRect")
                self.scene.setSceneRect(
                    top_left.x(),
                    top_left.y(),
                    bot_right.x() - top_left.x(),
                    bot_right.y() - top_left.y()
                )
                # logger.info(f"SceneRect sesuai origin: {self.scene.sceneRect()}")
            item = self.scene.addPixmap(pixmap)
            # logger.info(f"Add pixmap to scene...")
            self.layer_items[layer_id] = item

        # logger.info(f"{self.layer_items}")
        item.setTransform(transform)
        item.setTransformationMode(Qt.FastTransformation)
        item.setAcceptHoverEvents(False)
        self.update_z_order(self.list_ids)

        if init_view and dtype == 'uint16':
            self.fit_to_view()

    def _draw_poly_feature(self, coords, group, color):
        points = [
            QPointF(
                x - self.scene_origin_x, 
                y - self.scene_origin_y
            ) 
            for x, y in coords
        ]
        polygon_item = self.scene.addPolygon(QPolygonF(points))
        pen = QPen(QColor(0, 0, 0, 150))
        pen.setWidth(1)
        pen.setCosmetic(True)
        polygon_item.setPen(pen)
        polygon_item.setBrush(QColor(*color))
        group.addToGroup(polygon_item)

    def _draw_line_feature(self, coords, group, color):
        path = QPainterPath()
        points = [
            QPointF(
                x - self.scene_origin_x,
                y - self.scene_origin_y
            ) 
            for x, y in coords
        ]
        if points:
            path.moveTo(points[0])
            for pt in points[1:]:
                path.lineTo(pt)
        
        line_item = self.scene.addPath(path)
        pen = QPen(QColor(*color))
        pen.setWidth(5)
        pen.setCosmetic(True)
        line_item.setPen(pen)
        group.addToGroup(line_item)

    def _draw_point_feature(self, x, y, group, color):
        r = 0.1
        sx = x - self.scene_origin_x
        sy = y - self.scene_origin_y
        color = (0, 0, 0, 200)
        point_item = self.scene.addEllipse(sx-r, sy-r, r*2, r*2)
        pen = QPen(QColor(0, 0, 0, 150))
        pen.setWidth(1)
        pen.setCosmetic(True)
        point_item.setPen(pen)
        point_item.setBrush(QColor(*color))
        group.addToGroup(point_item)

    def _render_vector(self, layer_id):
        old_group = self.layer_items.get(layer_id)
        if old_group is not None:
            self.scene.removeItem(old_group)

        group = QGraphicsItemGroup()
        self.scene.addItem(group)
        self.layer_items[layer_id] = group
        layer = self.layer_manager.get_layer(layer_id)
        if not layer or layer.item is None:
            return
        gdf = layer.item
        info = layer.metadata
        legend_dict = info.get("Legend")
        class_col = info.get("class_col")
        
        for _, feature in gdf.iterrows():
            geom = feature.geometry
            color = (255, 255, 116, 80)
            if legend_dict and class_col:
                val = feature[class_col]
                str_val = str(val)
                try:
                    num_val = float(val)
                    is_num = True
                except (ValueError, TypeError):
                    is_num = False
                if is_num and num_val < 1: # Cek jika hasil prediksi berupa regresi (0-1)
                    for key, data in legend_dict.items():
                        low, high = data["range"]
                        if low <= num_val < high:
                            color = data["color"] # Warna berdasarkan rentang nilai
                            # logger.info(f"Value: {val} | Label: {data['label']} | Color: {color}")
                            break
                elif str_val in legend_dict:
                    color = legend_dict[str_val]["color"]
                    label = legend_dict[str_val]["label"]
                    # logger.info(f"Value: {val} | Label: {label} | Color: {color}")

            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "Polygon":
                self._draw_poly_feature(geom.exterior.coords, group, color)
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    self._draw_poly_feature(poly.exterior.coords, group, color)
            elif geom.geom_type == "LineString":
                self._draw_line_feature(geom.coords, group, color)
            elif geom.geom_type == "Point":
                self._draw_point_feature(geom.x, geom.y, group, color)
        self.update_z_order(self.list_ids)
        self.fit_to_view()
        
    def set_visible(self, layer_id):
        layer = self.layer_manager.get_layer(layer_id)
        if not layer: return
        self.layer_items[layer_id].setVisible(layer.is_visible)
        
    def remove_item(self, layer_id):
        item = self.layer_items.pop(layer_id, None)
        self.base_raster_size.pop(layer_id, None)
        if item:
            self.scene.removeItem(item)
            if hasattr(item, 'deleteLater'):
                item.deleteLater()
            if hasattr(item, 'setPixmap'):
                item.setPixmap(QPixmap())
            item = None
        # logger.info(f"layer_items  : {list(self.layer_items.keys())}")
        if not self.layer_items:
            self.scene_origin_x = None
            self.scene_origin_y = None
            self.scene.clear()
        # Clear viewer and scene
        self.viewer.viewport().update()
        self.scene.update()
        QPixmapCache.clear()

    def apply_view_transform(self):
        t = self.viewer.transform()
        if t.m22() > 0:
            self.viewer.scale(1, -1)

    def fit_to_view(self):
        # logger.info("Fit to view clicked.")
        combined_rect = QRectF()
        has_visible_item = False

        for item in self.scene.items():
            if item.isVisible():
                rect = item.sceneBoundingRect()

                if rect.isEmpty():
                    continue

                if not has_visible_item:
                    combined_rect = rect
                    has_visible_item = True
                else:
                    combined_rect = combined_rect.united(rect)

        if has_visible_item and not combined_rect.isEmpty():
            self.viewer.fitInView(
                combined_rect,
                Qt.KeepAspectRatio
            )
        else:
            logger.warning("No visible item to fit to view.")
        self.apply_view_transform()

    def set_pan_mode(self, enabled: bool):
        if enabled:
            # logger.info("action_pan: ON")
            self.viewer.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            # logger.info("action_pan: OFF")
            self.viewer.setDragMode(QGraphicsView.DragMode.NoDrag)

    def zoom_in(self):
        self._zoom += 1
        self.viewer.scale(1.05, 1.05)
        self.viewportChanged.emit()

    def zoom_out(self):
        self._zoom -= 1
        self.viewer.scale(0.8, 0.8)
        self.viewportChanged.emit()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.viewportChanged.emit()
    
    def eventFilter(self, source, event):
        if source is self.viewer.viewport() and event.type() == QEvent.Type.MouseMove:
            # Ambil posisi dan konversi ke scene
            scene_pos = self.viewer.mapToScene(event.pos())
            if self.scene_origin_x is not None and self.scene_origin_y is not None:
                world_x = round(scene_pos.x() + self.scene_origin_x, 2)
                world_y = round(scene_pos.y() + self.scene_origin_y, 2)
            else:
                world_x = round(scene_pos.x(), 2)
                world_y = round(scene_pos.y(), 2)
            self.mouseMoved.emit(world_x, world_y)
        return super().eventFilter(source, event)

    def mousePressEvent(self, event): 
        if self.active_tool:
            scene_pos = self.viewer.mapToScene(event.pos())
            self.active_tool.mouse_press(scene_pos, event.button())       

    def keyPressEvent(self, event):
        self.active_tool.key_press(event.key())


