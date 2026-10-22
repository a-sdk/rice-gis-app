from dataclasses import dataclass
from PyQt5.QtCore import QObject
from PyQt5.QtGui import QTransform
import logging

logger = logging.getLogger(__name__)

@dataclass
class Layer:
    sid: int
    name: str  
    item: object = None 
    layer_type: str = None      
    metadata: dict = None
    extent: tuple = None
    qtransform: QTransform = None
    is_visible: bool = None

class LayerManager(QObject):
    def __init__(self):
        super().__init__()
        self.layers = {}

    def add_layer(self, layer):
        self.layers[layer.sid] = layer
        # logger.info(
        #     f"Berhasil menambah "
        #     f"LAYER={layer.sid}, "
        #     f"NAME={layer.name}"
        # )

    def remove_layer(self, layer_id):
        layer = self.layers.pop(layer_id, None)
        if layer is not None:
            # logger.info(
            #     f"Berhasil menghapus "
            #     f"LAYER={layer.sid}, "
            #     f"NAME={layer.name}"
            # )
            return layer

    def get_layer(self, layer_id):
        if not layer_id:
            return 
        return self.layers.get(layer_id)

    def has_layer(self, layer_id):
        return layer_id in self.layers

    def count(self):
        return len(self.layers)
    
    def list_ids(self):
        ids = []
        for sid in self.layers:
            ids.append(sid)
        return ids
    
    def clear(self):
        self.layers.clear()

    def get_all_layers(self):
        return list(self.layers.values())
    
    @property
    def is_empty(self):
        return len(self.layers) == 0