from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFrame,
    QTreeWidget, QTreeWidgetItem,
    QTreeWidgetItemIterator, QAbstractItemView, QMenu
)
import logging

logger = logging.getLogger(__name__)

class LayerPanel(QWidget):
    infoMsg = pyqtSignal(str)
    layerSelected = pyqtSignal(int)
    layerUpdated = pyqtSignal(list)
    layerRemoveRequested = pyqtSignal(int)

    def __init__(self, layer_manager, parent, viewer):
        super().__init__(parent)
        self.layer_manager = layer_manager
        self.viewer = viewer
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.tree = QTreeWidget()
        self.main_layout.addWidget(self.tree)
        self.layer_ids = []
        self.tree_items = {}
        self._setup_tree()
        self._connect_signals()

    def _setup_tree(self):
        self.tree.setFrameShape(QFrame.NoFrame)
        self.tree.setLineWidth(0)
        self.tree.setStyleSheet("border: none;")
        self.tree.setAcceptDrops(True)
        self.tree.setHeaderHidden(True)
        self.tree.setDragEnabled(True)
        self.tree.setDropIndicatorShown(True)
        self.tree.setDragDropMode(QAbstractItemView.InternalMove)
        self.tree.setDefaultDropAction(Qt.MoveAction)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)

    # Menambah layer
    def add_layer_item(self, layer_id, name):
        item = QTreeWidgetItem([name])
        item.setData(0, Qt.UserRole, layer_id)
        item.setCheckState(0, Qt.Checked)
        layer = self.layer_manager.get_layer(layer_id)
        if not layer: return 
        layer.is_visible = item.checkState(0) == Qt.Checked
        self.tree.insertTopLevelItem(0, item)
        self.infoMsg.emit(f"{name} loaded.")
        self.tree_items[layer_id] = item
        logger.info(f"Succesfully added '{name}', ID: {layer_id}")

    # Toggle visibility (checkbox)
    def on_item_changed(self, item):
        layer_id = item.data(0, Qt.UserRole)
        layer = self.layer_manager.get_layer(layer_id)
        if not layer: return
        layer.is_visible = item.checkState(0) == Qt.Checked
        logger.info(f"Layer {layer_id} visibility changed")
        self.viewer.set_visible(layer_id)

    # Update z-order
    def update_z_order(self):
        logger.info("z-order layer_panel updated")
        ids = []
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            lid = item.data(0, Qt.UserRole)

            if lid is not None:
               ids.append(lid)
            
            iterator += 1
        self.layer_ids = ids
        self.layerUpdated.emit(self.layer_ids)
        logger.info(f"Layer IDs (top down): {self.layer_ids}")

    # Context menu
    def _open_menu(self, pos):
        item = self.tree.itemAt(pos)
        logger.info("Layer menu opened")
        if not item:
            return

        if item and item.data(0, Qt.UserRole) is not None:
            menu = QMenu()
            remove_action = menu.addAction("Remove Layer")
            prop_action = menu.addAction("Properties")

            action = menu.exec_(self.tree.viewport().mapToGlobal(pos))

        if action == remove_action:
            self.req_remove_layer(item)

        if action == prop_action:
            self._show_properties(item)

    def on_item_clicked(self, item):
        layer_id = item.data(0, Qt.UserRole)
        self.layerSelected.emit(layer_id)
        # logger.info(f"Layer {layer_id} clicked")
        
    def req_remove_layer(self, item):
        layer_id = item.data(0, Qt.UserRole)
        self.layerRemoveRequested.emit(layer_id)

    def remove_layer_item(self, layer_id):
        logger.info(f"Layer {layer_id} removed")
        item = self.tree_items.pop(layer_id, None)
        if not item:
            return
        parent = item.parent() or self.tree.invisibleRootItem()
        parent.removeChild(item)
        self.infoMsg.emit(f"{item.text(0)} removed.")

    # Properties layer
    def _show_properties(self, item):
        layer_id = item.data(0, Qt.UserRole)
        layer = self.layer_manager.get_layer(layer_id)
        if layer:
            metadata = layer.metadata
            name = layer.name
            from gui.widgets.property_dialog import PropertyDialog 
            dialog = PropertyDialog(name, metadata, self)
            dialog.exec_()

    # Menghubungkan signal
    def _connect_signals(self):
        # Sinyal drag drop layer
        self.tree.model().rowsInserted.connect(self.update_z_order)
        self.tree.model().rowsRemoved.connect(self.update_z_order)
        self.tree.model().rowsMoved.connect(self.update_z_order)
        self.tree.itemChanged.connect(self.on_item_changed)
        # Menu right-click
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._open_menu)
        # Layer dipilih
        self.tree.itemClicked.connect(self.on_item_clicked)

    