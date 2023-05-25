from PyQt5.QtWidgets import QApplication, QTableView, QHeaderView, QVBoxLayout, QWidget, QMainWindow
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex
import sys

class TableWidget(QWidget):
    def __init__(self, data):
        super().__init__()

        # Create table view and model
        self.table_view = QTableView()
        self.model = TableModel(data)
        self.table_view.setModel(self.model)

        # Set the header to stretch
        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        # Add table view to layout
        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        self.setLayout(layout)

class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, index=QModelIndex()):
        return len(self._data)

    def columnCount(self, index=QModelIndex()):
        return len(self._data[0])

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return str(self._data[index.row()][index.column()])
        return None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    data = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']]
    table_widget = TableWidget(data)
    table_widget.show()
    sys.exit(app.exec_())
