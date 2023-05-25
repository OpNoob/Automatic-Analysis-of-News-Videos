import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLineEdit, QHeaderView,
    QCheckBox  # Add QCheckBox import
)


class PersonTemplate(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("Person Template")

        # Set up the central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Set up the layout for the central widget
        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        # Add the person's image
        image_label = QLabel()
        pixmap = QPixmap("screenshots/Frame #900.0.png")
        pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label.setPixmap(pixmap)
        layout.addWidget(image_label)

        # Add a vertical layout for the person's information
        info_layout = QVBoxLayout()
        layout.addLayout(info_layout)

        # Add the person's name
        name_layout = QHBoxLayout()
        info_layout.addLayout(name_layout)

        name_label = QLabel("Name:")
        name_layout.addWidget(name_label)

        name_edit = QLineEdit("Original Name")
        # name_edit.setFixedWidth(200)
        name_layout.addWidget(name_edit)

        # Add a mark if the name has been changed
        name_changed_label = QLabel()
        name_changed_label.setStyleSheet("color: red")
        name_changed_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(name_changed_label)

        def check_name():
            if name_edit.text() != "Original Name":
                name_changed_label.setText("(Name changed)")
            else:
                name_changed_label.setText("")

        name_edit.textChanged.connect(check_name)

        # Add checkbox to mark if person should be exported to database
        export_layout = QHBoxLayout()
        export_checkbox = QCheckBox("Export to Database")
        export_layout.addWidget(export_checkbox)
        info_layout.addLayout(export_layout)

        # Add the person's duration and time taken
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Duration: 10 hrs")
        duration_layout.addWidget(duration_label)

        info_layout.addLayout(duration_layout)

        # Add a table to display start and end times
        time_table = QTableWidget()
        time_table.setColumnCount(2)
        time_table.setHorizontalHeaderLabels(["Start", "End"])

        for i in range(10):
            start_time_item = QTableWidgetItem("08:00")
            start_time_item.setTextAlignment(Qt.AlignCenter)
            start_time_item.setFlags(start_time_item.flags() ^ Qt.ItemIsEditable)  # remove ItemIsEditable flag

            end_time_item = QTableWidgetItem("18:00")
            end_time_item.setTextAlignment(Qt.AlignCenter)
            end_time_item.setFlags(end_time_item.flags() ^ Qt.ItemIsEditable)  # remove ItemIsEditable flag

            row_position = time_table.rowCount()
            time_table.insertRow(row_position)
            time_table.setItem(row_position, 0, start_time_item)
            time_table.setItem(row_position, 1, end_time_item)

        # Set the horizontal header to stretch mode
        header = time_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        info_layout.addWidget(time_table)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PersonTemplate()
    window.show()
    sys.exit(app.exec_())
