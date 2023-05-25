import copy
import os.path
import sys
from datetime import datetime, timedelta

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressBar, QCheckBox, QMainWindow, QWidget, QHBoxLayout, \
    QLabel, QVBoxLayout, QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy, QScrollArea, QPushButton, \
    QListView, QStatusBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QStringListModel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from results import *
from video_analysis import *

ICON_PATH = "assets/icon.png"
ICON_PATH = os.path.join(sys._MEIPASS, ICON_PATH) if hasattr(sys, '_MEIPASS') else ICON_PATH # Pyinstaller handling


class ExportProblems(QMainWindow):
    def __init__(self, successful: list, failed: list):
        super().__init__()  # call the constructor of the parent class
        self.setWindowTitle("Export Status")

        # Set window icon
        self.setWindowIcon(QIcon(ICON_PATH))

        # Create successful names list view
        self.success_list_view = QListView()
        self.success_list_view.setModel(QStringListModel(successful))
        self.success_list_view.setEditTriggers(QListView.NoEditTriggers)

        # Create failed names list view
        self.failed_list_view = QListView()
        self.failed_list_view.setModel(QStringListModel(failed))
        self.failed_list_view.setEditTriggers(QListView.NoEditTriggers)

        # Create status bar
        self.statusBar = QStatusBar()
        self.statusBar.showMessage("Finished")

        # Create a vertical layout for the two list views and add them to a widget
        layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(layout)

        # Add successful names title
        success_label = QLabel("Successful")
        success_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(success_label)
        layout.addWidget(self.success_list_view)

        # Add failed names title
        failed_label = QLabel("Failed")
        failed_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(failed_label)
        layout.addWidget(self.failed_list_view)

        # Set the central widget of the main window to the widget with the list views
        self.setCentralWidget(widget)

        # Set the status bar of the main window
        self.setStatusBar(self.statusBar)

        # Resize the main window to fit the contents of the widget
        self.adjustSize()


class PersonTemplate(QMainWindow):
    def __init__(self, person: NewsAnalysis.DurationsDB.Duration):
        super().__init__()
        face_info = person.face_info
        self.name = "" if face_info.name is None else face_info.name
        self.original_name = "" if self.name is None else self.name
        self.image = face_info.profile_image_data
        self.duration = utils.secondsStr(person.duration)
        self.timestamps = person.getTimeStampsFormatted()

        self.init_ui()

    def init_ui(self):

        # Set up the main window
        self.setWindowTitle("Person Template")

        # Set window icon
        self.setWindowIcon(QIcon(ICON_PATH))

        # Set up the central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Set up the layout for the central widget
        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        # Add the person's image
        image_label = QLabel()
        qimage = QImage(self.image.data, self.image.shape[1], self.image.shape[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimage)
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

        self.name_edit = QLineEdit(self.name)
        # name_edit.setFixedWidth(200)
        name_layout.addWidget(self.name_edit)

        # Add a mark if the name has been changed
        self.name_changed_label = QLabel()
        self.name_changed_label.setStyleSheet("color: red")
        self.name_changed_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.name_changed_label)

        self.name_edit.textChanged.connect(self.check_name)

        # Add checkbox to mark if person should be exported to database
        export_layout = QHBoxLayout()
        self.export_checkbox = QCheckBox("Export to Database")
        export_layout.addWidget(self.export_checkbox)
        info_layout.addLayout(export_layout)

        # Add the person's duration and time taken
        duration_layout = QHBoxLayout()
        duration_label = QLabel(f"Duration: {self.duration}")
        duration_layout.addWidget(duration_label)

        info_layout.addLayout(duration_layout)

        # Add a table to display start and end times
        time_table = QTableWidget()
        time_table.setColumnCount(2)
        time_table.setHorizontalHeaderLabels(["Start", "End"])

        for (start_time_str, end_time_str) in self.timestamps:
            start_time_item = QTableWidgetItem(start_time_str)
            start_time_item.setTextAlignment(Qt.AlignCenter)
            start_time_item.setFlags(start_time_item.flags() ^ Qt.ItemIsEditable)  # remove ItemIsEditable flag

            end_time_item = QTableWidgetItem(end_time_str)
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

    def check_name(self):
        if self.name_edit.text() != self.original_name:
            self.name_changed_label.setText("(Name changed)")
        else:
            self.name_changed_label.setText("")
        self.name = self.name_edit.text()


class DurationsWindow(QMainWindow):
    def __init__(self, durations: NewsAnalysis.DurationsDB):
        super().__init__()

        self.problems_window = None

        self.durations = durations
        self.persons = list()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Review and Export")

        # Set window icon
        self.setWindowIcon(QIcon(ICON_PATH))

        # Create top layout
        top_layout = QHBoxLayout()

        # Create file path input and export facial database button on the right
        export_layout = QVBoxLayout()
        export_label = QLabel("Export path:")
        export_layout.addWidget(export_label)
        export_layout.setAlignment(export_label, Qt.AlignTop)
        export_file_layout = QHBoxLayout()
        self.export_file_path = QLineEdit()
        self.export_file_path.setPlaceholderText("Enter export file path...")
        export_file_layout.addWidget(self.export_file_path)
        export_button = QPushButton("Choose Database")
        export_button.clicked.connect(self.select_file_path)
        export_file_layout.addWidget(export_button)
        use_default_button = QPushButton("Use Default")
        use_default_button.clicked.connect(self.create_default)
        export_file_layout.addWidget(use_default_button)
        export_layout.addLayout(export_file_layout)

        # Create Select All/Deselect All buttons on the right
        select_buttons_layout = QHBoxLayout()
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(lambda: self.export_all(True))
        select_buttons_layout.addWidget(select_all_button)
        deselect_all_button = QPushButton("Deselect All")
        deselect_all_button.clicked.connect(lambda: self.export_all(False))
        select_buttons_layout.addWidget(deselect_all_button)

        # Add the export and select buttons to the right layout
        export_layout.addLayout(select_buttons_layout)

        export_database_button = QPushButton("Export")
        export_database_button.clicked.connect(self.export)
        export_layout.addWidget(export_database_button)
        export_layout.setAlignment(export_database_button, Qt.AlignRight)

        # Add the export layout to the top layout
        top_layout.addLayout(export_layout)

        # Set up the scrollable section
        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setMinimumSize(800, 300)

        # Add the persons to the scrollable section
        for duration in self.durations:
            self.add_person(duration)

        # Set the central widget of the main window
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.addLayout(top_layout)
        central_layout.addWidget(self.scroll_area)
        self.setCentralWidget(central_widget)

    def select_file_path(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Facial Database",
            "",
            "Facial Database Files (*.dat *.pkl)",
            options=options,
        )
        if file_path:
            self.export_file_path.setText(file_path)
        else:
            QMessageBox.warning(self, 'Error', 'Please select a valid path.')
            return

    def create_default(self):
        self.export_file_path.setText("database/face_database.dat")

    def add_person(self, person: NewsAnalysis.DurationsDB.Duration):
        item = PersonTemplate(person)
        self.persons.append(item)
        self.scroll_layout.addWidget(item)

    def getExports(self):
        exports = list()
        for i, person_template in enumerate(self.persons):
            export_bool = person_template.export_checkbox.isChecked()
            if export_bool:
                name = person_template.name
                face_info = self.durations.durations[i].face_info

                face_info_copy = copy.deepcopy(face_info)
                face_info_copy.name = name if name != "" else None

                exports.append(face_info_copy)

        return exports

    def export(self):
        export_database_path = self.export_file_path.text()
        # export_database_path = get_absolute_path(export_database_path)

        directory = os.path.dirname(export_database_path)

        if not directory:
            QMessageBox.warning(self, 'Error', f"Please select a valid export path.")
            # QMessageBox.warning(self, 'Error', f"Please select a valid export path. '{os.access(directory, os.W_OK)}' '{directory}'")
            return

        os.makedirs(directory, exist_ok=True)

        export = self.getExports()
        export = [fi for fi in export if fi.name is not None and fi.name != ""]

        try:
            export_database = FaceInfoDatabase(export_database_path, load_available=True)
            problems = export_database.inputFaceInfo(export, duplicate_names=False)
            export_database.save()
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error!\n{str(e)}')
            return

        failed_names = [face_info.name for face_info in problems]
        successful_names = [face_info.name for face_info in export if face_info.name not in failed_names]
        self.problems_window = ExportProblems(
            successful_names,
            failed_names,
        )
        self.problems_window.show()

        return problems

    def export_all(self, select: bool):
        for person_template in self.persons:
            person_template.export_checkbox.setChecked(select)

    def edit_name(self, person_index, new_name):
        person = self.persons[person_index]
        if new_name != person.name:
            person.edited = True
        person.name = new_name

        # Update the name label in the corresponding section
        section_widget = self.scroll_layout.itemAt(person_index).widget()
        name_label = section_widget.layout().itemAt(1).widget()
        if person.edited:
            self.name_changed_label.setText("(Name changed)")
        else:
            self.name_changed_label.setText("")
        name_label.setText(new_name)


class AnalysisThread(QThread):
    progress_update = pyqtSignal(int, str, str)
    analysis_complete = pyqtSignal(str)

    def __init__(self, video_path, analysis_dir, show=False, facial_database=None):
        super().__init__()
        self.show = show

        self.na = NewsAnalysis(video_path, analysis_dir, overwrite_base_folder=True,
                               skip_seconds_check=1, resolution=(640, 360), tracker_type="KCF",
                               face_info_db_path=facial_database)

        self.start_time = None

    def run(self):
        self.start_time = time.time()
        self.na.analyseVideo(processes=0, show_frame=self.show, callback_update=self.update_progress,
                             show_progress=False)
        self.analysis_complete.emit(self.na._base_output_folder)

    def update_progress(self, progress, total_frames):
        progress_perc = (progress / total_frames) * 100
        progress_perc_int = int(round(progress_perc))

        # Calculate estimated time remaining
        elapsed_time = time.time() - self.start_time
        elapsed_time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
        if progress == 0:
            estimated_time = None
            estimated_time_str = ""
        else:
            estimated_time = elapsed_time * ((total_frames - progress) / progress)
            estimated_time_str = time.strftime('%H:%M:%S', time.gmtime(estimated_time))

        self.progress_update.emit(progress_perc_int, elapsed_time_str, estimated_time_str)


class VideoPathWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.review_export_window = None

        self.video_path = None
        self.analysis_path = None
        self.na = None

        self.analysis_basedir = "analysis"
        self.min_label_size = (100, 20)
        self.min_textbox_size = (300, 20)
        self.min_button_size = (90, 20)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Video Path Window')
        # self.setToolTip('Enter the path to the video file you want to analyse')

        # Set window icon
        self.setWindowIcon(QIcon(ICON_PATH))

        vbox = QtWidgets.QVBoxLayout(self)

        # Video input
        analysis_path_groupbox = QtWidgets.QGroupBox('Video Input')
        vbox_analysis_path = QtWidgets.QVBoxLayout(analysis_path_groupbox)
        vbox.addWidget(analysis_path_groupbox)

        # Video path
        video_hbox1 = QtWidgets.QHBoxLayout()
        vbox_analysis_path.addLayout(video_hbox1)

        self.video_path_label = QtWidgets.QLabel('Video Path:', self)
        self.video_path_label.setMinimumSize(*self.min_label_size)
        video_hbox1.addWidget(self.video_path_label)

        self.video_path_textbox = QtWidgets.QLineEdit(self)
        # self.video_path_textbox.textChanged.connect(self.change_video_path)
        self.video_path_textbox.setMinimumSize(*self.min_textbox_size)
        self.video_path_textbox.setToolTip('Enter the path to the video file you want to analyse')
        video_hbox1.addWidget(self.video_path_textbox)

        self.select_video_path_button = QtWidgets.QPushButton('Select Video', self)
        self.select_video_path_button.setMinimumSize(*self.min_button_size)
        self.select_video_path_button.setToolTip('Select the video file you want to analyse')
        video_hbox1.addWidget(self.select_video_path_button)

        self.select_video_path_button.clicked.connect(self.select_video_path)

        # Facial database
        video_hbox2 = QtWidgets.QHBoxLayout()
        vbox_analysis_path.addLayout(video_hbox2)

        self.analysis_db_label = QtWidgets.QLabel('Facial Database:', self)
        self.analysis_db_label.setMinimumSize(*self.min_label_size)
        video_hbox2.addWidget(self.analysis_db_label)

        self.analysis_db_textbox = QtWidgets.QLineEdit(self)
        self.analysis_db_textbox.setMinimumSize(*self.min_textbox_size)
        self.analysis_db_textbox.setToolTip('Enter the desired database path (optional)')
        video_hbox2.addWidget(self.analysis_db_textbox)

        self.select_analysis_db_button = QtWidgets.QPushButton('Select Directory', self)
        self.select_analysis_db_button.setMinimumSize(*self.min_button_size)
        self.select_analysis_db_button.setToolTip('Select the database directory (optional)')
        video_hbox2.addWidget(self.select_analysis_db_button)

        self.select_analysis_db_button.clicked.connect(self.select_analysis_db)

        # Output analysis directory
        video_hbox3 = QtWidgets.QHBoxLayout()
        vbox_analysis_path.addLayout(video_hbox3)

        self.analysis_dir_label = QtWidgets.QLabel('Analysis Output:', self)
        self.analysis_dir_label.setMinimumSize(*self.min_label_size)
        video_hbox3.addWidget(self.analysis_dir_label)

        self.analysis_dir_textbox = QtWidgets.QLineEdit(self)
        self.analysis_dir_textbox.setMinimumSize(*self.min_textbox_size)
        self.analysis_dir_textbox.setToolTip('Enter the path to the output analysis directory')
        video_hbox3.addWidget(self.analysis_dir_textbox)

        self.select_analysis_dir_button = QtWidgets.QPushButton('Select Directory', self)
        self.select_analysis_dir_button.setMinimumSize(*self.min_button_size)
        self.select_analysis_dir_button.setToolTip('Select the output analysis directory')
        video_hbox3.addWidget(self.select_analysis_dir_button)

        self.select_analysis_dir_button.clicked.connect(self.select_analysis_dir)

        # Analysis progress
        video_hbox4 = QtWidgets.QHBoxLayout()
        vbox_analysis_path.addLayout(video_hbox4)

        self.start_analysis_button = QtWidgets.QPushButton('Start Analysis', self)
        self.start_analysis_button.setMinimumSize(*self.min_button_size)
        self.start_analysis_button.setToolTip('Start analysing the selected video file')
        video_hbox4.addWidget(self.start_analysis_button)
        video_hbox4.setStretchFactor(self.start_analysis_button, 1)  # Set stretch factor of button to 1

        self.start_analysis_button.clicked.connect(self.start_analysis)

        self.show_toggle = QCheckBox('Show', self)
        self.show_toggle.setToolTip('Check to show the analysis as it is being performed')
        video_hbox4.addWidget(self.show_toggle)
        video_hbox4.setStretchFactor(self.show_toggle, 0)  # Set stretch factor of checkbox to 0

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setToolTip('Shows the progress of the analysis')
        video_hbox4.addWidget(self.progress_bar, 3)
        self.progress_bar.hide()

        self.estimated_time_label = QtWidgets.QLabel(self)
        self.progress_bar.setToolTip('Shows the estimated time left')
        video_hbox4.addWidget(self.estimated_time_label)
        video_hbox4.setStretchFactor(self.estimated_time_label, 0)  # Set stretch factor of label to 0

        # Analysis path
        analysis_vhbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(analysis_vhbox)

        self.analysis_path_label = QtWidgets.QLabel('Analysis Directory:', self)
        self.analysis_path_label.setMinimumSize(*self.min_label_size)
        analysis_vhbox.addWidget(self.analysis_path_label)

        self.analysis_path_textbox = QtWidgets.QLineEdit(self)
        # self.analysis_path_textbox.textChanged.connect(self.change_analysis_path)
        self.analysis_path_textbox.setMinimumSize(*self.min_textbox_size)
        self.analysis_path_textbox.setToolTip('Enter the path to the output analysis file')
        analysis_vhbox.addWidget(self.analysis_path_textbox)

        self.select_analysis_path_button = QtWidgets.QPushButton('Select Analysis', self)
        self.select_analysis_path_button.setMinimumSize(*self.min_button_size)
        self.select_analysis_path_button.setToolTip('Select the output analysis file')
        analysis_vhbox.addWidget(self.select_analysis_path_button)
        self.select_analysis_path_button.clicked.connect(self.select_analysis_path)

        self.submit_analysis_path_button = QtWidgets.QPushButton('Submit', self)
        self.submit_analysis_path_button.setMinimumSize(*self.min_button_size)
        self.submit_analysis_path_button.setToolTip('Submit analysis path')
        analysis_vhbox.addWidget(self.submit_analysis_path_button)
        self.submit_analysis_path_button.clicked.connect(self.submit_analysis)

        # Review and export
        revexp_vhbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(revexp_vhbox)

        self.review_export_button = QtWidgets.QPushButton('Review/Export Database', self)
        self.review_export_button.setMinimumSize(*self.min_button_size)
        self.review_export_button.setToolTip(
            'Views the people who appeared, and also allows for the option to export the database')
        self.review_export_button.setEnabled(False)
        revexp_vhbox.addWidget(self.review_export_button)
        self.review_export_button.clicked.connect(self.review_export_window_open)

        # Timeline
        timeline_vhbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(timeline_vhbox)

        self.timeline_canvas = FigureCanvas(None)
        self.timeline_canvas.setMinimumSize(200, 300)
        timeline_vhbox.setStretchFactor(self.timeline_canvas, 1)
        timeline_vhbox.addWidget(self.timeline_canvas)

        # Disable these sets first
        self.analysis_dir_textbox.setReadOnly(True)
        self.select_analysis_dir_button.setEnabled(False)

        # Save timeline button
        timeline_save_vhbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(timeline_save_vhbox)
        self.save_timeline_button = QtWidgets.QPushButton('Save Timeline Image', self)
        self.save_timeline_button.setMinimumSize(*self.min_button_size)
        self.save_timeline_button.setToolTip(
            'Click to save the timeline image')
        self.save_timeline_button.setEnabled(False)
        timeline_save_vhbox.addWidget(self.save_timeline_button)
        self.save_timeline_button.clicked.connect(self.save_timeline_image)

        # Do sizes
        self.adjustSize()

    def select_video_path(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter('Video Files (*.mp4 *.avi *.mkv *.mov)')
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            self.reset_analysis()

            file_path = file_dialog.selectedFiles()[0]
            self.video_path_textbox.setText(file_path)

            # Set analysis dir
            folder_name = os.path.splitext(os.path.basename(file_path))[0]
            analysis_dir = utils.getDuplicateName(self.analysis_basedir, folder_name, "")
            self.analysis_dir_textbox.setText(analysis_dir)

        self.timeline_canvas.figure.clear()
        self.timeline_canvas.draw()
        self.save_timeline_button.setEnabled(False)

    # def change_video_path(self, text):
    #     if text != self.video_path:
    #         self.reset_analysis()

    def select_analysis_db(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter('Facial Databases (*.pkl *.dat)')
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.analysis_db_textbox.setText(file_path)

    def select_analysis_dir(self):
        dir_dialog = QFileDialog(self)
        dir_dialog.setFileMode(QFileDialog.Directory)
        if dir_dialog.exec_():
            dir_path = dir_dialog.selectedFiles()[0]
            self.analysis_dir_textbox.setText(dir_path)

    def start_analysis(self):
        video_path = self.video_path_textbox.text()
        facial_database = self.analysis_db_textbox.text()

        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, 'Error', 'Please select a valid video path.')
            return

        if facial_database and not os.path.exists(facial_database):
            QMessageBox.warning(self, 'Error', 'Please enter a valid Facial Database')
            return

        show = self.show_toggle.isChecked()
        analysis_dir = self.analysis_dir_textbox.text()

        self.timeline_canvas.figure.clear()
        self.timeline_canvas.draw()
        self.save_timeline_button.setEnabled(False)

        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.start_analysis_button.setEnabled(False)
        self.show_toggle.setEnabled(False)

        self.analysis_db_textbox.setReadOnly(True)
        self.select_analysis_db_button.setEnabled(False)
        self.analysis_dir_textbox.setReadOnly(True)
        self.select_analysis_dir_button.setEnabled(False)

        self.video_path_textbox.setReadOnly(True)
        self.select_video_path_button.setEnabled(False)

        self.analysis_thread = AnalysisThread(video_path, analysis_dir, show=show, facial_database=facial_database)
        self.analysis_thread.progress_update.connect(self.update_progress)
        self.analysis_thread.analysis_complete.connect(self.analysis_complete)
        self.analysis_thread.start()

    def update_progress(self, value, elapsed_time_str, estimated_time_str):
        self.progress_bar.setValue(value)
        self.estimated_time_label.setText(f"Estimated Time Left: {estimated_time_str}")

    def analysis_complete(self, analysis_path=None, submit_analysis=True):
        self.progress_bar.hide()
        self.show_toggle.hide()
        self.estimated_time_label.hide()

        self.start_analysis_button.setText('Analysis Complete')
        self.start_analysis_button.setEnabled(False)

        self.video_path_textbox.setReadOnly(False)
        self.select_video_path_button.setEnabled(True)

        if analysis_path is not None:
            self.analysis_path_textbox.setText(analysis_path)

        # Automatically submit analysis
        if submit_analysis:
            self.submit_analysis()

    def reset_analysis(self):
        self.progress_bar.setValue(0)
        self.progress_bar.hide()

        self.show_toggle.show()
        self.estimated_time_label.show()

        self.analysis_db_textbox.setReadOnly(False)
        self.select_analysis_db_button.setEnabled(True)

        self.analysis_dir_textbox.setReadOnly(False)
        self.select_analysis_dir_button.setEnabled(True)

        self.start_analysis_button.setText('Start Analysis')
        self.start_analysis_button.setEnabled(True)
        self.show_toggle.setEnabled(True)
        self.analysis_path_textbox.clear()

        self.estimated_time_label.clear()

        self.review_export_button.setEnabled(False)
        self.save_timeline_button.setEnabled(False)

    def setup_analysis(self):
        self.analysis_complete(submit_analysis=False)  # If input directory, set analysis as complete
        self.video_path_textbox.clear()
        self.analysis_dir_textbox.clear()

        # self.analysis_db_textbox.setReadOnly(True)
        # self.select_analysis_db_button.setEnabled(False)
        self.analysis_dir_textbox.setReadOnly(True)
        self.select_analysis_dir_button.setEnabled(False)

    def select_analysis_path(self):
        # file_dialog = QFileDialog(self)
        # file_dialog.setNameFilter('Analysis Files (*.pkl)')
        # file_dialog.setFileMode(QFileDialog.ExistingFile)
        # if file_dialog.exec_():
        #     file_path = file_dialog.selectedFiles()[0]
        #     self.analysis_path_textbox.setText(file_path)
        #
        #     self.submit_analysis()

        dir_dialog = QFileDialog(self)
        dir_dialog.setFileMode(QFileDialog.Directory)
        if dir_dialog.exec_():
            dir_path = dir_dialog.selectedFiles()[0]
            self.analysis_path_textbox.setText(dir_path)

            self.submit_analysis()

    # def change_analysis_path(self, text):
    #     if text != self.analysis_path:
    #         self.setup_analysis()

    def submit_analysis(self):
        analysis_path = self.analysis_path_textbox.text()

        if not analysis_path or not os.path.exists(analysis_path):
            QMessageBox.warning(self, 'Error', 'Please select a valid analysis directory')
            return

        self.setup_analysis()

        plt.close('all')

        try:
            self.na = NewsAnalysis.staticLoad(analysis_path)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error!\n{str(e)}')
            return

        try:
            fig = generateTimelineAnalysis(
                self.na,
                # "data/comparing_segments/tracking_database_2/01.08.2022 segment #0",
                # analysis_path,
                # validation_path="data/Validation2/01.08.2022 segment #0.xlsx",
                show=False
            )
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error!\n{str(e)}')
            return

        self.timeline_canvas.figure.clear()
        self.timeline_canvas.figure = fig
        self.timeline_canvas.draw()
        new_size = self.timeline_canvas.size()  # Get the current size of the canvas
        self.timeline_canvas.resize(new_size.width() + 1,
                                    new_size.height() + 1)  # Resize the canvas by adding 1 pixel to each dimension
        self.timeline_canvas.resize(new_size.width(),
                                    new_size.height())  # Resize the canvas back to its original size

        self.review_export_button.setEnabled(True)
        self.save_timeline_button.setEnabled(True)

    def save_timeline_image(self):
        fig = self.timeline_canvas.figure

        fileName, _ = QFileDialog.getSaveFileName(self, 'Save File', '', 'PNG Files (*.png);;All Files (*)')
        if fileName:
            fig.savefig(fileName)

    def review_export_window_open(self):
        # create the new main window and show it
        durations = self.na.getDurations()
        self.review_export_window = DurationsWindow(durations)
        self.review_export_window.show()


def main():
    app = QtWidgets.QApplication(sys.argv)
    video_path_window = VideoPathWindow()
    video_path_window.show()
    sys.exit(app.exec_())


def testReviewExportWindow():
    app = QtWidgets.QApplication(sys.argv)

    na = NewsAnalysis.staticLoad("data/comparing_segments/tracking_database_2/01.08.2022 segment #0")
    video_path_window = DurationsWindow(na.getDurations())
    video_path_window.show()
    sys.exit(app.exec_())


def testExportProblemsWindow():
    app = QtWidgets.QApplication(sys.argv)

    video_path_window = ExportProblems(["John", "Jane", "Bob"], ["John", "Jane", "Bob"])
    video_path_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    # testReviewExportWindow()
    # testExportProblemsWindow()
