from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog


class SaveFileDialog(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 300, 200)
        self.setWindowTitle('Save File Dialog')

        # Create a button to open the file dialog
        btn = QPushButton('Save File', self)
        btn.resize(btn.sizeHint())
        btn.move(100, 80)
        btn.clicked.connect(self.showFileDialog)

    def showFileDialog(self):
        fileName, _ = QFileDialog.getSaveFileName(self, 'Save File', '', 'All Files (*);;Text Files (*.txt)')
        if fileName:
            print(f"File saved to {fileName}")


if __name__ == '__main__':
    app = QApplication([])
    window = SaveFileDialog()
    window.show()
    app.exec_()
