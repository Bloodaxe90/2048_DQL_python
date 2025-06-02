import os.path

from PySide6.QtCore import QFile
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import *
from PySide6.QtUiTools import QUiLoader

from src.UI.controller import Controller


class Application(QMainWindow):

    def __init__(self):
        super().__init__()

        loader = QUiLoader()
        ui_file = QFile(f"{os.getcwd()}/resources/UI/2048.ui")
        if not ui_file.open(QFile.ReadOnly):
            print(f"Failed to open file: {ui_file.errorString()}")
            return

        self.ui = loader.load(ui_file, self)
        ui_file.close()
        self.setCentralWidget(self.ui)

        self.controller = Controller(self.ui)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        self.controller.key_pressed(event)
