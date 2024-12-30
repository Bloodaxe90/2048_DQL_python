import os.path

import numpy as np
from PySide6.QtCore import QObject, Slot, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QWidget, QLabel, QRadioButton
from src.UI.tile import Tile
from src.game_modes.default import Default
from src.game_modes.dq_ai import QAI


class Controller(QObject):

    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        self.terminal_label: QLabel = self.scene.findChild(QLabel, "LTerminal")
        self.board: list = [
            [Tile(self.scene.findChild(QLabel, f"L{j}{i}")) for j in range(4)]
            for i in range(4)
        ]
        self.default_radio: QRadioButton = self.scene.findChild(QRadioButton, "DefaultRadio")
        self.qai_radio: QRadioButton = self.scene.findChild(QRadioButton, "AiRadio")
        self.stop: bool= False

        self.default = Default(self)
        self.qai = QAI(self,
                       "/Users/Eric/PycharmProjects/2048/resources/saved_models/main_net/the_big_one.pth",
                       hidden_neurons=(1024, 1024, 1024, 1024))

    @Slot()
    def key_pressed(self, event: QKeyEvent):
        if not self.stop:
            if self.default_radio.isChecked():
                match event.key():
                    case Qt.Key_Up:
                        self.default.play("UP")
                    case Qt.Key_Down:
                        self.default.play("DOWN")
                    case Qt.Key_Left:
                        self.default.play("LEFT")
                    case Qt.Key_Right:
                        self.default.play("RIGHT")
            elif self.qai_radio.isChecked():
                if event.key() == Qt.Key_S:
                    self.qai.play()
        if event.key() == Qt.Key_Space:
            self.reset()
            if self.default_radio.isChecked():
                self.default.reset()
            elif self.qai_radio.isChecked():
                self.qai.reset()



    def set_board(self, environment: np.ndarray):
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                tile: Tile = self.board[i][j]
                value: int = int(environment[(i,j)])
                tile.change_value(value)


    def game_over(self, result: str):
        self.stop = True
        self.terminal_label.setVisible(True)
        if result == "W":
            self.terminal_label.setText("WIN")
        elif result == "L":
            self.terminal_label.setText("LOSE")
        else:
            raise ValueError("Invalid result")

    def reset(self):
        self.stop = False
        self.terminal_label.setVisible(False)




