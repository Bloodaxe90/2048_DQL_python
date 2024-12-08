import numpy as np
from PySide6.QtCore import QObject, Slot, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QWidget, QLabel
from src.UI.tile import Tile
from src.game_modes.default import Default


class Controller(QObject):

    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        self.terminal_label: QLabel = self.scene.findChild(QLabel, "LTerminal")
        self.board: list = [
            [Tile(self.scene.findChild(QLabel, f"L{j}{i}")) for j in range(4)]
            for i in range(4)
        ]

        self.stop: bool= False

        self.default = Default(self)

    @Slot()
    def key_pressed(self, event: QKeyEvent):
        if not self.stop:
            match event.key():
                case Qt.Key_Up:
                    self.default.play("UP")
                case Qt.Key_Down:
                    self.default.play("DOWN")
                case Qt.Key_Left:
                    self.default.play("LEFT")
                case Qt.Key_Right:
                    self.default.play("RIGHT")
        elif event.key() == Qt.Key_Space:
            self.reset()
            self.default.reset()



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




