import numpy as np
from PySide6.QtCore import QObject, Slot, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QWidget, QLabel
from src.UI.tile import Tile
from src.game.tweny48 import Twenty48


class Controller(QObject):

    def __init__(self, scene):
        super().__init__()
        self.scene = scene

        self.board: list = [
            [Tile(self.scene.findChild(QLabel, f"L{j}{i}")) for j in range(4)]
            for i in range(4)
        ]

        self.twenty48 = Twenty48()

    @Slot()
    def key_pressed(self, event: QKeyEvent):
        match event.key():
            case Qt.Key_Up:
                self.twenty48.play("UP")
            case Qt.Key_Down:
                self.twenty48.play("DOWN")
            case Qt.Key_Left:
                self.twenty48.play("LEFT")
            case Qt.Key_Right:
                self.twenty48.play("RIGHT")

        self.set_board(self.twenty48.environment)


    def set_board(self, environment: np.ndarray):
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                tile: Tile = self.board[i][j]
                value: int = int(environment[(i,j)])
                tile.change_value(value)





