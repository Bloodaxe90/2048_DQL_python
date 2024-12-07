import numpy as np


class Twenty48:

    def __init__(self):
        self.environment: np.ndarray = np.zeros((4,4))
        self.ACTIONS: list= ["UP", "DOWN", "LEFT", "RIGHT"]

    def check_action(self, action: str) -> bool:
        return action in self.ACTIONS

    def __setitem__(self, xy: tuple, value: int):
        self.environment[xy] = value

    def __getitem__(self, xy: tuple) -> int:
        return self.environment[xy[0]][xy[1]]

    def __str__(self) -> str:
        return str(self.environment)

    def __len__(self) -> int:
        return len(self.environment)

    def clear(self):
        self.environment = np.zeros((4,4))