import numpy as np

from src.game.dynamics import *


class Twenty48:

    def __init__(self, win_val: int = 2048):
        self.environment: np.ndarray = np.zeros((4,4))
        self.ACTIONS: list= ["UP", "DOWN", "LEFT", "RIGHT"]
        self.win_val = win_val
        self.setup()

    def check_action(self, action: str) -> bool:
        return action in self.ACTIONS

    def check_terminal(self) -> str:
        if self.check_win():
            return "W"
        elif self.check_loss():
            return "L"
        else:
            return ""

    def check_win(self) -> bool:
        return any(value == self.win_val for value in self.environment.flatten())

    def check_loss(self) -> bool:
        return (check_full(self.environment)
                and all(not merge(self.environment.copy(), action) for action in self.ACTIONS))

    def __str__(self) -> str:
        return str(self.environment)

    def __len__(self) -> int:
        return len(self.environment)

    def setup(self):
        create_random(self.environment)
        create_random(self.environment)

    def clear(self):
        self.environment = np.zeros((4,4))

    def reset(self):
        self.clear()
        self.setup()
