import numpy as np

from src.game.dynamics import *


class Twenty48:

    def __init__(self):
        self.environment: np.ndarray = np.zeros((4,4))
        self.ACTIONS: list= ["UP", "DOWN", "LEFT", "RIGHT"]
        create_random(self.environment)
        create_random(self.environment)

    def check_action(self, action: str) -> bool:
        return action in self.ACTIONS

    def play(self, action):
        if not self.check_action(action):
            raise ValueError("Invalid Action")

        move(self.environment, action)
        merge(self.environment, action)
        move(self.environment, action)
        create_random(self.environment)



    def __str__(self) -> str:
        return str(self.environment)

    def __len__(self) -> int:
        return len(self.environment)

    def clear(self):
        self.environment = np.zeros((4,4))