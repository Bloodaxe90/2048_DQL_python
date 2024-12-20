from src.game.dynamics import *
from src.game.tweny48 import Twenty48


class Default(Twenty48):

    def __init__(self, controller):
        super().__init__()
        self.controller= controller
        self.controller.set_board(self.environment)


    def play(self, action):
        if not self.check_action(action):
            raise ValueError("Invalid Action")

        game_step(self.environment, action)

        result: str= self.check_terminal()
        if result in ("W", "L"):
            self.controller.game_over(result)
        self.controller.set_board(self.environment)

    def reset(self):
        super().reset()
        self.controller.set_board(self.environment)