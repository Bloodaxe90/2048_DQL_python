import time

from PySide6.QtWidgets import QApplication

from src.DQL.deep_q_agent import DeepQAgent
from src.game.dynamics import game_step
from src.utils.save_load import load_model


class QAI(DeepQAgent):

    def __init__(self,
                 controller,
                 model_dir: str,
                 hidden_neurons= (128, 128, 128, 128)
                 ):
        super().__init__(hidden_neurons= hidden_neurons)
        self.controller = controller
        self.controller.set_board(self.environment)
        load_model(self.main_network, model_dir, self.device)

    def play(self):
        print("Ai Playing")
        while (result := self.check_terminal()) == "":
            action = self.get_best_action()
            game_step(self.environment, action)

            self.controller.set_board(self.environment)
            QApplication.processEvents()
            time.sleep(0.5)

        self.controller.game_over(result)



    def reset(self):
        super().reset()
        self.controller.set_board(self.environment)


