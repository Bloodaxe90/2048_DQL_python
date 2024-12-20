import torch.optim
from PySide6.QtWidgets import QApplication

from src.DQL.deep_q_learning import DeepQLearning
from src.DQL.replay_buffer import ReplayBuffer
from src.UI.application import Application
from src.utils.dq_utils import *
from src.utils.inference import *



def main():
    print("Started")
    app = QApplication([])
    window = Application()
    window.show()
    window.resize(400,400)
    app.exec()

if __name__ == "__main__":
    #main()


    device = get_device()
    set_seed(20)
    print(device)
    dql = DeepQLearning(
        replay_buffer= ReplayBuffer(capacity=12800),
        loss_fn= torch.nn.MSELoss(),
        hidden_neurons= (512, 512, 512, 512),
        batch_size= 64,
        win_val=2048,
        device= device

    )

    dql.train(
        episodes=500
    )

    results = trail_ai(dql, 1000)
    plot_results(results)



