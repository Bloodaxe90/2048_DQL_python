import os.path

from PySide6.QtWidgets import QApplication

from src.DQL.deep_q_agent import DeepQAgent
from src.DQL.deep_q_learning import DeepQLearning
from src.UI.application import Application
from src.utils.dq_utils import *
from src.utils.inference import *
from src.utils.save_load import load_model


def main():
    print("Started")
    print(get_device())
    app = QApplication([])
    window = Application()
    window.show()
    window.setFixedSize(400,450)
    app.exec()

def training():
    SEED: int = 20
    EPISODES: int = 1
    HIDDEN_NEURONS: tuple =(128, 128, 128, 128)
    REPLAY_CAPACITY: int = 12800
    BATCH_SIZE: int = 32
    ALPHA: float = 0.00005
    GAMMA: float = 0.9
    TRIAL_NAME: str = "test"

    MAIN_UPDATE_COUNT: int = 100
    MAIN_UPDATE_FREQ: int = 1
    TARGET_UPDATE_FREQ: int = 20
    MODEL_SAVE_NAME: str = ''

    set_seed(SEED)

    print(get_device())
    dql = DeepQLearning(
        agent= DeepQAgent(hidden_neurons= HIDDEN_NEURONS,
                          replay_capacity= REPLAY_CAPACITY),
        loss_fn= torch.nn.MSELoss(),
        batch_size= BATCH_SIZE,
        alpha= ALPHA,
        gamma= GAMMA
    )

    dql.train(
        episodes=EPISODES,
        trail_name=TRIAL_NAME,
        main_update_count= MAIN_UPDATE_COUNT,
        main_update_freq= MAIN_UPDATE_FREQ,
        target_update_freq= TARGET_UPDATE_FREQ,
        model_save_name= MODEL_SAVE_NAME,
    )

    results = trail_ai(dql.agent, 400)
    #0.6 {128.0: 211, 64.0: 111, 256.0: 63, 32.0: 14}
    #0.4 {256.0: 78, 128.0: 221, 64.0: 91, 32.0: 7, 512.0: 2}
    #1.0 {64.0: 83, 128.0: 215, 256.0: 90, 32.0: 10, 512.0: 1}
    #1.2 {128.0: 208, 256.0: 99, 64.0: 88, 32.0: 3, 512.0: 1}
    #1.4 {256.0: 86, 128.0: 211, 64.0: 89, 32.0: 13}
    #1.8 {256.0: 84, 128.0: 215, 64.0: 94, 512.0: 1, 16.0: 1, 32.0: 4}

    #5gpu 0:01:22.371397
    #4gpu 0:01:12.566253
    #3gpu 0:01:10.321848
    #2gpu 0:01:21.551833
    #1gpu 0:01:22.157120

    #fail at 0.09891759605759104, ep 6400
    #fail at 0.5967424826153863, ep 8800
    #fail at 0.45066064556861013, ep 13300
    #fail at 0.3574634932646058, ep 16300
    #fail at 0.1732420098198584, ep 22700
    #fail at 0.02870087796735485, ep 28800

    #results 1000ep the_big_one:
    #{np.float64(512.0): 394,
    # np.float64(1024.0): 527,
    # np.float64(256.0): 53,
    # np.float64(2048.0): 22,
    # np.float64(128.0): 3}

    print(results)

if __name__ == "__main__":
    main()
    #training()



