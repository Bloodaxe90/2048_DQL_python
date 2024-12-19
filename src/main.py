import numpy as np
import torch.cuda
from PySide6.QtWidgets import QApplication
import matplotlib.pyplot as plt

from src.DQL.q_leanring import DQL
from src.UI.application import Application
from src.game.dynamics import check_terminal, game_step
from src.utils.utils import get_device


def main():
    print("Started")
    app = QApplication([])
    window = Application()
    window.show()
    window.resize(400,400)
    app.exec()

if __name__ == "__main__":
    #main()

    # Data

    print(get_device())
    dql = DQL()
    dql.train(
        200
    )
    wins = 0
    episodes = 1000
    dic = {}
    print("sum")
    for i in range(1, episodes):
        while (result := check_terminal(dql.environment, dql.ACTIONS, dql.win_val)) == "":
            action = dql.get_best_action()
            moved, merge = game_step(dql.environment, action)
            if not moved and not merge:
                action = dql.get_random_action()
                moved, merge = game_step(dql.environment, action)

        max_val = np.max(dql.environment)
        dql.reset()
        wins += 1 if result == "W" else 0
        if i % (episodes / 10) == 0:
            print(i)
        dic[max_val] = dic.get(max_val, 0) + 1
    print(wins / episodes)

    tile_values = [float(x) for x in dic.keys()]
    occurrences = list(dic.values())
    # Sort data for better visualization
    sorted_data = sorted(zip(tile_values, occurrences), key=lambda x: x[0])
    sorted_tiles, sorted_occurrences = zip(*sorted_data)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.bar(sorted_tiles, sorted_occurrences, color='skyblue', edgecolor='black')
    plt.xlabel("Tile Value", fontsize=12)
    plt.ylabel("Occurrences", fontsize=12)
    plt.title("Tile Occurrences in 2048", fontsize=14)
    plt.xticks(sorted_tiles)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()