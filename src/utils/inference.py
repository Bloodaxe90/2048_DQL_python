import math
import torch
from matplotlib import pyplot as plt
from torch import nn
from src.game.dynamics import *

def trail_ai(ai, episodes: int) -> dict:
    dic = {}
    for i in range(1, episodes):
        while ai.check_terminal() == "":
            action = ai.get_best_action()
            game_step(ai.environment, action)

        max_val = np.max(ai.environment)
        ai.reset()
        print(i)
        dic[max_val] = dic.get(max_val, 0) + 1
    return dic

def plot_results(results:dict):
    tile_values = [float(x) for x in results.keys()]
    occurrences = list(results.values())
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

