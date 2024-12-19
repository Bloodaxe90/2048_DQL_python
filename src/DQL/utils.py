import math
import random
from collections import deque

import numpy as np
import torch
from torch import nn

from src.game.dynamics import *
from src.utils.utils import get_device


def get_reward(current_state, next_state, win_val):
    current_max = np.max(current_state)
    next_max = np.max(next_state)
    reward = (math.log2(next_max) * 0.1)

    if next_max == current_max:
        reward = 0

    reward += len(empty(next_state)) - len(empty(current_state))

    return reward

def make_prediction(model: nn.Module ,state: np.ndarray) -> torch.tensor:
    model.eval()
    state = encode_state_batch([state])
    device = get_device()
    with torch.inference_mode():
        state = state.to(get_device())
        pred = model(state).to(device)

    return pred

class ReplayBuffer(deque):

    def __init__(self, capacity: int = 6000):
        super().__init__(maxlen=capacity)
        self.capacity: int = capacity

    def append(self, transition):
        super().append(transition)

    def full(self) -> bool:
        return len(self) >= self.maxlen

    def sample(self, batch_size: int):
        return random.sample(self, batch_size)

def encode_state_batch(states: list[np.ndarray]):

    states_flat = []
    for state in states:
        state_log = np.where(state == 0, 0, np.log2(state)).astype(np.int64)
        states_flat.append(state_log)

    states_tensor = torch.Tensor(np.array(states_flat)).long().to(get_device())

    states_one_hot = torch.nn.functional.one_hot(states_tensor, num_classes=12).float().to(get_device())

    states_one_hot = states_one_hot.permute(0, 3, 1, 2)

    return states_one_hot

