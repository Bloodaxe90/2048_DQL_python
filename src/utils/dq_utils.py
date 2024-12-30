import math
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.game.dynamics import *
from datetime import datetime
import os

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_reward(current_state, next_state):
    current_max = np.max(current_state)
    next_max = np.max(next_state)
    reward = (math.log2(next_max) * 0.1)

    if next_max == current_max:
        reward = 0

    reward += len(empty(next_state)) - len(empty(current_state))

    return reward

def make_prediction(model: nn.Module, state: torch.Tensor, device: str) -> torch.tensor:
    model.eval()
    with torch.inference_mode():
        pred = model(state).to(device)
    return pred

def one_hot_states(states: list[np.ndarray], num_classes: int, device: str) -> torch.Tensor:
    states_logged = []

    for state in states:
        state_log = np.where(state == 0, 0, np.log2(state)).astype(np.int64)
        states_logged.append(state_log)

    states_tensor = torch.Tensor(np.array(states_logged)).long().to(device)

    return torch.nn.functional.one_hot(states_tensor, num_classes= num_classes).float().to(device).permute(0, 3, 1, 2)

def create_summary_writer(model: str, trail_name: str) -> torch.utils.tensorboard.SummaryWriter:
    timestamp = datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.join("resources/runs", timestamp, model, trail_name)

    return SummaryWriter(log_dir=log_dir)
