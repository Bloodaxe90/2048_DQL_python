import random
import numpy as np


def check_terminal(state: np.ndarray) -> str:
    if check_win(state):
        return "W"
    elif check_loss(state):
        return "L"
    else:
        return ""

def check_win(state: np.ndarray, win_val: int= 2048 ) -> bool:
    return any(val == win_val for val in state.flat)

def check_loss(state: np.ndarray) -> int:
    return empty(state) is None


def move(state: np.ndarray, action: str):
    for i in range(len(state)):

        free_pos = 0 if action in ("UP", "LEFT") else len(state) - 1 # "left" or "right"

        for j in get_moving_range(state, action):
            if action in ("UP", "DOWN"):
                old_xy = (j, i)
                new_xy = (free_pos, i)
            else:  # "left" or "right"
                old_xy = (i, j)
                new_xy = (i, free_pos)

            if state[old_xy] != 0:
                if old_xy != new_xy:
                    state[new_xy] = state[old_xy]
                    delete(state, old_xy)

                free_pos += 1 if action in ("UP", "LEFT") else -1

def merge(state: np.ndarray, action: str):
    for i in range(len(state)):
        for j in get_moving_range(state, action):
            old_xy = (j, i) if action in ("UP", "DOWN") else (i, j) # "left" or "right"
            new_xy = (j+1, i) if action in ("UP", "DOWN") else (i, j+1) # "left" or "right"

            if j < len(state) -1 and state[old_xy] != 0 and state[old_xy] == state[new_xy]:
                state[new_xy] = state[new_xy] * 2
                delete(state, old_xy)


def get_moving_range(state: np.ndarray, action: str):
    if action in ("DOWN", "RIGHT"):
        return range(len(state) -1, -1, -1)
    elif action in ("UP", "LEFT"):
        return range(len(state))
    else:
        raise ValueError("Invalid action")


def create_random(
        state: np.ndarray,
        default_values: tuple= (2, 4),
        default_probs: tuple= (0.9, 0.1)
):
    random_xy: tuple = random.choice(empty(state))
    random_value: int = random.choices(
        default_values, weights=default_probs
    )[0]
    state[random_xy] = random_value


def empty(state: np.ndarray) -> list[tuple]:
    return [
        (i, j) for i in range(len(state)) for j in range(len(state)) if state[(i, j)] == 0
        ]

def delete(state: np.ndarray, xy: tuple):
    state[xy] = 0