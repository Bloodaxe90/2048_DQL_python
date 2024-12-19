import random
import numpy as np


def game_step(state: np.ndarray, action: str) -> tuple:
    moved = move(state, action)
    merged_vals = merge(state, action)
    if moved or merged_vals:
        move(state, action)
        create_random(state)

    return moved, merged_vals

def check_terminal(state: np.ndarray, actions: list, win_val: int= 2048) -> str:
    if check_win(state, win_val):
        return "W"
    elif check_loss(state, actions):
        return "L"
    else:
        return ""

def check_win(state: np.ndarray, win_val) -> bool:
    return np.any(state.flatten() == win_val)

def check_loss(state: np.ndarray, actions: list) -> bool:
    if check_full(state):
        return  all(not merge(state.copy(), action) for action in actions)

def check_full(state: np.ndarray) -> bool:
    return not empty(state)

def move(state: np.ndarray, action: str) -> bool:
    old_state = state.copy()
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

    return not np.array_equal(old_state, state) #if arrays are equal then a move has not occured
                
                

def merge(state: np.ndarray, action: str) -> list:
    merge_values = []
    for i in range(len(state)):
        for j in get_moving_range(state, action):
            old_xy = (j, i) if action in ("UP", "DOWN") else (i, j) # "left" or "right"
            new_xy = (j+1, i) if action in ("UP", "DOWN") else (i, j+1) # "left" or "right"

            if j < len(state) -1 and state[old_xy] != 0 and state[old_xy] == state[new_xy]:
                new_value = state[new_xy] * 2
                state[new_xy] = new_value
                delete(state, old_xy)
                merge_values.append(new_value)

    return merge_values

def get_invalid_actions(state: np.ndarray, actions: list) -> list:
    invalid_actions = []
    for action in actions:
        current_state = state.copy()
        moved = move(current_state, action)
        merged_vals = merge(current_state, action)
        if not moved and not merged_vals:
            invalid_actions.append(action)
    return invalid_actions

def get_moving_range(state: np.ndarray, action: str):
    if action in ("DOWN", "RIGHT"):
        return range(len(state) -1, -1, -1)
    elif action in ("UP", "LEFT"):
        return range(len(state))
    else:
        raise ValueError(f"Invalid action {action}")


def create_random(
        state: np.ndarray,
        default_values: tuple= (2, 4),
        default_probs: tuple= (0.9, 0.1)
):
    if not check_full(state):
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