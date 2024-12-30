import random
import numpy as np


def game_step(state: np.ndarray, action: str):
    moved = move(state, action)
    merged = merge(state, action)
    if moved or merged:
        move(state, action)
        create_random(state)

def check_full(state: np.ndarray) -> bool:
    return not empty(state)

def move(state: np.ndarray, action: str) -> bool:
    moved = False
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
                    moved = True

                free_pos += 1 if action in ("UP", "LEFT") else -1

    return moved
                
                

def merge(state: np.ndarray, action: str) -> bool:
    merged = False
    for i in range(len(state)):
        for j in get_moving_range(state, action):
            if action == "UP":
                old_xy = (j + 1, i)
                new_xy = (j, i)
            elif action == "DOWN":
                old_xy = (j, i)
                new_xy = (j + 1, i)
            elif action == "LEFT":
                old_xy = (i, j+1)
                new_xy = (i, j)
            elif action == "RIGHT":
                old_xy = (i, j)
                new_xy = (i, j+1)
            else:
                raise ValueError("Invalid Action")

            if j < len(state) -1 and state[old_xy] != 0 and state[old_xy] == state[new_xy]:
                new_value = state[new_xy] * 2
                state[new_xy] = new_value
                delete(state, old_xy)
                merged = True

    return merged

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