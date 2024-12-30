import math
import random

import numpy as np
import torch

from src.DQL.models import *
from src.DQL.replay_buffer import ReplayBuffer
from src.game.dynamics import game_step
from src.game.tweny48 import Twenty48
from src.utils.dq_utils import get_reward, make_prediction, one_hot_states, get_device


class DeepQAgent(Twenty48):

    def __init__(self,
                 hidden_neurons: tuple,
                 replay_capacity: int = 6400,
                 max_epsilon: float = 0.9,
                 min_epsilon: float = 0.01,
                 win_val: int = 2048,
                 ):
        super().__init__(win_val= win_val)
        self.device = get_device()
        self.input_neurons: int = int(math.log2(self.win_val) +1)
        self.hidden_neurons: tuple = hidden_neurons

        self.main_network = BasicDQCNN(input_neurons=self.input_neurons,
                     hidden_neurons=self.hidden_neurons,
                     output_neurons=len(self.ACTIONS),
                     state_size=len(self)).to(self.device)

        #Ensuring parallelism between all available GPUs
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            print("Parallel")
            self.main_network = nn.DataParallel(self.main_network, device_ids= [0, 1, 2])


        self.replay_buffer: ReplayBuffer = ReplayBuffer(capacity= replay_capacity)

        self.MAX_EPSILON = max_epsilon
        self.MIN_EPSILON = min_epsilon
        self.epsilon = self.MAX_EPSILON

    def interact(self):
        current_state = self.environment.copy()
        action = self.get_action()
        game_step(self.environment, action)
        reward = get_reward(current_state, self.environment)
        next_state = self.environment.copy()
        done = 1 if self.check_terminal() != "" else 0

        if self.check_terminal() != "" or len(self.replay_buffer) == 0 or not np.array_equal(current_state, next_state):
            self.replay_buffer.push((current_state, action, reward, next_state, done))

    def get_action(self) -> str:
        if random.random() >= self.epsilon:
            return self.get_best_action()
        else:
            return self.get_random_action()

    def get_random_action(self) -> str:
        # Get a random action after filtering out invalid actions
        valid_actions = [action for action in self.ACTIONS if action not in self.get_invalid_actions()]
        return random.choice(valid_actions)

    def get_best_action(self) -> str:
        # Make predication
        state = one_hot_states([self.environment], self.input_neurons, self.device)
        prediction = make_prediction(self.main_network, state, self.device).squeeze()

        # Mask invalid actions
        invalid_actions_indices = [self.ACTIONS.index(action) for action in self.ACTIONS
                                   if action in self.get_invalid_actions()]
        masked_prediction = prediction.clone()
        masked_prediction[invalid_actions_indices] = -float('inf')
        return self.ACTIONS[torch.argmax(masked_prediction).item()]

    def get_invalid_actions(self) -> list:
        invalid_actions = []
        for action in self.ACTIONS:
            current_state = self.environment.copy()
            game_step(current_state, action)
            if np.array_equal(current_state, self.environment):
                invalid_actions.append(action)
        return invalid_actions

    def decay_epsilon(self, episode, max_episodes, power: float= 1.2):
        fraction = episode/max_episodes
        self.epsilon = (self.MAX_EPSILON - self.MIN_EPSILON) * ((1 - fraction) ** power) + self.MIN_EPSILON