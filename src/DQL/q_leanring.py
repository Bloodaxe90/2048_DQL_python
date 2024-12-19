import random

import numpy as np
import torch

from src.DQL.model import *
from src.DQL.utils import *
from src.game.tweny48 import Twenty48


class DQL(Twenty48):

    def __init__(self, win_val = 2048, max_epsilon: float = 0.9, min_epsilon:float = 0.01, alpha: float = 0.00005):
        super().__init__()
        self.main_network: DQCNN = self.create_network().to(get_device())
        self.target_network: DQCNN = self.create_network().to(get_device())
        self.replay_buffer: ReplayBuffer = ReplayBuffer(6000)

        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.win_val = win_val
        self.epsilon = max_epsilon

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=alpha)

    def train(self, episodes, batch_size: int= 32):
        results = []
        total_scores = []
        steps = 0
        total_loss = 0

        for episode in range(1, episodes):
            while (result := check_terminal(self.environment, self.ACTIONS, self.win_val)) == "":
                self.interact()
                steps += 1
            self.decay_epsilon(episode, episodes)

            results.append(result)
            if len(self.replay_buffer) > batch_size:
                for _ in range(100):
                    total_loss += self.update_main_network(self.loss_fn, self.optimizer, batch_size)

            if episode % 20 == 0:
                self.update_target_network()

            if episode % 10 == 0:
                win_rate = sum([1 for r in results if r == "W"]) / 10
                print(f"Episode: {episode} | Result: {win_rate} | Highest_Val: {np.max(self.environment)}| avg_steps: {steps/episode} | epsilon: {self.epsilon} | loss: {total_loss/(100*10)}")
                results = []
                total_loss = 0

            total_scores.append(self.total)
            if episode > 50:
                average = sum(total_scores[-50:]) / 50
                print(f"50 episode running average: {average}")
            self.reset()
            self.total = 0


    def interact(self):
        current_state = self.copy()
        action = self.get_action()
        moved, merge_values = game_step(self.environment, action)
        reward = get_reward(current_state, self.environment, self.win_val)
        next_state = self.copy()
        done = 1 if check_terminal(self.environment, self.ACTIONS) != "" else 0

        self.total += sum(merge_values)

        if check_terminal(self.environment, self.ACTIONS) != "" or len(self.replay_buffer) == 0 or (moved or merge_values):
            self.replay_buffer.append((current_state, action, reward, next_state, done))

    def decay_epsilon(self, episode, max_episodes, power: float= 1.2):
        fraction = episode/max_episodes
        self.epsilon = (self.max_epsilon - self.min_epsilon) * ((1 - fraction) ** power) + self.min_epsilon


    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def update_main_network(self, loss_fn, optimizer, batch_size: int, gamma: float= 0.9):
        device = get_device()

        batch = list(self.replay_buffer.sample(batch_size))
        states = encode_state_batch([transition[0] for transition in batch]).to(get_device())
        actions = [transition[1] for transition in batch]
        rewards = torch.Tensor([transition[2] for transition in batch]).to(get_device())
        next_states = encode_state_batch([transition[3] for transition in batch]).to(get_device())
        dones = torch.Tensor([transition[4] for transition in batch]).to(get_device())

        self.main_network.train()
        self.target_network.eval()

        main_y = self.main_network(states).to(device)

        with torch.inference_mode():
            target_y = self.target_network(next_states).to(device)

        action_indices = [self.ACTIONS.index(action) for action in actions]
        current_q_values = main_y[range(len(actions)), action_indices]

        max_next_q_values = torch.amax(target_y, dim=1)

        target_q_values = rewards + ((1 - dones) * (gamma * max_next_q_values))


        loss = loss_fn(current_q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def get_action(self) -> str:
        if random.random() >= self.epsilon:
            return self.get_best_action()
        else:
            return self.get_random_action()

    def get_random_action(self) -> str:
        invalid_actions = get_invalid_actions(self.environment, self.ACTIONS)
        valid_actions = [action for action in self.ACTIONS if action not in invalid_actions]
        return random.choice(valid_actions)

    def get_best_action(self) -> str:
        invalid_actions  = get_invalid_actions(self.environment, self.ACTIONS)
        invalid_indices = [i for i, action in enumerate(self.ACTIONS) if action in invalid_actions]

        prediction = make_prediction(self.main_network, self.environment).squeeze()
        masked_q_values = prediction.clone()
        masked_q_values[invalid_indices] = -float('inf')
        return self.ACTIONS[torch.argmax(masked_q_values).item()]

    def create_network(self, hidden_neurons: tuple = (128, 128, 128, 128), ):
        return DQCNN(input_neurons=12,
                     hidden_neurons=hidden_neurons,
                     output_neurons=len(self.ACTIONS),
                     state_size=len(self))

    def update_target_network_advanced(self, tau: float = 0.001):
        for target_param, main_param in zip(self.target_network.parameters(), self.main_network.parameters()):
            target_param.data.copy_(
                tau * main_param.data + (1.0 - tau) * target_param.data
            )







