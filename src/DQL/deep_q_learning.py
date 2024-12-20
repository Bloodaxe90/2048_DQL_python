import torch.optim

from src.DQL.model import DQCNN
from src.DQL.replay_buffer import ReplayBuffer
from src.utils.dq_utils import *
from src.game.dynamics import game_step
from src.game.tweny48 import Twenty48


class DeepQLearning(Twenty48):

    def __init__(self,
                 replay_buffer: ReplayBuffer,
                 loss_fn: nn.Module,
                 hidden_neurons: tuple,
                 batch_size: int = 32,
                 alpha: float = 0.00005,
                 gamma: float = 0.9,
                 max_epsilon: float = 0.9,
                 min_epsilon: float = 0.01,
                 win_val: int = 2048,
                 device: str = "cpu",
                 ):
        super().__init__(win_val= win_val)
        self.input_neurons: int = int(math.log2(self.win_val) +1)
        self.hidden_neurons: tuple = hidden_neurons
        self.main_network: DQCNN = self.create_network().to(device)
        self.target_network: DQCNN = self.create_network().to(device)

        self.replay_buffer: ReplayBuffer = replay_buffer

        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr= alpha)
        self.loss_fn = loss_fn

        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.MAX_EPSILON = max_epsilon
        self.MIN_EPSILON = min_epsilon
        self.epsilon = self.MAX_EPSILON
        self.device = device

    def train(self,
              episodes: int,
              main_update_count: int = 100,
              main_update_freq: int = 1,
              target_update_freq: int = 20
              ):
        total_scores = []
        for episode in range(1, episodes + 1):
            steps = 0
            total_score = 0
            total_loss = 0
            #Run through an episode
            while self.check_terminal() == "":
                old_score = np.sum(self.environment)
                self.interact()
                steps += 1
                total_score += np.sum(self.environment) - old_score

            #Decay Epsilon
            self.decay_epsilon(episode, episodes)

            #Update main network
            if len(self.replay_buffer) > self.BATCH_SIZE and episode % main_update_freq == 0:
                for _ in range(main_update_count):
                    total_loss += self.update_main_network()

            #Update target network
            if episode % target_update_freq == 0:
                self.update_target_network()

            print(
                f"Episode: {episode}"
                f" | Highest_Val: {np.max(self.environment)}"
                f" | Steps: {steps}"
                f" | Epsilon: {self.epsilon}"
                f" | Loss: {total_loss / (main_update_count * main_update_freq)}")

            total_scores.append(total_score)
            if episode > 50:
                average = sum(total_scores[-50:]) / 50
                print(f"Average Score from last 50 episodes: {average}")

            #Reset environment for next episode
            self.reset()


    def interact(self):
        current_state = self.environment.copy()
        action = self.get_action()
        game_step(self.environment, action)
        reward = get_reward(current_state, self.environment)
        next_state = self.environment.copy()
        done = 1 if self.check_terminal() != "" else 0

        if self.check_terminal() != "" or len(self.replay_buffer) == 0 or not np.array_equal(current_state, next_state):
            self.replay_buffer.push((current_state, action, reward, next_state, done))

    def update_target_network(self):
        #Copy the main network parameters into the target networks
        self.target_network.load_state_dict(self.main_network.state_dict())

    def update_main_network(self) -> float:
        #Fetch Transitions
        transitions = self.replay_buffer.sample(self.BATCH_SIZE)

        #Format Transitions individual components for training
        states = one_hot_states([states[0] for states in transitions], self.input_neurons, self.device)
        action_indices = torch.tensor([self.ACTIONS.index(actions[1]) for actions in transitions], device= self.device)
        rewards = torch.tensor([rewards[2] for rewards in transitions], dtype= torch.int64, device= self.device)
        next_states = one_hot_states([next_states[3] for next_states in transitions], self.input_neurons, self.device)
        dones = torch.tensor([dones[4] for dones in transitions], device= self.device)

        self.main_network.train()
        self.target_network.eval()

        #prediction for staten s and action a
        current_q_values = self.main_network(states).to(self.device)

        #prediction for state s' across actions a'
        with torch.inference_mode():
            next_q_values = self.target_network(next_states).to(self.device)

        #Get the q_value of the action a taken in s
        current_q_values = current_q_values.gather(1, action_indices.unsqueeze(-1)).squeeze(-1)

        #Get the max next action from a'
        max_next_q_values = torch.amax(next_q_values, dim=1)

        #Calculate the TD target value
        td_target_values = rewards + ((1 - dones) * (self.GAMMA * max_next_q_values))


        #Calculate loss with the square of TD error
        loss = self.loss_fn(current_q_values, td_target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_action(self) -> str:
        if random.random() >= self.epsilon:
            return self.get_best_action()
        else:
            return self.get_random_action()

    def get_random_action(self) -> str:
        #Get a random action after filtering out invalid actions
        valid_actions = [action for action in self.ACTIONS if action not in self.get_invalid_actions()]
        return random.choice(valid_actions)

    def get_best_action(self) -> str:
        #Make predication
        prediction = make_prediction(self.main_network, self.environment, self.input_neurons, self.device).squeeze()

        #Mask invalid actions
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

    def create_network(self):
        return DQCNN(input_neurons=self.input_neurons,
                     hidden_neurons=self.hidden_neurons,
                     output_neurons=len(self.ACTIONS),
                     state_size=len(self))

