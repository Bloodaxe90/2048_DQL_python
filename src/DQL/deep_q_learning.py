import os.path

import torch.optim
from src.DQL.deep_q_agent import DeepQAgent
from src.DQL.models import DQCNN, BasicDQCNN
from src.utils.dq_utils import *
from src.utils.save_load import save_model
from datetime import datetime
import torch.distributed as dist
import torch.multiprocessing as mp


class DeepQLearning:

    def __init__(self,
                 agent: DeepQAgent,
                 loss_fn: nn.Module,
                 batch_size: int = 32,
                 alpha: float = 0.00005,
                 gamma: float = 0.9,
                 ):
        self.device = get_device()
        self.agent = agent
        self.target_network = BasicDQCNN(input_neurons=self.agent.input_neurons,
                     hidden_neurons=self.agent.hidden_neurons,
                     output_neurons=len(self.agent.ACTIONS),
                     state_size=len(self.agent)).to(self.device)

        self.ALPHA = alpha
        self.optimizer = torch.optim.Adam(self.agent.main_network.parameters(), lr= self.ALPHA)
        self.loss_fn = loss_fn

        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma

    def train(self,
              episodes: int,
              trail_name: str,
              main_update_count: int = 100,
              main_update_freq: int = 1,
              target_update_freq: int = 20,
              model_save_name: str = '',
    ):
        writer = create_summary_writer(
            trail_name= trail_name,
            model = self.agent.main_network.__class__.__name__
        )
        start_time = datetime.now()

        total_scores = []
        for episode in range(1, episodes + 1):
            steps = 0
            total_score = 0
            total_loss = 0
            #Run through an episode
            while self.agent.check_terminal() == "":
                old_score = np.sum(self.agent.environment)
                self.agent.interact()
                steps += 1
                total_score += np.sum(self.agent.environment) - old_score

            #Decay Epsilon
            self.agent.decay_epsilon(episode, episodes)

            #Update main network
            if len(self.agent.replay_buffer) > self.BATCH_SIZE and episode % main_update_freq == 0:
                for _ in range(main_update_count):
                    total_loss += self.update_main_network()

            #Update target network
            if episode % target_update_freq == 0:
                self.update_target_network()

            highest_val = np.max(self.agent.environment)
            loss = total_loss / (main_update_count * main_update_freq)
            print(
                f"Episode: {episode}"
                f" | Highest_Val: {highest_val}"
                f" | Steps: {steps}"
                f" | Epsilon: {self.agent.epsilon}"
                f" | Loss: {loss}")


            total_scores.append(total_score)
            if episode > 50:
                average_score = sum(total_scores[-50:]) / 50
                print(f"Average Score from last 50 episodes: {average_score}")
                writer.add_scalar(tag="AVERAGE_SCORE",
                                  scalar_value=average_score,
                                  global_step=episode)

            if episode % 100 == 0 and model_save_name != '':
                # Save model
                save_model(model=self.agent.main_network,
                           target_dir=os.path.join("resources/saved_models/main_net"),
                           model_name=model_save_name)
                save_model(model=self.target_network,
                           target_dir=os.path.join("resources/saved_models/target_net"),
                           model_name=model_save_name)

            writer.add_scalar(tag= "LOSS",
                              scalar_value= loss,
                              global_step= episode)
            writer.add_scalar(tag="HIGHEST_VALUE",
                              scalar_value= highest_val,
                              global_step=episode)
            writer.add_scalar(tag="EPSILON",
                              scalar_value=self.agent.epsilon,
                              global_step=episode)
            writer.add_scalar(tag="STEPS",
                              scalar_value=steps,
                              global_step=episode)

            #Reset environment for next episode
            self.agent.reset()
            end_time = datetime.now()
            print(end_time-start_time)
        writer.close()


    def update_target_network(self):
        #Copy the main network parameters into the target networks
        self.target_network.load_state_dict(self.agent.main_network.module.state_dict())

    def update_main_network(self) -> float:
        #Fetch Transitions
        transitions = self.agent.replay_buffer.sample(self.BATCH_SIZE)

        #Format Transitions individual components for training
        one_hot_device = "cuda:0" if self.device == "cuda" else self.device
        states = one_hot_states([states[0] for states in transitions], self.agent.input_neurons, self.device).to(one_hot_device)
        action_indices = torch.tensor([self.agent.ACTIONS.index(actions[1]) for actions in transitions], device= self.device)
        rewards = torch.tensor([rewards[2] for rewards in transitions], dtype= torch.int64, device= self.device)
        next_states = one_hot_states([next_states[3] for next_states in transitions], self.agent.input_neurons, self.device).to(one_hot_device)
        dones = torch.tensor([dones[4] for dones in transitions], device= self.device)

        self.agent.main_network.train()
        self.target_network.eval()

        #prediction for staten s and action a
        current_q_values = self.agent.main_network(states)

        #prediction for state s' across actions a'
        with torch.inference_mode():
            next_q_values = self.target_network(next_states)

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




