import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import numpy as np
from .kan import KANLayer
import matplotlib.pyplot as plt
from torchsummary import summary
from utils.logger import CustomLogger


logger = CustomLogger(log_dir="logs", log_file_prefix="experiment")


class KQNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, hidden_layers=None):
        """
        Dynamic KAN-Based Q-Network
        :param lr: Learning rate
        :param input_dims: Observation space dimensions (int)
        :param n_actions: Number of actions (output dimensions)
        :param hidden_layers: List specifying the number of neurons in hidden layers
                              If None, default [256, 256] is used.
        """
        super(KQNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.hidden_layers = hidden_layers or [256, 256]  # Default hidden layer sizes

        # Build dynamic model using KANLayers
        layers = []
        layer_dims = [self.input_dims] + self.hidden_layers + [self.n_actions]

        for i in range(0, len(layer_dims) - 2, 2):
            # Pass three arguments to KANLayer: input, intermediate, output dims
            layers.append(KANLayer([layer_dims[i], layer_dims[i + 1], layer_dims[i + 2]]))

        # Combine layers into a sequential model
        self.model = nn.Sequential(*layers)

        # Optimizer and loss
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.model(state)

    



class QNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, hidden_layers=None):
        """
        Dynamic Q-Network
        :param lr: Learning rate
        :param input_dims: Observation space dimensions (int)
        :param n_actions: Number of actions (output dimensions)
        :param hidden_layers: List specifying the number of neurons in hidden layers
                              If None, default [256, 256] is used.
        """
        super(QNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.hidden_layers = hidden_layers or [256, 256]  # Default hidden layer sizes

        # Build dynamic sequential model
        layers = []
        input_size = self.input_dims

        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, self.n_actions))

        self.model = nn.Sequential(*layers)

        # Optimizer and loss
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        return self.model(state)
    




class ReplayBuffer:

    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.stack(states), actions, rewards, np.stack(next_states), dones

    def __len__(self):
        return len(self.buffer)
    




class DQNAgent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, n_layer,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100
        self.name = "DQN"

        self.Q_eval = QNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims,
                                   hidden_layers=n_layer)
        
        logger.log("info", f"QNetwork Model Summary:\n {summary(self.Q_eval, input_size=(1, input_dims))}")

        
        self.alpha = self.gamma
        self.state_memory = np.zeros((self.mem_size, input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(
                self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = torch.tensor(
                self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(
                self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_model(self, filename):
        """
        Save the model's state to a file.
        
        Args:
            filename (str): The name of the file to save the model's state.
        """
        checkpoint = {
            'model_state_dict': self.Q_eval.state_dict(),
            'optimizer_state_dict': self.Q_eval.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'mem_cntr': self.mem_cntr,
            'iter_cntr': self.iter_cntr,
            'state_memory': self.state_memory,
            'new_state_memory': self.new_state_memory,
            'action_memory': self.action_memory,
            'reward_memory': self.reward_memory,
            'terminal_memory': self.terminal_memory
        }
        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """
        Load the model's state from a file.
        
        Args:
            filename (str): The name of the file to load the model's state from.
        """
        checkpoint = torch.load(filename)
        self.Q_eval.load_state_dict(checkpoint['model_state_dict'])
        self.Q_eval.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.mem_cntr = checkpoint['mem_cntr']
        self.iter_cntr = checkpoint['iter_cntr']
        self.state_memory = checkpoint['state_memory']
        self.new_state_memory = checkpoint['new_state_memory']
        self.action_memory = checkpoint['action_memory']
        self.reward_memory = checkpoint['reward_memory']
        self.terminal_memory = checkpoint['terminal_memory']
        print(f"Model loaded from {filename}")




class KDQNAgent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, n_layer,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100
        self.name = "Kan-DQN"

        self.Q_eval = KQNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims,
                                   hidden_layers=n_layer)
        
        #logger.log("info", f"QNetwork Model Summary:\n {summary(self.Q_eval, input_size=(1, input_dims))}")
        
        self.alpha = self.gamma
        
        self.state_memory = np.zeros((self.mem_size, input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(
                self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = torch.tensor(
                self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(
                self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


    def save_model(self, filename):
        """
        Save the model's state to a file.
        
        Args:
            filename (str): The name of the file to save the model's state.
        """
        checkpoint = {
            'model_state_dict': self.Q_eval.state_dict(),
            'optimizer_state_dict': self.Q_eval.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'mem_cntr': self.mem_cntr,
            'iter_cntr': self.iter_cntr,
            'state_memory': self.state_memory,
            'new_state_memory': self.new_state_memory,
            'action_memory': self.action_memory,
            'reward_memory': self.reward_memory,
            'terminal_memory': self.terminal_memory
        }
        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """
        Load the model's state from a file.
        
        Args:
            filename (str): The name of the file to load the model's state from.
        """
        checkpoint = torch.load(filename)
        self.Q_eval.load_state_dict(checkpoint['model_state_dict'])
        self.Q_eval.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.mem_cntr = checkpoint['mem_cntr']
        self.iter_cntr = checkpoint['iter_cntr']
        self.state_memory = checkpoint['state_memory']
        self.new_state_memory = checkpoint['new_state_memory']
        self.action_memory = checkpoint['action_memory']
        self.reward_memory = checkpoint['reward_memory']
        self.terminal_memory = checkpoint['terminal_memory']
        print(f"Model loaded from {filename}")





def generate_comparison_charts(dqn_results, kqn_results, filename_prefix):
    """
    Generate comparison charts for DQN and KQN.
    
    :param dqn_results: Tuple of (scores, epsilons) for DQN
    :param kqn_results: Tuple of (scores, epsilons) for KQN
    :param filename_prefix: Prefix for the chart filenames
    """
    # Unpack results
    dqn_scores, dqn_epsilons = dqn_results
    kqn_scores, kqn_epsilons = kqn_results

    # Generate x-axis values
    episodes = [i + 1 for i in range(len(dqn_scores))]

    # 1. Total Reward over Episodes
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, dqn_scores, label="DQN", alpha=0.7)
    plt.plot(episodes, kqn_scores, label="KQN", alpha=0.7)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Total Reward over Episodes")
    plt.legend()
    plt.savefig(f"{filename_prefix}_total_reward.png")
    plt.show()

    # 2. Sample Efficiency (Cumulative Reward per Episode)
    dqn_cumulative_reward = np.cumsum(dqn_scores) / episodes
    kqn_cumulative_reward = np.cumsum(kqn_scores) / episodes

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, dqn_cumulative_reward, label="DQN", alpha=0.7)
    plt.plot(episodes, kqn_cumulative_reward, label="KQN", alpha=0.7)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward per Episode")
    plt.title("Sample Efficiency")
    plt.legend()
    plt.savefig(f"{filename_prefix}_sample_efficiency.png")
    plt.show()

    # 3. Final Performance (Average Reward over Last N Episodes)
    N = 100  # Define the number of episodes to calculate final performance
    dqn_final_performance = np.mean(dqn_scores[-N:])
    kqn_final_performance = np.mean(kqn_scores[-N:])

    plt.figure(figsize=(10, 6))
    plt.bar(["DQN", "KQN"], [dqn_final_performance, kqn_final_performance], alpha=0.7)
    plt.ylabel("Final Performance (Average Reward)")
    plt.title(f"Final Performance (Last {N} Episodes)")
    plt.savefig(f"{filename_prefix}_final_performance.png")
    plt.show()

    # 4. Training Stability (Variance in Scores)
    dqn_variance = np.var(dqn_scores[-N:])
    kqn_variance = np.var(kqn_scores[-N:])

    plt.figure(figsize=(10, 6))
    plt.bar(["DQN", "KQN"], [dqn_variance, kqn_variance], alpha=0.7)
    plt.ylabel("Training Stability (Variance in Reward)")
    plt.title(f"Training Stability (Last {N} Episodes)")
    plt.savefig(f"{filename_prefix}_training_stability.png")
    plt.show()
