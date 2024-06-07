import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import numpy as np
from .kan import KANLayer

class KQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(KQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.alpha = lr
        self.fc1 = KANLayer([self.input_dims, self.fc1_dims, self.fc2_dims])
        self.fc2 = KANLayer([self.fc2_dims, self.fc1_dims, self.n_actions])

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)

        return x
    



class QNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(QNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.alpha = lr
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc1_dims)
        self.fc4 = nn.Linear(self.fc1_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)

        return actions
    


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
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
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
                                   fc1_dims=256, fc2_dims=512)
        
        self.alpha = self.Q_eval.alpha
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
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
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
                                   fc1_dims=16, fc2_dims=32)
        
        self.alpha = self.Q_eval.alpha
        
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


