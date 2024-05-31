import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from .kan import KANLayer

class GenericNetwork(nn.Module):
    def __init__(self, HP:dict):
        """
        Args:
            HP: dict - (alpha, input_dim, fc1, fc2, n_action)
        """
        super(GenericNetwork, self).__init__()
        self.input_dims = HP['input_dim']
        self.fc1_dim = HP['fc1']
        self.fc2_dim = HP['fc2']
        self.n_actions = HP['n_action']
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=HP['lr'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        state = torch.tensor(state).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class KANActorCriticNetwork(nn.Module):
    def __init__(self, HP:dict):
        """
        Args:
            HP: dict - (alpha, input_dim, fc1, fc2, n_action)
        """
        super(KANActorCriticNetwork, self).__init__()
        self.input_dims = HP['input_dim']
        self.fc1_dim = HP['fc1']
        self.fc2_dim = HP['fc2']
        self.n_actions = HP['n_action']
        self.fc1 = KANLayer([self.input_dims, 256, 512])
        self.pi = KANLayer([512, 256, self.n_actions])
        self.v = KANLayer([512, 128, 1])
        self.optimizer = optim.Adam(self.parameters(), lr=HP['lr'], weight_decay=1e-4)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, observation):
        state = torch.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)
        return (pi, v)



class Trainer:
    def __init__(self, agent, env, epochs):
        """
        This is class for training agent.

        Args:
            agent 
            env : currently supports gym env
            epochs (int)
        """
        self.agent = agent
        self.env = env
        self.epochs = epochs

    