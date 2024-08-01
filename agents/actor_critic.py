import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .kan import KANLayer
import os


class KANActorCriticNetwork(nn.Module):
    def __init__(self, input_dims=4, fc1_dims=16, fc2_dims=32, n_actions=2, lr=0.0003):
        """
        This class for build actor acritic network based on KAN layer

        Args:
            HP: dict - (alpha, input_dim, fc1, fc2, n_action)

        Returns:
            pi (tensor) - policy
            v (tensor) - value

        """
        super(KANActorCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.alpha = lr
        self.fc1 = KANLayer([self.input_dims, 16, 32])
        self.pi = KANLayer([32, 16, self.n_actions])
        self.v = KANLayer([32, 16, 1])
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        state = torch.Tensor(observation).to(self.device)
        x = self.fc1(state)
        pi = self.pi(x)
        v = self.v(x)
        return (pi, v)
    


class ActorCriticNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi = nn.Linear(self.fc2_dims, n_actions)
        self.v = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        state = torch.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)
        return (pi, v)
    

    



class KANACAgent(object):
    """ Agent class for use with a single actor critic network that shares
        the lowest layers. For use with more complex environments such as
        the discrete lunar lander
    """
    def __init__(self, alpha, input_dims=4, gamma=0.99,
                 layer1_size=16, layer2_size=32, action_dim=2):
        self.name = "KAN based Actor-Critic Agent"
        self.gamma = gamma
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.alpha = alpha
        self.actor_critic = KANActorCriticNetwork()

        self.log_probs = None

    def choose_action(self, observation):
        probabilities, _ = self.actor_critic.forward(observation)
        probabilities = F.softmax(probabilities, dim=-1)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.log_probs = log_probs

        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor_critic.optimizer.zero_grad()

        _, critic_value_ = self.actor_critic.forward(new_state)
        _, critic_value = self.actor_critic.forward(state)
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor_critic.device)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor_critic.optimizer.step()

    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor_critic.state_dict(), os.path.join(path, "actor_critic.pth"))

    def load_model(self, path):
        self.actor_critic.load_state_dict(torch.load(path))



class ACAgent(object):
    """ Agent class for use with a single actor critic network that shares
        the lowest layers. For use with more complex environments such as
        the discrete lunar lander
    """
    def __init__(self, alpha=0.0003, input_dims=4, gamma=0.99,
                 layer1_size=16, layer2_size=32, action_dim=2):
        self.name = "Vanilla Actor-Critic Agent"
        self.gamma = gamma
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.alpha = alpha
        
        self.actor_critic = ActorCriticNetwork(alpha=self.alpha, input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=action_dim)

        self.log_probs = None

    def choose_action(self, observation):
        probabilities, _ = self.actor_critic.forward(observation)
        probabilities = F.softmax(probabilities, dim=-1)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.log_probs = log_probs

        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor_critic.optimizer.zero_grad()

        _, critic_value_ = self.actor_critic.forward(new_state)
        _, critic_value = self.actor_critic.forward(state)
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor_critic.device)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor_critic.optimizer.step()

    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor_critic.state_dict(), os.path.join(path, "actor_critic.pth"))

    def load_model(self, path):
        self.actor_critic.load_state_dict(torch.load(path))




class GenericNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(GenericNetwork, self).__init__()
        self.lr = lr
        self.input_dims = fc1_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.to(device=self.device)


    def forward(self, observation):
        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    


class ContinousActorCritic(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions=2, layer1_size=64, n_outputs=1):
        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_outputs
        self.actor = GenericNetwork(alpha, input_dims, layer1_size, layer1_size, n_actions=n_actions)
        self.critic = GenericNetwork(beta, input_dims, layer1_size, layer1_size, n_actions=1)
        