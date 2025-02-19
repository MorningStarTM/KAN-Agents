import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.distributions.normal import Normal


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cent = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)


    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cent % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        
        self.mem_cent += 1

    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cent, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
    



class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dim, n_actions, name='Critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dim
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        

        self.fc1 = nn.Linear(self.input_dims+n_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)


    def forward(self, state, action):
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q
    

    def save_checkpoint(self, checkpoint_dir):
        self.checkpoint_file = os.path.join(checkpoint_dir, self.name+'_sac')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, checkpoint_dir):
        self.checkpoint_file = os.path.join(checkpoint_dir, self.name+'_sac')
        self.load_state_dict(torch.load(self.checkpoint_file))



class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, name='Value'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.name = name
        

        self.fc1 = nn.Linear(self.input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.v = nn.Linear(256, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)


    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)
        return v
    
    def save_checkpoint(self, checkpoint_dir):
        self.checkpoint_file = os.path.join(checkpoint_dir, self.name+'_sac')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, checkpoint_dir):
        self.checkpoint_file = os.path.join(checkpoint_dir, self.name+'_sac')
        self.load_state_dict(torch.load(self.checkpoint_file))



class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, n_layers, n_actions, name='actor'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name
        
        self.max_action = max_action
        self.reparam_noise = 1e-6
        self.hidden_layers = n_layers if n_layers else [256, 256]

        layers = []
        input_size = self.input_dims

        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        self.model = nn.Sequential(*layers)

        self.mu = nn.Linear(self.hidden_layers[-1], self.n_actions)
        self.sigma = nn.Linear(self.hidden_layers[-1], self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)


    def forward(self, state):
        state = state.to(torch.float32)  # ✅ Ensure input is float32
        x = self.model(state)

        mu = self.mu(x)
        mu = torch.tanh(mu) * self.max_action  # Scale actions within [-max_action, max_action]

        sigma = self.sigma(x)
        sigma = torch.clamp(sigma, min=-20, max=2)  # Prevent extremely large values
        sigma = torch.exp(sigma)  # Convert to positive values (log standard deviation)

        return mu, sigma

    


    def save_checkpoint(self, checkpoint_dir):
        self.checkpoint_file = os.path.join(checkpoint_dir, self.name+'_sac')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, checkpoint_dir):
        self.checkpoint_file = os.path.join(checkpoint_dir, self.name+'_sac')
        self.load_state_dict(torch.load(self.checkpoint_file))


    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = torch.tanh(actions)* torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2) + self.reparam_noise) #log_probs = log_probs.sum(1, keepdim=True) 
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs
    

#alpha=0.0003, beta=0.0003, input_dims=[8], env=None, max_action=0.4, gamma=0.99, n_actions=2, max_size=100000, tau=0.005, batch_size=256, reward_sclae=2
class SACAgent:
    def __init__(self, config):
        self.config = config
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']
        self.memory = ReplayBuffer(self.config['max_size'], self.config['input_dims'], self.config['n_actions'])
        self.batch_size = self.config['batch_size']
        self.n_actions = self.config['n_actions']
        self.name = "SAC"

        self.actor = ActorNetwork(alpha=self.config['alpha'], input_dims=self.config['input_dims'], n_actions=self.n_actions, n_layers=self.config['n_layers'], name='actor', max_action=self.config['max_action'])
        self.critic_1 = CriticNetwork(self.config['beta'], self.config['input_dims'], n_actions=self.n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(self.config['beta'], self.config['input_dims'], n_actions=self.n_actions, name='critic_2')
        self.value = ValueNetwork(self.config['beta'], self.config['input_dims'], name='value')
        self.target_value = ValueNetwork(self.config['beta'], self.config['input_dims'], name='target_value')

        self.scale = self.config['reward_sclae']
        self.updated_network_parameters(tau=1)


    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float32).to(self.actor.device)  # ✅ Fix: Convert to float32
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    
    def updated_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + (1-tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)


    def save_models(self, checkpoint):
        self.actor.save_checkpoint(checkpoint)
        self.value.save_checkpoint(checkpoint)
        self.target_value.save_checkpoint(checkpoint)
        self.critic_1.save_checkpoint(checkpoint)
        self.critic_2.save_checkpoint(checkpoint)
        print("__________models saved__________")

    def load_models(self, checkpoint):
        self.actor.load_checkpoint(checkpoint)
        self.value.load_checkpoint(checkpoint)
        self.target_value.load_checkpoint(checkpoint)
        self.critic_1.load_checkpoint(checkpoint)
        self.critic_2.load_checkpoint(checkpoint)
        print("-------------model loaded--------------")


    
    def learn(self):
        if self.memory.mem_cent < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()


        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)


        actor_loss = log_probs - critic_value  #actor_loss = (log_probs - critic_value).mean()
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()


        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = (self.scale * reward) + (self.gamma * value_)
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()


        self.updated_network_parameters()