from agents import ACAgent, KANACAgent, PPOAgent, KANPPOAgent, DQNAgent
from agents import Trainer, PPOTrainer, QTrainer
import gym

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

HP = {}
HP['alpha'] = 0.0003
HP['input_dims'] = 8
HP['gamma'] = 0.98
HP['fc1'] = 16
HP['fc2'] = 32
HP['n_action'] = 4
HP['node_feature'] = 3
HP['lr'] = 0.0003


env = gym.make("LunarLander-v2")
#agentt = PPOAgent(input_dims=env.observation_space.shape[0], n_actions=2, alpha=0.0003, gamma=0.9)
agentt = DQNAgent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=8, lr=0.001)

#trainer = PPOTrainer(agent=agentt, env=env, N=20, n_games=1000, n_epochs=20)
trainer = QTrainer(agent=agentt, env=env, n_episode=500)

trainer.train("result\\kan-dqn.png")
#agentt.save_model("models")