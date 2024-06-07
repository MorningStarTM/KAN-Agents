from agents import ACAgent, KANACAgent, PPOAgent, KANPPOAgent, DQNAgent, KDQNAgent
from agents import Trainer, PPOTrainer, QTrainer
import gym

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


env = gym.make('CartPole-v0')
agentt = KANACAgent(alpha=0.0003, input_dims=4)
#agentt = DQNAgent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
#                  input_dims=8, lr=0.0003)

trainer = Trainer(agent=agentt, env=env, epochs=1000)
#trainer = QTrainer(agent=agentt, env=env, n_episode=500)

trainer.train("result\\kan-actor-critic.png")
#agentt.save_model("models")

