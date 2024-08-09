from .utils import plot_learning, plot_learning_curve, AC_result, plotLearning
import GPUtil
import matplotlib.pyplot as plt
import time
import numpy as np
from .csv_logger import CSVLogger
from collections import deque
import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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
        self.history = np.array([])  # Initialize as empty NumPy array
        self.time_history = np.array([])  # Initialize as empty NumPy array
        self.gpu_usage = np.array([])  # Initialize as empty NumPy array
        self.c_point = 0
        self.total_duration = 0
        self.best_score = 0
        self.score_history = []

    def train(self, filename):
        total_start_time = time.time()

        for i in range(self.epochs):

            done = False
            score = 0
            observation, _ = self.env.reset()
            while not done:
                observation = torch.tensor(observation, dtype=torch.float)
                action = self.agent.choose_action(observation)
                observation_, reward, done, info, _ = self.env.step(action)
                observation_ = torch.tensor(observation_, dtype=torch.float)
                self.agent.learn(observation, reward, observation_, done)
                observation = observation_
                score += reward
            self.score_history.append(score)

            if score > self.best_score:
                self.best_score = score
                self.agent.save_model("result\\lunar")
                print(f'Best score {self.best_score}  Saving model')  # Append episode duration

            if score >= 210.0:
                break

            print('episode: ', i, 'score: %.3f' % score)
        
        total_end_time = time.time()
        self.total_duration = total_end_time - total_start_time
        print(f"Total Time: {self.total_duration} seconds")

        
        #np.save("reward", self.history)
        AC_result(self.score_history, filename=filename, window=5)
        self.csvlogger = CSVLogger(agent=self.agent, epochs=self.epochs, c_point=self.c_point, time=self.total_duration)
        self.csvlogger.log()

    

class PPO:
    def __init__(self,
                 env_name, 
                 has_continuous_action_space=False, 
                 max_ep_len=400,
                 max_training_timesteps = int(1e5),
                 save_model_freq = int(2e4),
                 action_std=None,
                 K_epochs=40,
                 eps_clip=0.2,
                 gamma=0.99,
                 lr_actor=0.0003,
                 lr_critic=0.001,):
        
        self.env_name = "LunarLander-v2"
        self.has_continuous_action_space = has_continuous_action_space

        self.max_ep_len = max_ep_len                    # max timesteps in one episode
        self.max_training_timesteps = max_training_timesteps   # break training loop if timeteps > max_training_timesteps

        self.print_freq = self.max_ep_len * 4     # print avg reward in the interval (in num timesteps)
        self.log_freq = self.max_ep_len * 2       # log avg reward in the interval (in num timesteps)
        self.save_model_freq = save_model_freq      # save model frequency (in num timesteps)

        self.action_std = action_std

        self.update_timestep = self.max_ep_len * 4      # update policy every n timesteps
        self.K_epochs = K_epochs               # update policy for K epochs
        self.eps_clip = eps_clip              # clip parameter for PPO
        self.gamma = gamma             # discount factor

        self.lr_actor = lr_actor       # learning rate for actor network
        self.lr_critic = lr_critic      # learning rate for critic network

        self.random_seed = 0     






class QTrainer:
    def __init__(self, agent, env, n_episode=1000):
        self.agent = agent
        self.env = env
        self.n_episode = n_episode
        self.best_score = 0


    def train(self, filename):
        scores, eps_history = [], []
        total_start_time = time.time()

        for i in range(self.n_episode):

            score = 0
            done = False
            observation,_ = self.env.reset()
            while not done:
                action = self.agent.choose_action(observation)
                observation_, reward, done, info, _ = self.env.step(action)
                score += reward
                self.agent.store_transition(observation, action, reward, 
                                        observation_, done)
                self.agent.learn()
                observation = observation_
            scores.append(score)
            eps_history.append(self.agent.epsilon)

            avg_score = np.mean(scores[-100:])
            if self.best_score < score:
                self.best_score = score
                self.agent.save_model(f"models\\{self.agent.name}.pth")

            print('episode ', i, 'score %.2f' % score,
                    'average score %.2f' % avg_score,
                    'epsilon %.2f' % self.agent.epsilon)
            
        total_end_time = time.time()
        self.total_duration = total_end_time - total_start_time

        
        self.csvlogger = CSVLogger(agent=self.agent, epochs=self.n_episode, c_point=i, time=self.total_duration)
        self.csvlogger.log()

        x = [i+1 for i in range(self.n_episode)]
        
        plotLearning(x, scores, eps_history, filename)
