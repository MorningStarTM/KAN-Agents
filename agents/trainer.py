from .utils import plot_learning, plot_learning_curve, plotLearning
import GPUtil
import matplotlib.pyplot as plt
import time
import numpy as np
from .csv_logger import CSVLogger
from collections import deque


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
        

    def monitor_resources(self):
        """
        Monitor and log the RAM and GPU usage.
        """
        # GPU usage
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_usage = max(gpu.memoryUtil * 100 for gpu in gpus)  # Get max GPU usage in percentage
        else:
            gpu_usage = 0
        self.gpu_usage = np.append(self.gpu_usage, gpu_usage)  # Append GPU usage to the array

    def train(self):
        total_start_time = time.time()

        for i in range(self.epochs):
            self.monitor_resources()  # Monitor resources at the beginning of each episode
            episode_start_time = time.time()

            done = False
            score = 0
            observation, _ = self.env.reset()

            while not done:
                action = self.agent.choose_action(observation)
                observation_, reward, done, _, _ = self.env.step(action)
                self.agent.learn(observation, reward, observation_, done)
                observation = observation
                score += reward
            if score >= 200:
                self.c_point = i
                break

            self.history = np.append(self.history, score)  # Append the score to the history array
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            self.time_history = np.append(self.time_history, episode_duration)  # Append episode duration

            print(f"Episode {i} Score {score}")
        
        total_end_time = time.time()
        self.total_duration = total_end_time - total_start_time
        print(f"Total Time: {self.total_duration} seconds")

        filename = "result\\result_kan.png"
        #np.save("reward", self.history)
        plot_learning(self.history, filename=filename, window=50)
        self.csvlogger = CSVLogger(agent=self.agent, epochs=self.epochs, c_point=self.c_point, time=self.total_duration)
        self.csvlogger.log()

    

class PPOTrainer:
    def __init__(self, agent, env, N, n_games, n_epochs):
        self.agent = agent
        self.env = env
        self.n_games = n_games
        self.N = N
        self.n_epochs = n_epochs
        self.best_score = env.reward_range[0]
        self.score_history = []

        self.learn_iters = 0
        self.avg_score = 0
        self.n_steps = 0
        

    def train(self, figure_file):
        total_start_time = time.time()


        for i in range(self.n_games):
            observation, _ = self.env.reset()
            done = False
            score = 0
            while not done:
                action, prob, val = self.agent.choose_action(observation)
                observation_, reward, done, info, _ = self.env.step(action)
                self.n_steps += 1
                score += reward
                self.agent.remember(observation, action, prob, val, reward, done)
                if self.n_steps % self.N == 0:
                    self.agent.learn()
                    self.learn_iters += 1
                observation = observation_
            self.score_history.append(score)
            self.avg_score = np.mean(self.score_history[-100:])

            if self.avg_score > self.best_score:
                self.best_score = self.avg_score
                self.agent.save_models()

            
            print('episode', i, 'score %.1f' % score, 'avg score %.1f' % self.avg_score,
                    'time_steps', self.n_steps, 'learning_steps', self.learn_iters)
            
            if self.avg_score >= 200:
                break
        total_end_time = time.time()
        self.total_duration = total_end_time - total_start_time

        
        self.csvlogger = CSVLogger(agent=self.agent, epochs=self.n_epochs, c_point=i, time=self.total_duration)
        self.csvlogger.log()

        x = [i+1 for i in range(len(self.score_history))]
        plot_learning_curve(x, self.score_history, figure_file)


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
                self.agent.save_model("models\\dqn.pth")

            print('episode ', i, 'score %.2f' % score,
                    'average score %.2f' % avg_score,
                    'epsilon %.2f' % self.agent.epsilon)
            
        total_end_time = time.time()
        self.total_duration = total_end_time - total_start_time

        
        self.csvlogger = CSVLogger(agent=self.agent, epochs=self.n_episode, c_point=i, time=self.total_duration)
        self.csvlogger.log()

        x = [i+1 for i in range(self.n_episode)]
        filename = 'lunar_lander.png'
        plotLearning(x, scores, eps_history, filename)
