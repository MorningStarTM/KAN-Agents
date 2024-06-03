from .utils import plot_learning
import GPUtil
import matplotlib.pyplot as plt
import time
import numpy as np
from .csv_logger import CSVLogger


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
        self.total_duration = 0
        self.c_point = 0

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

        filename = "result.png"
        np.save("reward", self.history)
        plot_learning(self.history, filename=filename, window=50)
        CSVLogger(agent=self.agent, epochs=self.epochs, c_point=self.c_point)
