from .utils import plot_learning
import GPUtil
import matplotlib.pyplot as plt
import time

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
        self.history = []
        self.score = []
        self.time_history = []
        self.gpu_usage = []
        self.total_duration = 0

    def monitor_resources(self):
        """
        Monitor and log the RAM and GPU usage.
        """
        # RAM usage
        
        # GPU usage
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_usage = max(gpu.memoryUtil * 100 for gpu in gpus)  # Get max GPU usage in percentage
        else:
            gpu_usage = 0
        self.gpu_usage.append(gpu_usage)

   
    def train(self):
        self.monitor_resources()
        total_start_time = time.time()

        score = 0
        for i in range(self.epochs):
            episode_start_time = time.time()

            done = False
            score = 0
            observation, _ = self.env.reset()

            while not done:
                action = self.agent.choose_action(observation)
                observation_, reward, done, _, _= self.env.step(action)
                self.agent.learn(observation, reward, observation_, done)
                observation = observation
                score += reward

            self.history.append(score)
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            self.time_history.append(episode_duration)
            print(f"Episode {i} Score {score}")
        
        total_end_time = time.time()
        self.total_duration = total_end_time - total_start_time
        print(f"Total Time : {self.total_duration}")

        self.plot_resource_usage()
        
        filename = "result.png"
        plot_learning(self.history, filename=filename, window=50)