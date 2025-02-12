import GPUtil
import matplotlib.pyplot as plt
import time
import numpy as np
from .csv_logger import CSVLogger
from collections import deque
import torch
import gymnasium as gym
from datetime import datetime
from agents.dqn import DQNAgent, KDQNAgent
from agents.ppo import PPOAgent
from utils.logger import CustomLogger

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


logger = CustomLogger(log_dir="logs", log_file_prefix="experiment")



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
        self.csvlogger = CSVLogger(agent=self.agent, epochs=self.epochs, c_point=self.c_point, time=self.total_duration)
        self.csvlogger.log()

    


class PPOTrainer:
    def __init__(self, agent:PPOAgent, env_name, config):
        self.agent = agent
        self.env_name = env_name
        self.config = config

        # Extract config parameters
        self.has_continuous_action_space = config['has_continuous_action_space']
        self.max_ep_len = config['max_ep_len']
        self.max_training_timesteps = config['max_training_timesteps']
        self.print_freq = config['print_freq']
        self.log_freq = config['log_freq']
        self.save_model_freq = config['save_model_freq']
        self.action_std_decay_rate = config['action_std_decay_rate']
        self.action_std_decay_freq = config['action_std_decay_freq']
        self.min_action_std = config['min_action_std']
        self.update_timestep = config['update_timestep']
        self.K_epochs = config['K_epochs']
        self.eps_clip = config['eps_clip']
        self.gamma = config['gamma']
        self.lr_actor = config['lr_actor']
        self.lr_critic = config['lr_critic']
        self.random_seed = config['random_seed']

        self.scores = []  # Stores total episode rewards
        self.timesteps = []  # Stores timestep count per episode

        # Environment
        self.env = gym.make(env_name)
        logger.log("info", f"{env_name} loaded")
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0] if self.has_continuous_action_space else self.env.action_space.n

        # Logging setup
        self.log_dir = os.path.join("result", self.agent.name)
        os.makedirs(self.log_dir, exist_ok=True)
        logger.log("info", f"{self.log_dir} is created")

        self.log_dir = os.path.join(self.log_dir, self.env_name)
        os.makedirs(self.log_dir, exist_ok=True)
        logger.log("info", f"{self.log_dir} is created")

        current_num_files = next(os.walk(self.log_dir))[2]
        run_num = len(current_num_files)

        self.log_f_name = os.path.join(self.log_dir, f'{self.agent.name}_{self.env_name}_log_{run_num}.csv')

        logger.log("info", f"current logging run number for {self.env_name} : {run_num}")
        logger.log("info", f"logging at : {self.log_f_name}")

        # Checkpoint setup
        run_num_pretrained = 0
        directory = "result"
        os.makedirs(directory, exist_ok=True)

        directory = os.path.join(directory, self.env_name)
        os.makedirs(directory, exist_ok=True)

        self.checkpoint_path = os.path.join(directory, f"{self.agent.name}_{self.env_name}_{self.random_seed}_{run_num_pretrained}.pth")
        logger.log("info", f"save checkpoint path : {self.checkpoint_path}")

        # Training metadata logging
        logger.log("info", f"max training timesteps : {self.max_training_timesteps}")
        logger.log("info", f"max timesteps per episode : {self.max_ep_len}")
        logger.log("info", f"model saving frequency : {self.save_model_freq} timesteps")
        logger.log("info", f"log frequency : {self.log_freq} timesteps")
        logger.log("info", f"printing average reward over episodes in last : {self.print_freq} timesteps")
        logger.log("info", f"state space dimension : {self.state_dim}")
        logger.log("info", f"action space dimension : {self.action_dim}")

        if self.has_continuous_action_space:
            logger.log("info", f"Initializing a continuous action space policy")
            logger.log("info", f"starting std of action distribution : {config['action_std']}")
            logger.log("info", f"decay rate of std of action distribution : {self.action_std_decay_rate}")
            logger.log("info", f"minimum std of action distribution : {self.min_action_std}")
            logger.log("info", f"decay frequency of std of action distribution : {self.action_std_decay_freq} timesteps")
        else:
            logger.log("info", "Initializing a discrete action space policy")

        logger.log("info", f"PPO update frequency : {self.update_timestep} timesteps")
        logger.log("info", f"PPO K epochs : {self.K_epochs}")
        logger.log("info", f"PPO epsilon clip : {self.eps_clip}")
        logger.log("info", f"discount factor (gamma) : {self.gamma}")
        logger.log("info", f"optimizer learning rate actor : {self.lr_actor}")
        logger.log("info", f"optimizer learning rate critic : {self.lr_critic}")

        if self.random_seed:
            logger.log("info", f"setting random seed to {self.random_seed}")
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            self.env.seed(self.random_seed)

    def train(self):
        start_time = datetime.now().replace(microsecond=0)
        logger.log("info", f"Started training at (GMT) : {start_time}")

        log_f = open(self.log_f_name, "w+")
        log_f.write('episode,timestep,reward\n')

        # Training loop variables
        print_running_reward = 0
        print_running_episodes = 0
        log_running_reward = 0
        log_running_episodes = 0
        time_step = 0
        i_episode = 0

        while time_step <= self.max_training_timesteps:
            state, _ = self.env.reset()
            current_ep_reward = 0

            for t in range(1, self.max_ep_len + 1):
                action = self.agent.select_action(state)
                state, reward, done, _, _ = self.env.step(action)

                self.agent.buffer.rewards.append(reward)
                self.agent.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += reward

                # Update PPO
                if time_step % self.update_timestep == 0:
                    self.agent.update()

                # Decay action standard deviation (if applicable)
                if self.has_continuous_action_space and time_step % self.action_std_decay_freq == 0:
                    self.agent.decay_action_std(self.action_std_decay_rate, self.min_action_std)

                # Log results
                if time_step % self.log_freq == 0:
                    log_avg_reward = log_running_reward / max(log_running_episodes, 1)
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write(f'{i_episode},{time_step},{log_avg_reward}\n')
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # Print results
                if time_step % self.print_freq == 0:
                    print_avg_reward = print_running_reward / max(print_running_episodes, 1)
                    print_avg_reward = round(print_avg_reward, 2)

                    logger.log("info", f"Episode: {i_episode}, Timestep: {time_step}, Average Reward: {print_avg_reward}")

                    print_running_reward = 0
                    print_running_episodes = 0

                # Save model checkpoint
                if time_step % self.save_model_freq == 0:
                    logger.log("info", f"saving model at : {self.checkpoint_path}")
                    self.agent.save(self.checkpoint_path)
                    logger.log("info", "model saved")
                    logger.log("info", f"Elapsed Time  : {datetime.now().replace(microsecond=0) - start_time}")

                if done:
                    break
            
            self.scores.append(current_ep_reward)
            self.timesteps.append(time_step)

            print_running_reward += current_ep_reward
            print_running_episodes += 1
            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        self.get_results(f"result\\{self.agent.name}\\{self.env_name}")
        log_f.close()
        self.env.close()


    
    def get_results(self, filename_prefix):
        """
        Get the collected training results and save them as .npy files.
        """
        np.save(f"{filename_prefix}\\scores.npy", np.array(self.scores))
        np.save(f"{filename_prefix}\\timesteps.npy", np.array(self.timesteps))
        logger.log("info", f"Results saved: {filename_prefix}_scores.npy, {filename_prefix}_timesteps.npy")
        


class QTrainer:
    def __init__(self, agent:DQNAgent, env_name, n_episode=1000):
        self.agent = agent
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.n_episode = n_episode
        self.best_score = 0
        self.scores = []  # To store scores for each episode
        self.eps_history = []  # To store epsilon values for each episode

        logger.log("info", f"{env_name} is loaded")

    def train(self, filename):
        logger.log(f"info", "Training started")
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

            self.scores.append(score)
            self.eps_history.append(self.agent.epsilon)

            avg_score = np.mean(self.scores[-100:])
            if self.best_score < score:
                self.best_score = score
                self.agent.save_model(f"models\\{self.agent.name}_{self.env_name}.pth")

            print('episode ', i, 'score %.2f' % score,
                    'average score %.2f' % avg_score,
                    'epsilon %.2f' % self.agent.epsilon)
            
        total_end_time = time.time()
        self.total_duration = total_end_time - total_start_time

        
        self.csvlogger = CSVLogger(agent=self.agent, epochs=self.n_episode, c_point=i, time=self.total_duration)
        self.csvlogger.log()

        x = [i+1 for i in range(self.n_episode)]
        
        #plotLearning(x, scores, eps_history, filename)

        np.save(f"{filename}_scores.npy", self.scores)
        np.save(f"{filename}_eps_history.npy", self.eps_history)
        logger.log(f"info", "result histories are saved")
        
    
    def get_results(self, filename_prefix):
        """
        Get the scores and epsilon history after training, and save them as .npy files.
        :param filename_prefix: Prefix for the filenames to save scores and eps_history.
        :return: Tuple (scores, eps_history)
        """
        # Save scores and epsilons to .npy files
        np.save(f"{filename_prefix}_scores.npy", self.scores)
        np.save(f"{filename_prefix}_eps_history.npy", self.eps_history)

        print(f"Results saved: {filename_prefix}_scores.npy, {filename_prefix}_eps_history.npy")
    
