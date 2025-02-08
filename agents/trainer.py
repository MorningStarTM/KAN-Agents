import GPUtil
import matplotlib.pyplot as plt
import time
import numpy as np
from .csv_logger import CSVLogger
from collections import deque
import torch
import gym
from datetime import datetime
from agents.dqn import DQNAgent, KDQNAgent
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
    def __init__(self,
                 env_name, 
                 agent,
                 has_continuous_action_space=False, 
                 max_ep_len=400,
                 max_training_timesteps = int(1e5),
                 save_model_freq = int(2e4),
                 action_std=None):
        
        self.env_name = env_name
        self.agent = agent
        self.env = gym.make(env_name)
        self.has_continuous_action_space = has_continuous_action_space

        self.max_ep_len = max_ep_len                    # max timesteps in one episode
        self.max_training_timesteps = max_training_timesteps   # break training loop if timeteps > max_training_timesteps

        self.print_freq = self.max_ep_len * 4     # print avg reward in the interval (in num timesteps)
        self.log_freq = self.max_ep_len * 2       # log avg reward in the interval (in num timesteps)
        self.save_model_freq = save_model_freq      # save model frequency (in num timesteps)

        self.action_std = action_std

        self.update_timestep = self.max_ep_len * 4      # update policy every n timesteps

        self.random_seed = 0     
        self.log_dir = "models\\PPO_logs"
        self.log_f_name = ""
        self.checkpoint_path = ""
    
    def init_train(self):
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.log_dir = self.log_dir + '/' + self.env_name + '/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        run_num = 0
        current_num_files = next(os.walk(self.log_dir))[2]
        run_num = len(current_num_files)


        #### create new log file for each run 
        self.log_f_name = self.log_dir + '/PPO_' + self.env_name + "_log_" + str(run_num) + ".csv"

        print("current logging run number for " + self.env_name + " : ", run_num)
        print("logging at : " + self.log_f_name)

        run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

        directory = "PPO_preTrained"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/' + self.env_name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)


        checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(self.env_name, self.random_seed, run_num_pretrained)
        print("save checkpoint path : " + checkpoint_path)



    def train(self):
        self.init_train()
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        log_f = open(self.log_f_name,"w+")
        log_f.write('episode,timestep,reward\n')


        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0

        while time_step <= self.max_training_timesteps:
            
            state, _ = self.env.reset()
            current_ep_reward = 0

            for t in range(1, self.max_ep_len+1):
                
                # select action with policy
                action = self.agent.select_action(state)
                state, reward, done, _, _ = self.env.step(action)
                
                # saving reward and is_terminals
                self.agent.buffer.rewards.append(reward)
                self.agent.buffer.is_terminals.append(done)
                
                time_step +=1
                current_ep_reward += reward

                # update PPO agent
                if time_step % self.update_timestep == 0:
                    self.agent.update()

                # log in logging file
                if time_step % self.log_freq == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % self.print_freq == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0
                    
                # save model weights
                if time_step % self.save_model_freq == 0:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + self.checkpoint_path)
                    self.agent.save(self.checkpoint_path)
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")
                    
                # break; if the episode is over
                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1


        log_f.close()
        self.env.close()


class QTrainer:
    def __init__(self, agent:DQNAgent, env, n_episode=1000):
        self.agent = agent
        self.env = env
        self.n_episode = n_episode
        self.best_score = 0
        self.scores = []  # To store scores for each episode
        self.eps_history = []  # To store epsilon values for each episode


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
                self.agent.save_model(f"models\\{self.agent.name}.pth")

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
    
