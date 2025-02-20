import os
import torch
import numpy as np
import gymnasium as gym
from robotic_agent.sac import SAC, ReplayMemory
from utils.logger import CustomLogger
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
from tqdm import tqdm

logger = CustomLogger(log_dir="logs", log_file_prefix="experiment")



class SACTrainer(object):
    def __init__(self, agent:SAC, config) -> None:
        self.agent = agent
        self.config = config
        self.env = gym.make(self.config['env_name'])
        self.rb = ReplayMemory(config["replay_size"], config["seed"])
        logger.log("info", "replay buffer initialized")

        self.scores_path = os.path.join("results", self.agent.name)
        os.makedirs(self.scores_path, exist_ok=True)
        logger.log("info", f"result folder {self.scores_path} created")

        self.scores = []

        self.checkpoint_path = os.path.join("models", self.config['env_name'])
        os.makedirs(self.checkpoint_path, exist_ok=True)
        logger.log("info", f"{self.checkpoint_path} created for model checkpoint")


        torch.manual_seed(123456)
        np.random.seed(123456)


    def train(self):
        total_numsteps = 0
        updates = 0
        
        for i_episode in itertools.count(1):
            episode_reward = 0
            episode_steps = 0
            done = False
            state, _ = self.env.reset()

            with tqdm(total=self.config["start_steps"], desc="Exploration Phase", disable=total_numsteps > self.config["start_steps"]) as pbar:
                while not done:
                    if self.config["start_steps"] > total_numsteps:
                        action = self.env.action_space.sample()  # Sample random action
                        pbar.update(1)
                    else:
                        action = self.agent.select_action(state)  # Sample action from policy

                    if len(self.rb) > self.config["batch_size"]:
                        # Number of updates per step in environment
                        for _ in range(self.config["updates_per_step"]):
                            self.agent.update_parameters(self.rb, self.config["batch_size"], updates)
                            updates += 1

                    next_state, reward, terminated, truncated, _ = self.env.step(action)  # Step
                    done = terminated or truncated
                    episode_steps += 1
                    total_numsteps += 1
                    episode_reward += reward

                    if total_numsteps % 10000 == 0:
                        
                        self.agent.save_checkpoint(self.checkpoint_path)
                        logger.log("info", f"Model saved at {total_numsteps} steps: {self.checkpoint_path}")


                    # Ignore the "done" signal if it comes from hitting the time horizon.
                    mask = 1 if truncated else float(not terminated)
                    self.rb.push(state, action, reward, next_state, mask)  # Append transition to memory
                    state = next_state

            self.scores.append(episode_reward)  
            if total_numsteps > self.config["num_steps"]:
                break

            logger.log("info",f"Episode: {i_episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, reward: {round(episode_reward, 2)}")

            print(100*"=")
            logger.log("info", "Evaluating Agent")
            if i_episode % 10 == 0 and self.config["eval"]:
                avg_reward = 0.
                episodes = 10
                for _ in range(episodes):
                    state, _ = self.env.reset()
                    episode_reward = 0
                    done = False
                    while not done:
                        action = self.agent.select_action(state, evaluate=True)
                        next_state, reward, terminated, truncated, _ = self.env.step(action)
                        done = terminated or truncated
                        episode_reward += reward
                        state = next_state
                    avg_reward += episode_reward
                avg_reward /= episodes
                print("----------------------------------------")
                logger.log("info", f"Test Episodes: {episodes}, Avg. Reward: {round(avg_reward, 2)}")
                print("----------------------------------------")

        self.env.close()
        logger.log("info", "✅ Training completed")

        np.save(self.scores_path, np.array(self.scores))
        logger.log("info", f"Final Scores saved at: {self.scores_path}")

        

