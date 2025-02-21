import gymnasium as gym
import numpy as np


class CustomObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomObsWrapper, self).__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.merge_obs(obs)  # Merge observation and achieved goal
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Modify reward function (dense reward based on distance)
        achieved_goal = obs['achieved_goal']
        desired_goal = obs['desired_goal']
        reward = -np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        # Merge observation and achieved goal
        obs = self.merge_obs(obs)

        return obs, reward, done, truncated, info

    def merge_obs(self, obs):
        """
        Merges observation and achieved goal into a single vector.
        """
        merged_obs = np.concatenate([obs['observation'], obs['achieved_goal']])
        return merged_obs

