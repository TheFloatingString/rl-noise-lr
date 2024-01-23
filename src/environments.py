import gymnasium as gym
import numpy as np

class NoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_param, shape):
        super().__init__(env)
        self.noise_param = noise_param
        self.shape = shape
        self.observation_space = gym.spaces.Box(shape=self.shape, low=-np.inf, high=np.inf)
        self.iter = 0
    def observation(self, obs):
        obs += np.random.uniform(low=-self.noise_param, high=self.noise_param, size=self.shape)
        if self.iter > 12000:
            # obs += np.random.uniform(low=0, high=0, size=self.shape)
            return obs
        if self.iter > 10000:
            obs += np.random.uniform(low=-0.5, high=0.5, size=self.shape)
            return obs
        self.iter += 1
        return obs