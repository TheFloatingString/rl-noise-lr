import gymnasium as gym
import numpy as np

class NoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_param, shape):
        super().__init__(env)
        self.noise_param = noise_param
        self.shape = shape
        self.observation_space = gym.spaces.Box(shape=self.shape, low=-np.inf, high=np.inf)
    def observation(self, obs):
        obs += np.random.uniform(low=-self.noise_param, high=self.noise_param, size=self.shape)
        return obs