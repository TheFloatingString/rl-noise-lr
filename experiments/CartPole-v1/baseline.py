import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from src.environments import NoiseWrapper 
from typing import Callable
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": -1,
    "env_name": "CartPole-v1"
}
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return 0.5*(progress_remaining * initial_value)+0.5*initial_value
    return func
tmp_path = "/tmp/sb3_log/"
env = NoiseWrapper(gym.make("CartPole-v1", render_mode
=None), noise_param=0, shape=(4,))
model = PPO("MlpPolicy", env=env, verbose=1, learning_rate=linear_schedule(0.003), tensorboard_log='./tmp_tensorboard', n_steps=100)
for i in range(1):
    print(model.learning_rate)
    model.learn(total_timesteps=20000, log_interval=1, progress_bar=True)
