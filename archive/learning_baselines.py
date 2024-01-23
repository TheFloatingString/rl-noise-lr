from typing import Callable, Dict
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

class CustomWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(shape=(4,), low=-np.inf, high=np.inf)
    def observation(self, obs):
        obs += np.random.uniform(low=-0.01, high=0.01, size=(4,))
        # print(type(obs))
        # raise KeyError
        return obs
    
class CustomWrapperZero(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(shape=(4,), low=-np.inf, high=np.inf)
    def observation(self, obs):
        # obs += np.random.uniform(low=-0.01, high=0.01, size=(4,))
        # # print(type(obs))
        # raise KeyError
        return obs
    
override_lr = 0.003

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        global override_lr
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        # print(progress_remaining)
        return override_lr

    return func

# Initial learning rate of 0.001
# model = PPO("MlpPolicy", "CartPole, learning_rate=linear_schedule(0.001), verbose=1)
env = CustomWrapper(gym.make("CartPole-v1", render_mode=None))
model = PPO("MlpPolicy", env=env, learning_rate=0, verbose=1)
# policy = model.policy
# policy.optimizer.
# policy.optimizer.learning_rate = 0.3

# def mutate(params: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
#     """Mutate parameters by adding normal noise to them"""
#     return dict((name, param + th.randn_like(param)) for name, param in params.items())

print(model.learning_rate)
model.learning_rate = 0.001
print(model.learning_rate)

for i in range(5):
    model.learn(total_timesteps=200)
    # print(model.policy.state_dict().keys())
    # model.policy.load_state_dict({"learning_rate":0.003}, strict=False)
    # model.set_parameters({"learning_rate":np.asarray(0.003)}, exact_match=False)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=25)
    # print(model.learning_rate)
    print(model.learning_rate)
    print(mean_reward, std_reward)
    # override_lr = 0.1

model.save("saved_model")

loaded_model = PPO.load("saved_model.zip")
print("loaded model.")
mean_reward, std_reward = evaluate_policy(loaded_model, gym.make("CartPole-v1", render_mode=None), n_eval_episodes=25)
loaded_model.set_env(env=CustomWrapperZero(gym.make("CartPole-v1", render_mode=None)))
print(mean_reward, std_reward)
for i in range(5):
    loaded_model.learn(total_timesteps=200)
    mean_reward, std_reward = evaluate_policy(loaded_model, loaded_model.get_env(), n_eval_episodes=25)
    print(mean_reward, std_reward)


# By default, `reset_num_timesteps` is True, in which case the learning rate schedule resets.
# progress_remaining = 1.0 - (num_timesteps / total_timesteps)
# model.learn(total_timesteps=10000, reset_num_timesteps=True)