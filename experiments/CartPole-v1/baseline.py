import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from src.environments import NoiseWrapper 
from wandb.integration.sb3 import WandbCallback
import wandb
from typing import Callable
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": -1,
    "env_name": "CartPole-v1"
}
# run = wandb.init(
#     project="rl-noise-rl",
#     config = config,
#     sync_tensorboard=True,
#     monitor_gym=False,
#     save_code=False
# )
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
        # if progress_remaining < 0.5: return 0
        return 0.5*(progress_remaining * initial_value)+0.5*initial_value

    return func
tmp_path = "/tmp/sb3_log/"
# logger = configure(tmp_path, ["csv", "tensorboard"])
env = NoiseWrapper(gym.make("CartPole-v1", render_mode
=None), noise_param=0, shape=(4,))
# env = gym.make("CartPole-v1", render_mode=None)
model = PPO("MlpPolicy", env=env, verbose=1, learning_rate=linear_schedule(0.003), tensorboard_log='./tmp_tensorboard', n_steps=100)
# params = model.get_parameters()
# print(params["policy.optimizer"]["param_groups"][0]["lr"])
# params["policy.optimizer"]["param_groups"][0]["lr"] = 0.01
# print(params["policy.optimizer"]["param_groups"][0]["lr"])
# model.set_parameters(params, exact_match=True)
# new_params = model.get_parameters()
# print(new_params["policy.optimizer"]["param_groups"][0]["lr"])
# raise KeyError
# model.set_logger(logger)
for i in range(1):
    # model.learning_rate = 1000
    print(model.learning_rate)
    model.learn(total_timesteps=20000, log_interval=1, progress_bar=True)
    # model.learn(total_timesteps=100,callback=WandbCallback(
    #     gradient_save_freq=1,
    #     verbose=1
    # ))
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # print(i,mean_reward, std_reward)
# run.finish()
# PPO.load()
# model.set_lear