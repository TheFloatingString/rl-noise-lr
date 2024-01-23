import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from src.environments import NoiseWrapper 
from wandb.integration.sb3 import WandbCallback
import wandb
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
tmp_path = "/tmp/sb3_log/"
# logger = configure(tmp_path, ["csv", "tensorboard"])
env = NoiseWrapper(gym.make("CartPole-v1", render_mode
=None), noise_param=0, shape=(4,))
# env = gym.make("CartPole-v1", render_mode=None)
model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log='./tmp_tensorboard')
# model.set_logger(logger)
for i in range(3):
    model.learning_rate += 0.001
    print(model.learning_rate)
    model.learn(total_timesteps=10000, log_interval=1, progress_bar=True)
    # model.learn(total_timesteps=100,callback=WandbCallback(
    #     gradient_save_freq=1,
    #     verbose=1
    # ))
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(i,mean_reward, std_reward)
# run.finish()