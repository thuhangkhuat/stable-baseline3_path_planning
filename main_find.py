import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import DQN,PPO
import datetime
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env import MySim_D

if __name__=="__main__":
    
    # DQN,A2C,HER,PPO,QR-DQN,TRPO,Maskable PPO can use
    env=Monitor(MySim_D())
    model =DQN(policy="MlpPolicy", env=env,)
    model.learn(total_timesteps=200000)
    # timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    # model.save(f'test_{timestamp}')
    model.save(f'test_DQN_dynamic_obs_circle')
    episode_lengths = env.get_episode_lengths()
    episode_rewards = env.get_episode_rewards()
   
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(episode_lengths, label='Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.title('Episode Lengths Over Time')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(episode_rewards, label='Episode Rewards', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Episode Rewards Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()