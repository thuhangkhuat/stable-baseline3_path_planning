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
    # Load the model
    model = DQN.load('test_DQN_dynamic_obs_circle')

    for i in range(10):
        obs, info = env.unwrapped.reset()
        terminated = False
        truncated = False
        print(f"Starting Episode {i+1}")
        while not (terminated or truncated):
            action, _ = model.predict(observation=obs)
            obs, reward, terminated, truncated, info = env.unwrapped.step(action)

        path_xdata = env.unwrapped.path_xdata
        path_ydata = env.unwrapped.path_ydata
        # Create a figure and axis for plotting
        fig, ax = env.unwrapped.render(mode='human')
        # Line object to update in the animation
        line, = ax.plot([], [], marker='H', color='blue', label='Path',linestyle='--',linewidth=2)
        # Update function for animation
        for frame in env.unwrapped.dynamic_obstacles:
            dynamic_obs_patches = [plt.Rectangle((ob[0], ob[1]), 1, 1, color="purple", fill=True) for ob in frame]
        for patch in dynamic_obs_patches:
            ax.add_patch(patch)
        def update(frame):
            # Update dynamic obstacles (move obstacles)
            dynamic_obs_positions = env.unwrapped.dynamic_obstacles[frame-1]
            for patch, pos in zip(dynamic_obs_patches, dynamic_obs_positions):
                patch.set_xy((pos[0], pos[1]))
            line.set_data(path_ydata[:frame+1], path_xdata[:frame+1])
            return line, *dynamic_obs_patches

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(path_xdata), interval=500, blit=False)

        # Display the animation
        plt.show()

        print(f"Path for Episode {i+1}: X - {path_xdata}, Y - {path_ydata}")
