import gym
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from util import *
class MySim_D(gym.Env):
    def __init__(self):
        self.shape = (10,10)
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Discrete(self.shape[0]*self.shape[1])
        self.actions = [(-1,1),(-1,0),(-1,-1),(0,1),(0,-1),(1,1),(1,0),(1,-1)]
        self.goal = (9,9)
        self.pro = preprocessing(self.shape)
        self.robot_location = None
        self.map = Map_D(self.goal,self.shape)
        self.dies_times = 0   # Times of death
        self.arrive_times = 0 # Times of arriving
        self.over_times = 0
        self.step_times = 0   # Calculate the steps of agent
        self.max_step = 200   # Max step

        # Add dynamic_obstacles as an empty list initially for visualize
        self.dynamic_obstacles = []  

    def step(self, action):
        # Get the new observation based on the action
        observation = self.get_obs(action)
        # Update the positions of dynamic obstacles
        self.map.update_dynamic_obstacles()
        self.dynamic_obstacles.append(self.map.dynamic_obs_position)
        # Update the robot's path with the new observation
        self.path_xdata.append(self.pro.index2loc(observation)[0])
        self.path_ydata.append(self.pro.index2loc(observation)[1])
        # Increment the step count
        self.step_times += 1
        # Calculate the reward for the current ste
        reward = self.get_reward(observation)
        # Check if the episode is done (either goal reached or collision)
        done,typ = self.get_done(observation)
        terminated = done and (typ == 0)  # End when reached goal
        truncated = done and (typ in [1, 2])  # End due to collision
        # Return the step information including step count and path length
        info = {"step_count": self.step_times, "path_length": len(self.path_xdata)}
        return observation, reward, terminated, truncated, info
    
    def reset(self,seed = None, options = None):
      super().reset(seed=seed)
      while True:
        state = self.observation_space.sample()
        if self.pro.index2loc(state) not in self.map.obstacle_list  and self.pro.index2loc(state) != self.goal:
          break
      self.step_times = 0
      self.robot_location = state
      self.path_xdata = [self.pro.index2loc(state)[0]]
      self.path_ydata = [self.pro.index2loc(state)[1]]
      obs = self.robot_location
      info = {"initial_location": self.robot_location}
      return obs, info
    
    def render(self, mode = 'human'):
        if mode != 'human':
          raise NotImplementedError("Only 'human' mode is supported.")

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, self.shape[1])
        ax.set_ylim(0, self.shape[0])
        ax.set_aspect('equal') 
        ax.grid(True, which='both', color='lightgray', linestyle='--', linewidth=0.5)

        # for (ox, oy) in self.map.obstacle_list:
        #   ax.add_patch(plt.Rectangle((oy, ox), 1, 1, color="black"))  # Obstacle cells in red

        # Render circular obstacles
        for (center, radius) in self.map.circle_obstacles:
          circle = plt.Circle((center[1] + 0.5, center[0] + 0.5), radius, color="black", fill=True, linestyle="--")
          ax.add_patch(circle)
        
        # Render rectangular obstacles
        for (x_start, y_start), (width, height) in self.map.rectangle_obstacles:
          rect = plt.Rectangle((y_start, x_start), height, width, color="black", fill=True, linestyle="--")
          ax.add_patch(rect)

        ax.scatter(self.goal[1], self.goal[0], c='green', label='Goal', s=100, marker='*')

        # Render dynamic obstacles
        # for (dx, dy), _ in self.map.dynamic_obstacles:
        #   ax.add_patch(plt.Rectangle((dy, dx), 1, 1, color="purple", fill=True))

        # current_loc = self.pro.index2loc(self.robot_location)
        # ax.scatter(current_loc[1], current_loc[0], c='blue', label='Robot', s=100, marker='o')

        # if len(self.path_xdata) > 1:
        #     ax.plot(self.path_ydata, self.path_xdata, c='blue', label='Path')

        ax.legend(loc='lower left')
        ax.set_title("Environment State")
        return fig, ax
    
    def seed(self, seed = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_reward(self,observation):
      loc_goal = self.goal
      loc_state = self.pro.index2loc(observation)
      total_rewards = 0
      if observation == self.pro.loc2index(self.goal):
        total_rewards += 500
      elif self.pro.index2loc(observation) in self.map.obstacle_list:
        total_rewards -= 1000
      else:
        distance = (((loc_goal[0]-loc_state[0])**2)+((loc_goal[1]-loc_state[1])**2))**0.5 #L2 loss
        total_rewards -= 0.2 * distance
      return total_rewards
    
    def get_done(self,observation):
      goal_index = self.pro.loc2index(self.goal)
      current_loc = self.pro.index2loc(observation)
      if observation == goal_index:
          self.arrive_times += 1
          print('cumulative times:{0}, steps:{1}'.format(self.arrive_times, self.step_times))
          return True, 0
      elif current_loc in self.map.obstacle_list:
          self.dies_times += 1
          print('Die times:{0}, steps:{1}'.format(self.dies_times, self.step_times))
          return True, 1
      elif self.step_times > self.max_step:
          self.over_times += 1
          print('Over max step times:{}'.format(self.over_times))
          return True, 2
      else:
          return False, 3
    
    def get_obs(self,action):
      now = self.pro.index2loc(self.robot_location)
      x = now[0]+self.actions[action][0]
      y = now[1]+self.actions[action][1]
      x = np.clip(x,0,self.shape[0]-1)  # Maybe need to adjust lower limit
      y = np.clip(y,0,self.shape[1]-1)  # Maybe need to adjust lower limit
      new_state=(x,y)

      self.robot_location = self.pro.loc2index(new_state)

      return self.robot_location

env = MySim_D()
check_env(env)
env.render(mode = 'human')

# if __name__=="__main__":
#     env = MySim_D()
#     print(env)
#     check_env(env)
