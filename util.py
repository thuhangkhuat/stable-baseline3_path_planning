import numpy as np
import random
import matplotlib.pyplot as plt

class preprocessing:
  '''the function for transform location and index'''
  def __init__(self,shape):
    self.shape=shape
    self.axis_dict=self.make_dict()
  def make_dict(self):
    '''produce a dict for checking(index,location)'''
    axis_list=[]
    for x in range(self.shape[0]):
      for y in range(self.shape[1]):
        axis_list.append((x,y))
    axis_dict=dict()
    for idx,axis in enumerate(axis_list):
      axis_dict[idx]=axis
    return axis_dict

  def loc2index(self,loc):
    '''turn location to index'''
    return list(self.axis_dict.values()).index(loc)
  def index2loc(self,idx):
    '''turn index to location'''
    return self.axis_dict[idx]
# Circular obstacles: ((center_x, center_y), radius)
circle_obstacles = [
    # ((2, 2), 1),  # Circle with center (2, 2) and radius 2
    # ((5, 5), 1),  # Circle with center (5, 5) and radius 3
]

# Rectangular obstacles: ((x_start, y_start), (width, height))
rectangle_obstacles = [
    # ((0, 6), (1, 2)),  # Rectangle starting at (0, 6) with width 2 and height 3
    # ((6, 0), (2, 2)),  # Rectangle starting at (6, 0) with width 3 and height 2
]

dynamic_obstacles = [
                ("circle", (1, 1), 1, (0, 1)),      # Circle moving diagonally
                ("circle", (8, 8), 1, (0, -1)),     # Circle moving diagonally
                # ("rectangle", (2, 7), (2, 2), (0, 1)),  # Rectangle moving up
                # ("rectangle", (7, 2), (2, 2), (-1, 0))  # Rectangle moving down
                ]  
dynamic_obs_position = []
class Map_D:
  '''Discrete map function to create a grid map with obstacles'''

  def __init__(self, goal, grid_shape):
      """
      Parameters:
      - goal: Tuple[int, int], position of the goal on the grid
      - grid_shape: Tuple[int, int], dimensions of the grid (rows, cols)
      - circle_obstacles: List[Tuple[Tuple[int, int], int]], circular obstacles
                          Each element is ((center_x, center_y), radius)
      - rectangle_obstacles: List[Tuple[Tuple[int, int], Tuple[int, int]]], rectangular obstacles
                              Each element is ((x_start, y_start), (width, height))
      """
      self.goal = goal
      self.grid_shape = grid_shape
      self.circle_obstacles = circle_obstacles
      self.rectangle_obstacles = rectangle_obstacles
      self.dynamic_obstacles = dynamic_obstacles
      self.obstacle_list = self.generate_circle_obstacles() + self.generate_rectangle_obstacles()
      self.map = self.create_grid_map()
      self.dynamic_obs_position = dynamic_obs_position

  def generate_circle_obstacles(self):
      '''Generate grid points occupied by circular obstacles'''
      obstacle_list = set()
      for (center_x, center_y), radius in self.circle_obstacles:
          # Add all grid points that fall within the circle
          for x in range(self.grid_shape[0]):
              for y in range(self.grid_shape[1]):
                  if np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2) <= radius:
                      obstacle_list.add((x, y))

      # Ensure the goal point is not blocked
      if self.goal in obstacle_list:
          obstacle_list.remove(self.goal)

      return list(obstacle_list)

  def generate_rectangle_obstacles(self):
      '''Generate grid points occupied by rectangular obstacles'''
      obstacle_list = set()
      for (x_start, y_start), (width, height) in self.rectangle_obstacles:
          # Add all grid points within the rectangle
          for x in range(x_start, x_start + width):
              for y in range(y_start, y_start + height):
                  # Ensure the point is within grid bounds
                  if 0 <= x < self.grid_shape[0] and 0 <= y < self.grid_shape[1]:
                      obstacle_list.add((x, y))

      # Ensure the goal point is not blocked
      if self.goal in obstacle_list:
          obstacle_list.remove(self.goal)

      return list(obstacle_list)
  def update_dynamic_obstacles(self):
    '''Update the positions of dynamic obstacles'''
    updated_obstacles = []
    obs_position = []
    for shape_type, position, size, velocity in self.dynamic_obstacles:
        if shape_type == "circle":
            cx, cy = position
            vx, vy = velocity
            new_cx = cx + vx
            new_cy = cy + vy

            # Bounce if hitting boundary
            if new_cx < 0 or new_cx + size >= self.grid_shape[0]:
                vx *= -1
                new_cx = max(0, min(self.grid_shape[0] - size, new_cx))
            if new_cy < 0 or new_cy + size >= self.grid_shape[1]:
                vy *= -1
                new_cy = max(0, min(self.grid_shape[1] - size, new_cy))

            updated_obstacles.append(("circle", (new_cx, new_cy), size, (vx, vy)))

            # Update occupied positions for the circle
            for x in range(self.grid_shape[0]):
                for y in range(self.grid_shape[1]):
                    if np.sqrt((new_cx - x) ** 2 + (new_cy - y) ** 2) <= size:
                        obs_position.append((x, y))

        elif shape_type == "rectangle":
            x_start, y_start = position
            width, height = size
            vx, vy = velocity
            new_x_start = x_start + vx
            new_y_start = y_start + vy

            # Bounce if hitting boundary
            if new_x_start < 0 or new_x_start + width > self.grid_shape[0]:
                vx *= -1
                new_x_start = max(0, min(self.grid_shape[0] - width, new_x_start))
            if new_y_start < 0 or new_y_start + height > self.grid_shape[1]:
                vy *= -1
                new_y_start = max(0, min(self.grid_shape[1] - height, new_y_start))

            updated_obstacles.append(("rectangle", (new_x_start, new_y_start), size, (vx, vy)))

            # Update occupied positions for the rectangle
            for x in range(new_x_start, new_x_start + width):
                for y in range(new_y_start, new_y_start + height):
                    obs_position.append((x, y))
    self.dynamic_obstacles = updated_obstacles  # Update class attribute
    self.dynamic_obs_position = obs_position
    self.obstacle_list = self.generate_circle_obstacles() + self.generate_rectangle_obstacles() + list(obs_position)


  def create_grid_map(self):
      '''Create a grid map with obstacles marked'''
      grid_map = np.zeros(self.grid_shape, dtype=int)  # 0: Free space, 1: Obstacle
      for (ox, oy) in self.obstacle_list:
          grid_map[ox, oy] = 1  # Mark as an obstacle
      return grid_map

  def display_map(self):
      '''Display the grid map with circular and rectangular obstacles'''
      fig, ax = plt.subplots()
      ax.set_xlim(0, self.grid_shape[1])
      ax.set_ylim(0, self.grid_shape[0])
      ax.set_xticks(np.arange(0, self.grid_shape[1], 1))
      ax.set_yticks(np.arange(0, self.grid_shape[0], 1))
      ax.grid(True)

      # Plot obstacles
      for (ox, oy) in self.obstacle_list:
          ax.add_patch(plt.Rectangle((oy, ox), 1, 1, color="red"))  # Obstacle cells in red

      # Plot the goal point
      ax.add_patch(plt.Rectangle((self.goal[1], self.goal[0]), 1, 1, color="blue"))  # Goal cell in blue

      # Draw circles to represent circular obstacles
      for (center, radius) in self.circle_obstacles:
          circle = plt.Circle((center[1] + 0.5, center[0] + 0.5), radius, color="green", fill=False, linestyle="--")
          ax.add_patch(circle)

      # Draw rectangles to represent rectangular obstacles
      for (x_start, y_start), (width, height) in self.rectangle_obstacles:
          rect = plt.Rectangle((y_start, x_start), height, width, color="orange", fill=False, linestyle="--")
          ax.add_patch(rect)

      # Draw dynamic obstacles (moving ones)
      for (dx, dy), _ in self.dynamic_obstacles:
          ax.add_patch(plt.Rectangle((dy, dx), 1, 1, color="purple", fill=True))

      plt.title("Grid Map")
      plt.gca().invert_yaxis()  # Invert Y-axis to match grid visualization
      plt.show()


# # Example Usage
# goal = (8, 8)
# grid_shape = (10, 10)

# # Create map
# map_d = Map_D(goal=goal, grid_shape=grid_shape)

# # Display the map
# map_d.display_map()
