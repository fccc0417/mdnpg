import gym
import numpy as np
from gym import spaces


class GridWorldEnv(gym.Env):
    """Class of GridWorld.
    In a grid map, we specify  positions of goal and of obstacles.
    """
    def __init__(self, grid_map_path, seed=0, agent_pos=np.array([0, 0])):
        self.seed = seed
        np.random.seed(self.seed)
        self.init_pos = agent_pos
        self.pos_dim = 2
        self.actions = [0, 1, 2, 3]
        self.action_space = spaces.Discrete(4)
        # self.action_dict = {0: "up", 1: "down", 2: "left", 3: "right"}
        self.action_dict = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}

        self.grid_map = np.load(grid_map_path)
        self.agent_pos = self.init_pos
        self.goal_pos = np.squeeze(np.asarray(np.where(self.grid_map == 'G')))
        self.obstacle_pos = np.asarray(np.where(self.grid_map == 'O')).T
        self.grid_shape = self.grid_map.shape
        obs = self.get_observation()
        self.obs_dim = len(obs)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32)
        print('\n')
        print(self.grid_map)

    def reset(self):
        """Reset the GridWorld."""
        self.agent_pos = self.init_pos
        obs = self.get_observation()
        return obs

    def step(self, action):
        """Take an action.
        Return: observation (state), reward, done (whether termination), info (other information)
        """
        pos = self.agent_pos + self.action_dict[action]
        if 0 <= pos[0] < self.grid_shape[0] and 0 <= pos[1] < self.grid_shape[1]:
            self.agent_pos = pos

        obs = self.get_observation()
        reward = self.get_reward()
        done = self.get_done()
        info = {}
        return obs, reward, done, info

    def get_observation(self):
        """Obtain the observation of the GridWorld."""
        obs = self.agent_pos
        return obs

    def get_reward(self):
        """Obtain the reward from the GridWorld."""
        reward = -0.1 * np.sqrt(
            (self.agent_pos[0] - self.goal_pos[0]) ** 2 + (self.agent_pos[1] - self.goal_pos[1]) ** 2)
        pos = tuple(self.agent_pos)
        if self.grid_map[pos] == 'O':
            reward -= 10
        elif self.grid_map[pos] == 'G':
            reward += 10
        return reward

    def get_done(self):
        """Determine whether the trajectory is terminated."""
        done = False
        pos = tuple(self.agent_pos)
        if self.grid_map[pos] == 'G' or self.grid_map[pos] == 'O':
            done = True
        return done


'''
Usage:

args.random_loc = False

map_path_0 = "../envs/grid_maps/map_0.npy"
map_path_1 = "../envs/grid_maps/map_1.npy"
map_path_2 = "../envs/grid_maps/map_2.npy"
map_path_3 = "../envs/grid_maps/map_3.npy"
map_path_4 = "../envs/grid_maps/map_4.npy"
map_paths = [map_path_0, map_path_1, map_path_2, map_path_3, map_path_4]

env = GridWorldEnv(grid_map_path=map_paths[i])
'''

