import numpy as np
import torch
from momentum_npg import MomentumNPG
from momentum_pg_en import MomentumPG
from momentum_pg import MomentumPG
import os
from envs.gridworld_4_test import GridWorldEnv

map_path_0 = "../envs/grid_maps/map_0.npy"
map_path_1 = "../envs/grid_maps/map_1.npy"
map_path_2 = "../envs/grid_maps/map_2.npy"
map_path_3 = "../envs/grid_maps/map_3.npy"
map_path_4 = "../envs/grid_maps/map_4.npy"
map_paths = [map_path_0, map_path_1, map_path_2, map_path_3, map_path_4]

num_agents = 5
max_eps_len = 200
episodes = 100
agent_pos = np.array([0, 0])
seeds = [11, 12, 17, 19, 15]
algos = ['npgnew', 'pgennew', 'pgnew']  #, 'pg_en2', 'pg2'
reward_lists = []
seed_value = 2
np.random.seed(seed_value)
torch.manual_seed(seed_value)

for algo in algos:
    agents = []
    for idx in range(num_agents):
        agent = torch.load(os.path.join('../', 'multi_agents', 'datas', algo, 'agent'+str(idx)+'.pth'))
        agents.append(agent)

    envs = []
    for idx in range(num_agents):
        env = GridWorldEnv(grid_map_path=map_paths[idx])
        # print(env.obstacle_pos)
        # print(env.goal_pos)
        # print(env.goal_pos)
        envs.append(env)

    reward_list=[]
    for idx, (agent, env) in enumerate(zip(agents, envs)):
        returns = 0
        for i in range(episodes):
            rewards = []
            state = env.reset()
            done = False
            for t in range(max_eps_len):
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                rewards.append(reward)
                if done:
                    break
            returns += np.sum(rewards)
        reward_list.append(returns/episodes)
    reward_lists.append(reward_list)

for r_list, algo in zip(reward_lists, algos):
    print(algo)
    print(np.round(r_list, 2))
    print(np.round(np.sum(r_list), 2))
