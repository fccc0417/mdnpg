from momentum_trpo_discrete import *
from envs.gridworld_4_test import GridWorldEnv

map_path_0 = "../envs/grid_maps/map_0.npy"
map_path_1 = "../envs/grid_maps/map_1.npy"
map_path_2 = "../envs/grid_maps/map_2.npy"
map_path_3 = "../envs/grid_maps/map_3.npy"
map_path_4 = "../envs/grid_maps/map_4.npy"
map_paths = [map_path_0, map_path_1, map_path_2, map_path_3, map_path_4]
num_agents = 5
agent_pos = np.array([0, 0])

seeds = [0, 1]#0, 1, 2, 3,
max_eps_len = 200
episodes = 11
algos = ['npg']
reward_lists = []
seed_value = 8
np.random.seed(seed_value)
torch.manual_seed(seed_value)
for algo in algos:
    agents = []
    for seed in seeds:
        agent = torch.load('../single_agent/agents/'+algo+'_agent_'+str(seed)+'.pth')
        agents.append(agent)

    envs = []
    for path in map_paths:
        print(seed)
        env = GridWorldEnv(grid_map_path=path)
        envs.append(env)

    reward_list=[]
    for agent in agents:
        reward_4_agent = []
        for env in envs:
            returns = 0
            for i in range(episodes):
                state = env.reset()
                done = False
                for t in range(max_eps_len):
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    returns += reward
                    if done:
                        break
            reward_4_agent.append(returns/episodes)
        reward_list.append(reward_4_agent)
    reward_lists.append(reward_list)

for r_list, algo in zip(reward_lists, algos):
    print(algo)
    for reward in r_list:
        print(np.round(reward, 2))
        print(np.round(np.sum(reward), 2))



# map_0 = \
#     [['T', 'T', 'T', 'T', 'T', 'T' ,'T' ,'T' ,'T' ,'T'],
#     ['T', 'T', 'T', 'T', 'O', 'T', 'T', 'T', 'T', 'T'],
#     ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#     ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T' ,'T' ,'T'],
#     ['T', 'T', 'O' ,'T' ,'T' ,'T' ,'T', 'T' ,'T', 'T'],
#     ['T', 'T', 'T', 'T', 'T', 'O', 'T', 'T', 'T' ,'O'],
#     ['T', 'T', 'O', 'T', 'T', 'T', 'T', 'T', 'T' ,'T'],
#     ['T', 'T' ,'T' ,'T', 'T', 'T' ,'T' ,'T', 'T', 'T'],
#     ['T', 'T', 'T', 'T' ,'T' ,'T' ,'T', 'T', 'T', 'T'],
#     ['T' ,'T', 'T', 'G', 'T', 'T' ,'T', 'T', 'T' ,'T']]
# map_0 = np.array(map_0)
#
# map_1 = \
# [['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'O', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'G'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'O', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'O', 'O'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'O', 'T']]
# map_1 = np.array(map_1)
#
# map_2 = \
# [['T', 'T', 'T', 'T', 'T', 'O', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'O', 'T', 'T', 'T', 'T', 'T', 'O'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'G', 'T', 'T', 'T', 'O', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'O', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T']]
# map_2 = np.array(map_2)
#
# map_3 = \
# [['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'O', 'T', 'O', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'O', 'T'],
#  ['T', 'O', 'T', 'T', 'T', 'T', 'G', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'O', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T']]
# map_3 = np.array(map_3)
#
# map_4 = \
# [['T', 'T', 'T', 'T', 'O', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'O', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'O', 'T', 'O', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'O', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'],
#  ['T', 'T', 'T', 'T', 'G', 'T', 'T', 'T', 'T', 'T']]
# map_4 = np.array(map_4)


# ########################################################

# [['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'O' 'T' 'O' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'O' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'O' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'O' 'T']
#  ['G' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']]
#
#
# [['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'O']
#  ['T' 'O' 'T' 'T' 'O' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'O' 'T']
#  ['O' 'G' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']]
#
#
# [['T' 'T' 'T' 'T' 'T' 'T' 'O' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['G' 'T' 'T' 'T' 'O' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'O' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'O' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'O' 'T' 'T' 'T' 'T' 'T' 'T']]
#
#
# [['T' 'T' 'T' 'T' 'T' 'T' 'T' 'O' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['O' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'O' 'T' 'T' 'O']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'G' 'T' 'T' 'O']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']]
#
#
# [['T' 'T' 'T' 'T' 'O' 'T' 'T' 'O' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'O' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'O' 'O' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T' 'T']
#  ['T' 'T' 'T' 'T' 'T' 'G' 'T' 'T' 'T' 'T']