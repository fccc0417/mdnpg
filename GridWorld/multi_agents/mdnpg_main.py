import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from rl_utils import *
from tools.tool import *
from momentum_npg import MomentumNPG
import torch
import copy
import os
from envs.gridworld import GridWorldEnv
# from envs.gridworld_4_test import GridWorldEnv
from envs.init_agent_pos_4_all_envs import *

map_path_0 = "../envs/grid_maps/map_0.npy"
map_path_1 = "../envs/grid_maps/map_1.npy"
map_path_2 = "../envs/grid_maps/map_2.npy"
map_path_3 = "../envs/grid_maps/map_3.npy"
map_path_4 = "../envs/grid_maps/map_4.npy"
map_paths = [map_path_0, map_path_1, map_path_2, map_path_3, map_path_4]
seeds = [0, 1, 2, 3, 4]
seed = 0
device = torch.device("cpu")
np.random.seed(seed)
torch.manual_seed(seed)

def set_args(num_agents=1, beta=0.2, topology='dense'):
    parser = argparse.ArgumentParser(description='Multi-agent example')
    parser.add_argument('--num_agents', type=int, default=num_agents, help='Number of agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
    parser.add_argument('--critic_lr', type=float, default=1e-2, help='critic_lr')
    parser.add_argument('--grad_lr', type=float, default=5e-4, help='used to update params')
    parser.add_argument('--lmbda', type=float, default=0.95, help='lambda')
    parser.add_argument('--kl_constraint', type=float, default=1e-5, help='kl_constraint') # 0.0005  5e-5
    parser.add_argument('--alpha', type=float, default=0.01, help='alpha')
    parser.add_argument('--max_eps_len', type=int, default=100, help='Number of steps per episode')
    parser.add_argument('--num_episodes', type=int, default=10000, help='Number training episodes')
    parser.add_argument('--beta', type=float, default=beta, help='Beta term for surrogate gradient')
    parser.add_argument('--min_isw', type=float, default=0.0, help='Minimum value to set ISW')
    parser.add_argument('--topology', type=str, default=topology, choices=('dense', 'ring', 'bipartite'))
    parser.add_argument('--init_minibatch_size', type=int, default=32, help='Number of trajectory for warm startup')
    parser.add_argument('--random_loc', type=bool, default=True, help='Each episode use random initial location')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()
    return args

def run(args):
    # timestr = str(time()).replace('.', 'p')

    fpath2 = os.path.join('records', 'md_npg_gt_logs', str(num_agents) + 'D', 'beta=' + str(args.beta), topology)
    if not os.path.isdir(fpath2):
        os.makedirs(fpath2)
    else:
        del_file(fpath2)
    writer = SummaryWriter(fpath2)

    agents = []
    agent_pos = np.random.randint(0, 10, 2)
    envs = []
    print(agent_pos)
    for i in range(args.num_agents):
        env = GridWorldEnv(seed=seeds[i], agent_pos=agent_pos)
        # env = GridWorldEnv(grid_map_path=map_paths[i])
        envs.append(env)
        agents.append(MomentumNPG(state_space=env.observation_space, action_space=env.action_space, lmbda=args.lmbda,
                                  kl_constraint=args.kl_constraint, alpha=args.alpha,
                                  critic_lr=args.critic_lr, gamma=args.gamma,
                                  device=device, min_isw=args.min_isw, beta=args.beta))

    print('observation Space:', env.observation_space)
    print('Action Space:', env.action_space)

    # create weight matrix
    pi = load_pi(num_agents=args.num_agents, topology=args.topology)

    old_policies = []
    for agent in agents:
        old_policy = copy.deepcopy(agent.actor)
        old_policies.append(old_policy)

    prev_u_list, v_k_list = initialization_gt(sample_envs=envs, agents=agents, pi=pi, lr=args.grad_lr, minibatch_size=args.init_minibatch_size,
                                           max_eps_len=args.max_eps_len)

    return_list = []
    error_list = []
    for i in range(10):
        with tqdm(total=int(args.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episodes / 10)):
                phi_list = copy.deepcopy(old_policies)
                # old_agent is now updated agent
                old_policies = []
                for agent in agents:
                    old_policy = copy.deepcopy(agent.actor)
                    old_policies.append(old_policy)

                # consensus error
                errors = 0
                para_list = []
                for agent in agents:
                    params = torch.nn.utils.convert_parameters.parameters_to_vector(agent.actor.parameters()).detach()
                    para_list.append(params)
                for k in range(len(para_list)):
                    for t in range(len(para_list)):
                        errors += torch.norm(para_list[k]-para_list[t])
                errors = errors / (num_agents**2)
                error_list.append(errors.numpy())

                episode_returns = 0
                u_k_list = []
                states_lists = []
                transition_dicts = []
                advantage_list = []
                if args.random_loc:
                    agent_pos = agent_pos_reset_4_envs(envs)
                for idx, (agent, env) in enumerate(zip(agents, envs)):
                    states_list = []
                    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                    state = env.reset()
                    for t in range(args.max_eps_len):
                        states_list.append(state)
                        action = agent.take_action(state)
                        next_state, reward, done, _ = env.step(action)
                        transition_dict['states'].append(state)
                        transition_dict['actions'].append(action)
                        transition_dict['next_states'].append(next_state)
                        transition_dict['rewards'].append(reward)
                        transition_dict['dones'].append(done)
                        state = next_state
                        episode_returns += np.sum(reward)
                        reset = t == args.max_eps_len - 1
                        if done or reset:
                            break

                    advantage = agent.update_value(transition_dict)
                    single_traj_u = agent.compute_u_k(transition_dict, advantage, prev_u_list[idx], phi_list[idx], args.beta)
                    u_k_list.append(single_traj_u)
                    states_lists.append(states_list)
                    transition_dicts.append(transition_dict)
                    advantage_list.append(advantage)

                return_list.append(episode_returns)

                # tracking
                v_k_list = take_grad_consensus(v_k_list, pi, agents)
                next_v_k_list = []
                for idx, agent in enumerate(agents):
                    v_k_new = update_v(v_k_list[idx], u_k_list[idx], prev_u_list[idx])
                    next_v_k_list.append(v_k_new)

                update_grad_list = []
                for s_list, v_k, agent, transition_dict, advantage in zip(states_lists, next_v_k_list, agents, transition_dicts, advantage_list):
                    direction_grad = agent.compute_precondition_with_v(s_list, v_k, transition_dict, advantage)
                    update_grad_list.append(direction_grad)

                consensus_grad_list = take_grad_consensus(update_grad_list, pi, agents)
                agents = take_param_consensus(agents, pi)

                for agent, grad in zip(agents, consensus_grad_list):
                    update_param(agent, grad, lr=1)  #lr=args.grad_lr

                prev_u_list = copy.deepcopy(u_k_list)
                v_k_list = copy.deepcopy(next_v_k_list)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (args.num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:]),
                                      'error':  '%.3f' % np.mean(error_list[-10:]),
                                      })
                    writer.add_scalar("rewards", np.mean(return_list[-10:]), args.num_episodes / 10 * i + i_episode + 1)
                pbar.update(1)

    mv_return_list = moving_average(return_list, 9)
    return return_list, mv_return_list, agents, error_list




if __name__ == '__main__':
    env_name = 'GridWorld'
    topologies = ['ring']  # 'dense'
    num_agents = 5
    betas = [0.2]
    labels = ['beta=0.2']
    return_lists = []
    mv_return_lists = []
    error_lists = []

    for beta, label, topology in zip(betas, labels, topologies):
        args = set_args(num_agents=num_agents, beta=beta, topology=topology)
        fpath = os.path.join('md_npg_gt_results', env_name, str(num_agents) + 'D',
                             'beta=' + str(beta) + '_' + topology)  # + '_' + timestr
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
        print(f"beta={beta}")

        return_list, mv_return_list, agents, error_list = run(args=args)
        np.save(os.path.join(fpath, 'return.npy'), return_list)
        np.save(os.path.join(fpath, 'avg_return.npy'), mv_return_list)
        np.save(os.path.join(fpath, 'error_list.npy'), error_list)
        return_lists.append(return_list)
        mv_return_lists.append(mv_return_list)
        error_lists.append(error_list)

        for idx, agent in enumerate(agents):
            torch.save(agent, os.path.join(fpath, 'agent' + str(idx) + '.pth'))


    plt.figure()
    for return_list, label in zip(return_lists, labels):
        plt.plot(return_list, label=label)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.legend()
    plt.title('{}-agent Momentum NPG on {}'.format(num_agents, env_name))
    plt.savefig(os.path.join(fpath, 'return.jpg'))
    plt.show()

    plt.figure()
    for return_list, label in zip(mv_return_lists, labels):
        plt.plot(return_list, label=label)
    plt.xlabel('Episodes')
    plt.ylabel('Moving_average Returns')
    plt.title('{}-agent Momentum NPG on {}'.format(num_agents, env_name))
    plt.legend()
    plt.savefig(os.path.join(fpath, 'avg_return.jpg'))
    plt.show()

    plt.figure()
    for return_list, label in zip(error_lists, labels):
        plt.plot(error_list, label=label)
    plt.xlabel('Episodes')
    plt.ylabel('Consensus error')
    plt.legend()
    plt.title('{}-agent Momentum NPG on {}'.format(num_agents, env_name))
    plt.savefig(os.path.join(fpath, 'error.jpg'))
    plt.show()
