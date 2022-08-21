import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from rl_utils_vp import *
from tools.tool import *
from value_propagation_multi_step import ValuePropagation
from particle_envs import make_particleworld
import torch
import os

seed = 4
torch.manual_seed(seed)
# device = torch.device("cpu")  # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_args(num_agents=1, topology='dense'):
    parser = argparse.ArgumentParser(description='Multi-agent example')
    parser.add_argument('--num_agents', type=int, default=num_agents, help='Number of agents')
    parser.add_argument('--num_landmarks', type=int, default=num_agents, help='Number of landmarks')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
    parser.add_argument('--eta', type=float, default=0.1, help='eta, trade-off parameter')
    parser.add_argument('--value_lr', type=float, default=3e-4, help='update lr')
    parser.add_argument('--policy_lr', type=float, default=3e-4, help='update lr')
    parser.add_argument('--dual_lr', type=float, default=3e-4, help='update lr')
    parser.add_argument('--lmbda', type=float, default=0.01, help='lambda')
    parser.add_argument('--T_dual', type=float, default=4, help='T_dual')
    parser.add_argument('--topology', type=str, default=topology, choices=('dense', 'ring', 'bipartite'))
    parser.add_argument('--max_eps_len', type=int, default=20+20, help='Number of steps per episode')
    parser.add_argument('--n_steps', type=int, default=20, help='Number of steps per episode')
    parser.add_argument('--num_episodes', type=int, default=20000, help='Number training episodes')
    args = parser.parse_args()
    return args


def run(args, env_name):
    # timestr = str(time()).replace('.', 'p')
    fpath2 = os.path.join('records', 'vaule_propagation_logs', str(num_agents) + 'D', topology)
    if not os.path.isdir(fpath2):
        os.makedirs(fpath2)
    else:
        del_file(fpath2)
    writer = SummaryWriter(fpath2)

    sample_env = make_particleworld.make_env(env_name, num_agents=args.num_agents, num_landmarks=args.num_landmarks)
    sample_env.discrete_action_input = True  # set action space to take in discrete numbers 0,1,2,3
    sample_env.seed(seed)

    print('Observation Space:', sample_env.observation_space)
    print('Action Space:', sample_env.action_space)
    print('Number of agents:', sample_env.n)
    sample_obs = sample_env.reset()
    sample_obs = np.concatenate(sample_obs).ravel()  # .tolist() #多个智能体需要合并再变成一维

    # load connectivity matrix
    pi = load_pi(num_agents=args.num_agents, topology=args.topology)

    agents = []
    for _ in range(args.num_agents):
        agents.append(ValuePropagation(num_agents=args.num_agents, state_dim=len(sample_obs), action_dim=5, pi=pi,
                                       eta=args.eta, lmbda=args.lmbda, gamma=args.gamma, T_dual=args.T_dual,
                                       value_lr=args.value_lr, policy_lr=args.policy_lr, dual_lr=args.dual_lr,
                                       n_steps=args.n_steps, max_eps_len=args.max_eps_len))

    env = make_particleworld.make_env(env_name, num_agents=args.num_agents, num_landmarks=args.num_landmarks)
    env.discrete_action_input = True
    env.seed(seed)

    return_list = []
    for i in range(10):
        with tqdm(total=int(args.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episodes / 10)):
                for dual_epoch in range(args.T_dual):
                    state = env.reset()
                    state = np.concatenate(state).ravel()
                    state_list = []
                    actions_list = []
                    next_state_list = []
                    reward_lists = [[] for _ in range(args.num_agents)]
                    done_list = []
                    for t in range(args.max_eps_len):
                        actions = []
                        for actor in agents:
                            actions.append(actor.take_action(state))
                        next_state, rewards, dones, _ = env.step(actions)
                        next_state = np.concatenate(next_state).ravel()
                        done = all(item == True for item in dones)
                        state_list.append(state)
                        actions_list.append(actions)
                        next_state_list.append(next_state)
                        for j in range(args.num_agents):
                            reward_lists[j].append(rewards[j])
                        done_list.append(done)
                        state = next_state
                        reset = t == args.max_eps_len - 1
                        if done or reset:
                            break
                    for idx, agent in enumerate(agents):
                        transition_dict = {'states': state_list, 'actions': actions_list, 'next_states': next_state_list,
                                           'rewards': reward_lists[idx], 'dones': done_list}
                        agent.calc_dual_grad(idx, transition_dict)
                    agents = take_dual_param_consensus(agents, pi)
                    for idx, agent in enumerate(agents):
                        agent.update_dual_problem()

                episode_returns = 0
                state = env.reset()
                state = np.concatenate(state).ravel()
                state_list = []
                actions_list = []
                next_state_list = []
                reward_lists = [[] for _ in range(args.num_agents)]
                done_list = []
                for t in range(args.max_eps_len):
                    actions = []
                    for actor in agents:
                        actions.append(actor.take_action(state))
                    next_state, rewards, dones, _ = env.step(actions)
                    next_state = np.concatenate(next_state).ravel()
                    done = all(item == True for item in dones)
                    state_list.append(state)
                    actions_list.append(actions)
                    next_state_list.append(next_state)
                    for j in range(args.num_agents):
                        reward_lists[j].append(rewards[j])
                    done_list.append(done)
                    state = next_state
                    if t < args.max_eps_len - args.n_steps:
                        episode_returns += np.sum(rewards)
                    reset = t == args.max_eps_len - 1
                    if done or reset:
                        break

                for idx, agent in enumerate(agents):
                    transition_dict = {'states': state_list, 'actions': actions_list, 'next_states': next_state_list,
                                       'rewards': reward_lists[idx], 'dones': done_list}
                    agent.calc_primal_grad(idx, transition_dict)
                agents = take_value_param_consensus(agents, pi)
                for agent in agents:
                    agent.update_primal_problem()

                return_list.append(episode_returns)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (args.num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                    writer.add_scalar("rewards", np.mean(return_list[-10:]), args.num_episodes / 10 * i + i_episode + 1)

                pbar.update(1)
    mv_return_list = moving_average(return_list, 9)
    return return_list, mv_return_list, agents


if __name__ == '__main__':
    env_name = 'CooperativeNavigation'
    num_agents = 5
    topologies = ['ring']  #'bipartite', 'ring'
    labels = ['ring'] 
    return_lists = []
    mv_return_lists = []


    for label, topology in zip(labels, topologies):
        args = set_args(num_agents=num_agents, topology=topology)
        fpath = os.path.join('vaule_propagation_results', env_name, str(num_agents) + 'D',
                              topology)
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
        print(f"topology={topology}")

        return_list, mv_return_list, agents = run(args=args, env_name=env_name)
        np.save(os.path.join(fpath, 'return.npy'), return_list)
        np.save(os.path.join(fpath, 'avg_return.npy'), mv_return_list)
        return_lists.append(return_list)
        mv_return_lists.append(mv_return_list)

        for idx, agent in enumerate(agents):
            torch.save(agent, os.path.join(fpath, 'agent' + str(idx) + '.pth'))

