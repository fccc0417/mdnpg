"""
Reference: https://github.com/xylee95/MD-PGT
Paper: "MDPGT: Momentum-based Decentralized Policy Gradient Tracking"
"""""
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from rl_utils import *
from GridWorld.tools.tool import *
from momentum_pg import MomentumPG
from GridWorld.envs.gridworld import GridWorldEnv
import torch
import copy
import os


seeds = [0, 1, 2, 3, 4]
seed = 5
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cpu")


def set_args(num_agents=1, beta=0.2, topology='dense'):
    parser = argparse.ArgumentParser(description='MDPGT')
    parser.add_argument('--num_agents', type=int, default=num_agents, help='number of agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
    parser.add_argument('--critic_lr', type=float, default=1e-2, help='value learning rate')
    parser.add_argument('--grad_lr', type=float, default=1e-3, help='policy learning rate')
    parser.add_argument('--lmbda', type=float, default=0.95, help='lambda for GAE')
    parser.add_argument('--max_eps_len', type=int, default=100, help='number of steps per episode')
    parser.add_argument('--num_episodes', type=int, default=10000, help='Number training episodes')
    parser.add_argument('--beta', type=float, default=beta, help='beta for momentum-based VR')
    parser.add_argument('--min_isw', type=float, default=0.0, help='minimum value of importance sampling')
    parser.add_argument('--topology', type=str, default=topology, choices=('dense', 'ring', 'bipartite'))
    parser.add_argument('--init_minibatch_size', type=int, default=10, help='number of trajectories for batch gradient in initialization')
    args = parser.parse_args()
    return args


def run(args):
    fpath2 = os.path.join('records', 'mdpgt_logs', str(num_agents) + 'D', 'beta=' + str(args.beta), topology)
    if not os.path.isdir(fpath2):
        os.makedirs(fpath2)
    else:
        del_file(fpath2)
    writer = SummaryWriter(fpath2)

    agents = []
    envs = []
    for i in range(args.num_agents):
        env = GridWorldEnv(seed=seeds[i])
        envs.append(env)

        agents.append(
            MomentumPG(state_space=env.observation_space, action_space=env.action_space, lmbda=args.lmbda,
                       critic_lr=args.critic_lr, gamma=args.gamma,
                       device=device, min_isw=args.min_isw, beta=args.beta))  # .to(device))

    print('observation Space:', env.observation_space)
    print('Action Space:', env.action_space)

    # Generate weight matrix.
    pi = load_pi(num_agents=args.num_agents, topology=args.topology)

    old_policies = []
    for agent in agents:
        old_policy = copy.deepcopy(agent.actor)
        old_policies.append(old_policy)

    # Initialization.
    prev_v_list, y_list = initialization_gt(sample_envs=envs, agents=agents, pi=pi, lr=args.grad_lr,
                                            minibatch_size=args.init_minibatch_size, max_eps_len=args.max_eps_len)

    return_list = []
    error_list = []
    for i in range(10):
        with tqdm(total=int(args.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episodes / 10)):
                # Copy old policies.
                phi_list = copy.deepcopy(old_policies)
                old_policies = []
                for agent in agents:
                    old_policy = copy.deepcopy(agent.actor)
                    old_policies.append(old_policy)

                # Consensus error.
                errors = 0
                para_list = []
                for agent in agents:
                    params = torch.nn.utils.convert_parameters.parameters_to_vector(agent.actor.parameters()).detach()
                    para_list.append(params)
                for k in range(len(para_list)):
                    for t in range(len(para_list)):
                        errors += torch.norm(para_list[k] - para_list[t])
                errors = errors / (num_agents ** 2)
                error_list.append(errors.numpy())

                episode_returns = 0
                v_list = []

                for idx, (agent, env) in enumerate(zip(agents, envs)):
                    # Sample an episode.
                    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                    state = env.reset()
                    for t in range(args.max_eps_len):
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

                    # Generate gradient estimator based on momentum-based variance reduction.
                    advantage = agent.update_value(transition_dict)
                    single_traj_v = agent.compute_v(transition_dict, advantage, prev_v_list[idx], phi_list[idx], args.beta)
                    v_list.append(single_traj_v)

                return_list.append(episode_returns)

                # Gradient tracking.
                y_list = take_grad_consensus(y_list, pi)
                next_y_list = []
                for idx, agent in enumerate(agents):
                    y_new = update_y(y_list[idx], v_list[idx], prev_v_list[idx])
                    next_y_list.append(y_new)

                # Take consensus for gradients and parameters.
                consensus_next_y_list = take_grad_consensus(next_y_list, pi)
                agents = take_param_consensus(agents, pi)

                # Update parameters.
                for agent, grad in zip(agents, consensus_next_y_list):
                    update_param(agent, grad, lr=args.grad_lr)

                prev_v_list = copy.deepcopy(v_list)
                y_list = copy.deepcopy(next_y_list)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (args.num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:]),
                                      'error':  '%.3f' % np.mean(error_list[-10:]),
                                      })
                    writer.add_scalar("rewards", np.mean(return_list[-10:]), args.num_episodes / 10 * i + i_episode + 1)

                pbar.update(1)

    mv_return_list = moving_average(return_list, 9)
    return return_list, mv_return_list, error_list, agents


if __name__ == '__main__':
    env_name = 'GridWorld'
    topologies = ['ring']  # 'dense', 'bipartite'
    num_agents = 5
    betas = [0.2]
    labels = ['beta=0.2']

    for beta, label, topology in zip(betas, labels, topologies):
        args = set_args(num_agents=num_agents, beta=beta, topology=topology)
        fpath = os.path.join('mdpgt_results', env_name, str(num_agents) + '_agents',
                             label + '_' + topology)
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
        print(f"beta={beta}")

        return_list, mv_return_list, error_list, agents = run(args=args)

        np.save(os.path.join(fpath, 'return.npy'), return_list)
        np.save(os.path.join(fpath, 'avg_return.npy'), mv_return_list)
        np.save(os.path.join(fpath, 'error.npy'), error_list)

        # Save the trained models.
        for idx, agent in enumerate(agents):
            torch.save(agent, os.path.join(fpath, 'agent' + str(idx) + '.pth'))


