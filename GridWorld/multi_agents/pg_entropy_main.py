"""
Paper: "A Decentralized Policy Gradient Approach to Multi-Task Reinforcement Learning"
"""
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from rl_utils import *
from GridWorld.tools.tool import *
from pg_entropy import PGwithEntropy
from GridWorld.envs.gridworld import GridWorldEnv
from GridWorld.envs.init_agent_pos_4_all_envs import *
import torch
import os


seeds = [6, 7, 8, 9, 10]
seed = 11
device = torch.device("cpu")
np.random.seed(seed)
torch.manual_seed(seed)


def set_args(num_agents=1, topology='dense'):
    parser = argparse.ArgumentParser(description='Policy gradient with entropy')
    parser.add_argument('--entropy_para', type=float, default=0.1, help='entropy parameter')
    parser.add_argument('--num_agents', type=int, default=num_agents, help='number of agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--critic_lr', type=float, default=1e-2, help='value learning rate')
    parser.add_argument('--grad_lr', type=float, default=5e-4, help='policy learning rate')
    parser.add_argument('--lmbda', type=float, default=0.95, help='lambda for GAE')
    parser.add_argument('--max_eps_len', type=int, default=200, help='number of steps per episode')
    parser.add_argument('--num_episodes', type=int, default=10000, help='number training episodes')
    parser.add_argument('--topology', type=str, default=topology, choices=('dense', 'ring', 'bipartite'))
    parser.add_argument('--random_loc', type=bool, default=True, help='whether each episode uses a random initial location for all agents')
    args = parser.parse_args()
    return args


def run(args):
    # timestr = str(time()).replace('.', 'p')
    fpath2 = os.path.join('records', 'pg_en_logs', str(num_agents) + 'D', topology)
    if not os.path.isdir(fpath2):
        os.makedirs(fpath2)
    else:
        del_file(fpath2)
    writer = SummaryWriter(fpath2)

    agents = []
    envs = []
    agent_pos = np.random.randint(0, 10, 2)
    print(agent_pos)
    for i in range(args.num_agents):
        env = GridWorldEnv(seed=seeds[i], agent_pos=agent_pos)
        envs.append(env)
        agents.append(
            PGwithEntropy(state_space=env.observation_space, action_space=env.action_space, lmbda=args.lmbda,
                       critic_lr=args.critic_lr, gamma=args.gamma, entropy_para=args.entropy_para, device=device))

    print('observation Space:', env.observation_space)
    print('Action Space:', env.action_space)

    # Generate weight matrix.
    pi = load_pi(num_agents=args.num_agents, topology=args.topology)

    return_list = []
    error_list = []
    for i in range(10):
        with tqdm(total=int(args.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episodes / 10)):
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
                grad_list = []

                # Whether randomly generate the initial location of agents.
                if args.random_loc:
                    agent_pos = agent_pos_reset_4_envs(envs)

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

                    # Generate policy gradient estimator.
                    advantage = agent.update_value(transition_dict)
                    single_traj_grad = agent.compute_grads(transition_dict, advantage)
                    grad_list.append(single_traj_grad)

                return_list.append(episode_returns)

                # Take paramters consensus.
                agents = take_param_consensus(agents, pi)

                # Update parameters.
                for agent, grad in zip(agents, grad_list):
                    update_param(agent, grad, lr=args.grad_lr)

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
    topologies = ['ring']
    num_agents = 5

    labels = ['ring']

    for label, topology in zip(labels, topologies):
        args = set_args(num_agents=num_agents, topology=topology)
        fpath = os.path.join('pg_en_results', env_name, str(num_agents) + '_agents', topology)
        if not os.path.isdir(fpath):
            os.makedirs(fpath)

        return_list, mv_return_list, agents, error_list = run(args=args)

        np.save(os.path.join(fpath, 'return.npy'), return_list)
        np.save(os.path.join(fpath, 'avg_return.npy'), mv_return_list)
        np.save(os.path.join(fpath, 'error.npy'), error_list)

        # Save the trained models.
        for idx, agent in enumerate(agents):
            torch.save(agent, os.path.join(fpath, 'agent' + str(idx) + '.pth'))

