import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from rl_utils import *
from tools.tool import *
from momentum_npg import MomentumNPG
from particle_envs import make_particleworld
import torch
import copy
import os

seed = 0
torch.manual_seed(seed)
device = torch.device("cpu") 


def set_args(num_agents=1, beta=0.2, topology='dense'):
    parser = argparse.ArgumentParser(description='Multi-agent example')
    parser.add_argument('--num_agents', type=int, default=num_agents, help='Number of agents')
    parser.add_argument('--num_landmarks', type=int, default=num_agents, help='Number of landmarks')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='critic_lr')
    parser.add_argument('--grad_lr', type=float, default=5e-3, help='update lr')
    parser.add_argument('--lmbda', type=float, default=0.95, help='lambda')
    parser.add_argument('--kl_constraint', type=float, default=0.0005, help='kl_constraint') # 0.0005
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    parser.add_argument('--max_eps_len', type=int, default=20, help='Number of steps per episode')
    parser.add_argument('--num_episodes', type=int, default=20000, help='Number training episodes')
    parser.add_argument('--beta', type=float, default=beta, help='Beta term for surrogate gradient')
    parser.add_argument('--min_isw', type=float, default=0.0, help='Minimum value to set ISW')
    parser.add_argument('--topology', type=str, default=topology, choices=('dense', 'ring', 'bipartite'))
    parser.add_argument('--init_lr', type=float, default=3e-10, help='used to update param in initialization')
    parser.add_argument('--minibatch_size', type=int, default=1, help='Number of trajectory for warm startup')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()
    return args


def run(args, env_name):
    # timestr = str(time()).replace('.', 'p')
    fpath2 = os.path.join('records', 'mdnpg_logs', str(num_agents) + 'D', 'beta=' + str(args.beta), topology)
    if not os.path.isdir(fpath2):
        os.makedirs(fpath2)
    else:
        del_file(fpath2)

    writer = SummaryWriter(fpath2)
    sample_envs = []

    sample_env = make_particleworld.make_env(env_name, num_agents=args.num_agents, num_landmarks=args.num_landmarks)
    sample_env.discrete_action_input = True  # set action space to take in discrete numbers 0,1,2,3,4
    sample_env.seed(seed)
    for _ in range(args.num_agents):
        env_copy = copy.deepcopy(sample_env)
        sample_envs.append(env_copy)

    print('Observation Space:', sample_env.observation_space)
    print('Action Space:', sample_env.action_space)
    print('Number of agents:', sample_env.n)
    sample_obs = sample_env.reset()
    sample_obs = np.concatenate(sample_obs).ravel() 

    agents = []
    for i in range(args.num_agents):
        agents.append(MomentumNPG(num_agents=args.num_agents, state_dim=len(sample_obs), action_dim=5, lmbda=args.lmbda,
                                  kl_constraint=args.kl_constraint, alpha=args.alpha,
                                  critic_lr=args.critic_lr, gamma=args.gamma,
                                  device=device, min_isw=args.min_isw, beta=args.beta))  

    # load connectivity matrix
    pi = load_pi(num_agents=args.num_agents, topology=args.topology)

    old_policies = []
    for agent in agents:
        old_policy = copy.deepcopy(agent.actors)
        old_policies.append(old_policy)

    prev_v_lists, y_lists = initialization_gt(sample_envs, agents, pi, lr=args.init_lr, minibatch_size=1,
                                           max_eps_len=args.max_eps_len)

    envs = []
    env = make_particleworld.make_env(env_name, num_agents=args.num_agents, num_landmarks=args.num_landmarks)
    env.discrete_action_input = True
    env.seed(seed)
    for _ in range(args.num_agents):
        env_copy = copy.deepcopy(env)
        envs.append(env_copy)

    return_list = []
    error_list = []

    for i in range(10):
        with tqdm(total=int(args.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episodes / 10)):
                phis_list = copy.deepcopy(old_policies)
                # old_agent is now updated agent
                old_policies = []
                for agent in agents:
                    old_policy = copy.deepcopy(agent.actors)
                    old_policies.append(old_policy)

                episode_returns = 0
                v_lists = []
                states_lists = []

                # consensus error
                errors = 0
                param_list = []
                for agent in agents:
                    params = []
                    for actor in agent.actors:
                        param = torch.nn.utils.convert_parameters.parameters_to_vector(actor.parameters()).detach()
                        params.append(param)
                    mparams = torch.cat([p.view(-1) for p in params]).detach()
                    param_list.append(mparams)
                for k in range(len(param_list)):
                    for t in range(len(param_list)):
                        errors += torch.norm(param_list[k]-param_list[t], 2)
                errors = errors / (num_agents**2)
                error_list.append(errors.numpy())

                for idx, (agent, env) in enumerate(zip(agents, envs)):
                    minibatch_v = []
                    states_list = []
                    for b in range(args.minibatch_size):
                        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                        state = env.reset()
                        state = np.concatenate(state).ravel()
                        for t in range(args.max_eps_len):
                            states_list.append(state)
                            actions = agent.take_actions(state)
                            next_state, rewards, dones, _ = env.step(actions)
                            next_state = np.concatenate(next_state).ravel()
                            done = all(item == True for item in dones)
                            transition_dict['states'].append(state)
                            transition_dict['actions'].append(actions)
                            transition_dict['next_states'].append(next_state)
                            transition_dict['rewards'].append(rewards[idx])
                            transition_dict['dones'].append(dones[idx])
                            state = next_state
                            episode_returns += np.sum(rewards[idx])
                            reset = t == args.max_eps_len - 1
                            if done or reset:
                                break

                        advantage = agent.update_value(transition_dict)
                        single_traj_v = agent.compute_v_list(transition_dict, advantage, prev_v_lists[idx], phis_list[idx], args.beta)
                        single_traj_v = torch.stack(single_traj_v, dim=0)
                        minibatch_v.append(single_traj_v)

                    # episode_returns += episode_return
                    minibatch_v = torch.stack(minibatch_v, dim=0)
                    grad_v = torch.mean(minibatch_v, dim=0)
                    v_lists.append(grad_v)
                    states_lists.append(states_list)

                return_list.append(episode_returns/args.minibatch_size)

                # tracking
                y_lists = take_grad_consensus(y_lists, pi)
                next_y_lists = update_y_lists(y_lists, prev_v_lists, v_lists)

                update_grad_lists = []
                for next_y_list, agent, states_list in zip(next_y_lists, agents, states_lists):
                    direction_grad_list = agent.compute_precondition_with_y(states_list, next_y_list)
                    update_grad_lists.append(direction_grad_list)

                consensus_grad_lists = take_grad_consensus(update_grad_lists, pi)
                agents = take_param_consensus(agents, pi)

                for agent, grads in zip(agents, consensus_grad_lists):
                    update_param(agent, grads, lr=args.grad_lr)

                y_lists = copy.deepcopy(next_y_lists)
                prev_v_lists = copy.deepcopy(v_lists)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (args.num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:]),
                                      'error': '%.3f' % np.mean(error_list[-10:])})
                    writer.add_scalar("rewards", np.mean(return_list[-10:]), args.num_episodes / 10 * i + i_episode + 1)

                pbar.update(1)
    mv_return_list = moving_average(return_list, 9)
    return return_list, mv_return_list, error_list, agents


if __name__ == '__main__':
    env_name = 'CooperativeNavigation'
    num_agents = 5
    topologies = ['ring']  #['dense', bipartite, ring]
    betas =  [0.2]
    labels = ['beta=0.2']
    return_lists = []
    mv_return_lists = []
    error_lists = []

    for beta, label, topology in zip(betas, labels, topologies):
        args = set_args(num_agents=num_agents, beta=beta, topology=topology)
        fpath = os.path.join('mdnpg_results', env_name, str(num_agents) + '_agents',
                             label + '_' + topology)  # + '_' + timestr
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
        print(f"beta={beta}")

        return_list, mv_return_list, error_list, agents = run(args=args, env_name=env_name)
        np.save(os.path.join(fpath, 'return.npy'), return_list)
        np.save(os.path.join(fpath, 'avg_return.npy'), mv_return_list)
        np.save(os.path.join(fpath, 'error_list.npy'), error_list)
        return_lists.append(return_list)
        error_lists.append(error_list)
        mv_return_lists.append(mv_return_list)

        for idx, agent in enumerate(agents):
            torch.save(agent, os.path.join(fpath, 'agent' + str(idx) + '.pth'))
            