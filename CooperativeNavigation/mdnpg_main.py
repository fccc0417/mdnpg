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
    """Set parameters."""
    parser = argparse.ArgumentParser(description='MDNPG')
    parser.add_argument('--num_agents', type=int, default=num_agents, help='number of agents')
    parser.add_argument('--num_landmarks', type=int, default=num_agents, help='number of landmarks')
    parser.add_argument('--action_dim', type=int, default=5, help='number of actions')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='value learning rate')
    parser.add_argument('--grad_lr', type=float, default=5e-3, help='policy learning rate')
    parser.add_argument('--lmbda', type=float, default=0.95, help='lambda for GAE')
    parser.add_argument('--kl_constraint', type=float, default=0.0005, help='KL constraint') # 0.0005
    parser.add_argument('--max_eps_len', type=int, default=20, help='number of steps per episode')
    parser.add_argument('--num_episodes', type=int, default=20000, help='number training episodes')
    parser.add_argument('--beta', type=float, default=beta, help='beta for momentum-based variance reduction')
    parser.add_argument('--min_isw', type=float, default=0.0, help='minimum value to set importance weight')
    parser.add_argument('--topology', type=str, default=topology, choices=('dense', 'ring', 'bipartite'))
    parser.add_argument('--init_lr', type=float, default=3e-5, help='policy learning rate for initialization')
    parser.add_argument('--minibatch_size', type=int, default=1, help='number of trajectories for batch gradient')
    parser.add_argument('--init_minibatch_size', type=int, default=1, help='number of trajectories for batch gradient in initialization')
    args = parser.parse_args()
    return args


def run(args, env_name):
    """Run MDNPG algorithm."""
    # timestr = str(time()).replace('.', 'p')
    fpath2 = os.path.join('records', 'mdnpg_logs', str(num_agents) + '_agents', 'beta=' + str(args.beta), topology)
    if not os.path.isdir(fpath2):
        os.makedirs(fpath2)
    else:
        del_file(fpath2)
    writer = SummaryWriter(fpath2)

    # Generate an initial environment list for agents.
    sample_envs = []
    sample_env = make_particleworld.make_env(env_name, num_agents=args.num_agents, num_landmarks=args.num_landmarks)
    sample_env.discrete_action_input = True  # set action space to take in discrete numbers 0,1,2,3,4
    sample_env.seed(seed)  # set random seed for environment
    for _ in range(args.num_agents):
        env_copy = copy.deepcopy(sample_env)
        sample_envs.append(env_copy)

    print('Observation Space:', sample_env.observation_space)
    print('Action Space:', sample_env.action_space)
    print('Number of agents:', sample_env.n)
    sample_obs = sample_env.reset()
    sample_obs = np.concatenate(sample_obs).ravel() 

    # Generate a training agent list.
    agents = []
    for i in range(args.num_agents):
        agents.append(MomentumNPG(num_agents=args.num_agents, state_dim=len(sample_obs), action_dim=args.action_dim, lmbda=args.lmbda,
                                  kl_constraint=args.kl_constraint, 
                                  critic_lr=args.critic_lr, gamma=args.gamma,
                                  device=device, min_isw=args.min_isw, beta=args.beta))  

    # Load the connectivity weight matrix.
    pi = load_pi(num_agents=args.num_agents, topology=args.topology)

    # Copy old polices.
    old_policies = []
    for agent in agents:
        old_policy = copy.deepcopy(agent.actors)
        old_policies.append(old_policy)

    # Mini-batch initialization.
    prev_v_lists, y_lists = initialization_gt(sample_envs, agents, pi, lr=args.init_lr, minibatch_size=args.init_minibatch_size,
                                           max_eps_len=args.max_eps_len)

    # Generate an environment list for agents for training.
    envs = []
    env = make_particleworld.make_env(env_name, num_agents=args.num_agents, num_landmarks=args.num_landmarks)
    env.discrete_action_input = True
    env.seed(seed)
    for _ in range(args.num_agents):
        env_copy = copy.deepcopy(env)
        envs.append(env_copy)

    return_list = []  # return list for all episodes
    error_list = []  # consensus error list for all episodes
    for i in range(10):
        with tqdm(total=int(args.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episodes / 10)):
                # Copy old agents at timestep t-1 (\theta^{t-1}).
                phis_list = copy.deepcopy(old_policies)
                # Copy training agents at timestep t (\theta^{t}), used for the next episode.
                old_policies = []
                for agent in agents:
                    old_policy = copy.deepcopy(agent.actors)
                    old_policies.append(old_policy)

                episode_returns = 0
                v_lists = []  # list of gradient estimator v^t for all agents
                states_lists = []  # list of states for all agents

                # Calculate consensus error.
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

                # Generate gradient estimators.
                for idx, (agent, env) in enumerate(zip(agents, envs)):
                    minibatch_v = []  # list of minibatch gradients v^t for each agent
                    states_list = []  # list of states for each agent
                    for b in range(args.minibatch_size):  # Only need minibatch_size=1 for momentum-based variance reduction.
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

                        # Update value function and calculate advantage function.
                        advantage = agent.update_value(transition_dict)
                        # Calculate gradient estimator v^t for each episode.
                        single_traj_v = agent.compute_v_list(transition_dict, advantage, prev_v_lists[idx], phis_list[idx], args.beta)
                        single_traj_v = torch.stack(single_traj_v, dim=0)  # tensor stack
                        minibatch_v.append(single_traj_v)

                    minibatch_v = torch.stack(minibatch_v, dim=0)
                    grad_v = torch.mean(minibatch_v, dim=0)  # average the batch gradient estimators
                    v_lists.append(grad_v)
                    states_lists.append(states_list)

                return_list.append(episode_returns/args.minibatch_size)

                # Take gradient consensus for gradient estimator y^t.
                y_lists = take_grad_consensus(y_lists, pi)
                # Generate y^{t+1} by gradient tracking.
                next_y_lists = update_y_lists(y_lists, prev_v_lists, v_lists)

                # Calculate preconditioner using natural policy gradient.
                update_grad_lists = []
                for next_y_list, agent, states_list in zip(next_y_lists, agents, states_lists):
                    direction_grad_list = agent.compute_precondition_with_y(states_list, next_y_list)
                    update_grad_lists.append(direction_grad_list)

                # Take gradient consensus for H^t*y^{t+1}.
                consensus_grad_lists = take_grad_consensus(update_grad_lists, pi)
                # Take parameters consensus for \theta^t.
                agents = take_param_consensus(agents, pi)

                # Update parameters for all agents.
                for agent, grads in zip(agents, consensus_grad_lists):
                    update_param(agent, grads, lr=args.grad_lr)

                y_lists = copy.deepcopy(next_y_lists)  # copies of the last y
                prev_v_lists = copy.deepcopy(v_lists)  # copies of the last v

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (args.num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:]),
                                      'error': '%.3f' % np.mean(error_list[-10:])})
                    writer.add_scalar("rewards", np.mean(return_list[-10:]), args.num_episodes / 10 * i + i_episode + 1)

                pbar.update(1)
    mv_return_list = moving_average(return_list, 9)
    return return_list, mv_return_list, error_list, agents


if __name__ == '__main__':
    env_name = 'simple_spread'
    num_agents = 5
    topologies = ['ring']  # ['dense', bipartite, ring]
    betas = [0.2]
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

        # Save the trained models.
        for idx, agent in enumerate(agents):
            torch.save(agent, os.path.join(fpath, 'agent' + str(idx) + '.pth'))
            