import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from rl_utils_vp import *
from tools.tool import *
from value_propagation import ValuePropagation
from particle_envs import make_particleworld
import torch
import os

seed = 0
torch.manual_seed(seed)
# device = torch.device("cpu")  # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_args(num_agents=1, topology='dense'):
    parser = argparse.ArgumentParser(description='Multi-agent example')
    parser.add_argument('--num_agents', type=int, default=num_agents, help='Number of agents')
    parser.add_argument('--num_landmarks', type=int, default=num_agents, help='Number of landmarks')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
    parser.add_argument('--eta', type=float, default=0.1, help='eta, trade-off parameter')
    parser.add_argument('--adam_lr', type=float, default=5e-4, help='update lr')
    parser.add_argument('--lmbda', type=float, default=0.01, help='lambda')
    parser.add_argument('--T_dual', type=float, default=10, help='T_dual')
    parser.add_argument('--topology', type=str, default=topology, choices=('dense', 'ring', 'bipartite'))
    parser.add_argument('--max_eps_len', type=int, default=20, help='Number of steps per episode')
    parser.add_argument('--num_episodes', type=int, default=5000, help='Number training episodes')
    parser.add_argument('--buffer_epoches', type=int, default=1000, help='to collect buffer elements')
    parser.add_argument('--buffer_size', type=int, default=100000, help='buffer size')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--minimal_size', type=int, default=1000, help='minimal buffer size')
    args = parser.parse_args()
    return args

def initialize_buffer(sample_env, agents, minimal_size=50, buffer_epoches=1000):
    replay_buffer = ReplayBuffer(args.buffer_size)
    for eps in range(buffer_epoches):
        episode_returns = 0
        state = sample_env.reset()
        state = np.concatenate(state).ravel()
        for t in range(args.max_eps_len):
            actions = []
            for actor in agents:
                actions.append(actor.take_action(state))
            next_state, rewards, dones, _ = sample_env.step(actions)
            next_state = np.concatenate(next_state).ravel()
            done = all(item == True for item in dones)
            episode_returns += sum(rewards)
            replay_buffer.add(state, actions, rewards, next_state, done)
            state = next_state
            if done:
                break
        if replay_buffer.size() > minimal_size:
            print(f"End the initiliziation of  replybuff!!! AND rewards={episode_returns}\n")
            return replay_buffer


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
                                       eta=args.eta, lmbda=args.lmbda, gamma=args.gamma,
                                       T_dual=args.T_dual, adam_lr=args.adam_lr))

    env = make_particleworld.make_env(env_name, num_agents=args.num_agents, num_landmarks=args.num_landmarks)
    env.discrete_action_input = True
    env.seed(seed)

    replay_buffer = initialize_buffer(sample_env, agents, args.minimal_size, args.buffer_epoches)

    return_list = []

    for i in range(10):
        with tqdm(total=int(args.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episodes / 10)):

                # transition_dicts_d = []
                # transition_dicts_p = []

                episode_returns = 0
                state = env.reset()
                state = np.concatenate(state).ravel()

                for t in range(args.max_eps_len):
                    actions = []
                    for actor in agents:
                        actions.append(actor.take_action(state))
                    next_state, rewards, dones, _ = env.step(actions)
                    next_state = np.concatenate(next_state).ravel()
                    done = all(item == True for item in dones)
                    replay_buffer.add(state, actions, rewards, next_state, done)
                    state = next_state
                    episode_returns += np.sum(rewards)
                    reset = t == args.max_eps_len - 1
                    if done or reset:
                        # print('Batch Initial Trajectory ' + str(i) + ': Reward', episode_return, 'Done', done)
                        break
                for dual_epoch in range(args.T_dual):
                    b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(args.batch_size)
                    for idx, agent in enumerate(agents):
                        rewards = []
                        for j in range(len(b_rewards)):
                            rewards.append(b_rewards[j][idx])
                        transition_dict = {'states': b_states, 'actions': b_actions, 'next_states': b_next_states,
                                           'rewards': rewards, 'dones': b_dones}
                        agent.calc_dual_grad(idx, transition_dict)
                    agents = take_dual_param_consensus(agents, pi)
                    for agent in agents:
                        agent.update_dual_problem()


                b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(args.batch_size)
                for idx, agent in enumerate(agents):
                    rewards = []
                    for j in range(len(b_rewards)):
                        rewards.append(b_rewards[j][idx])
                    transition_dict = {'states': b_states, 'actions': b_actions, 'next_states': b_next_states,
                                       'rewards': rewards, 'dones': b_dones}
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
    env_name = 'simple_spread'
    num_agents = 5
    topologies = ['bipartite', 'dense', 'ring']
    labels = ['bipartite'] #, 'bipartite', 'ring'
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

    plt.figure()
    for return_list, label in zip(return_lists, labels):
        plt.plot(return_list, label=label)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.legend()
    plt.title('{}-agent Value Propagation on {} (discrete)'.format(num_agents, env_name))
    plt.savefig(os.path.join(fpath, 'return.jpg'))
    plt.show()

    plt.figure()
    for return_list, label in zip(mv_return_lists, labels):
        plt.plot(return_list, label=label)
    plt.xlabel('Episodes')
    plt.ylabel('Moving_average Returns')
    plt.title('{}-agent Value Propagation with GT on {} (discrete)'.format(num_agents, env_name))
    plt.legend()
    plt.savefig(os.path.join(fpath, 'avg_return.jpg'))
    plt.show()

