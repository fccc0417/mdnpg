import argparse
from tqdm import tqdm
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy
import os


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


def initialization(sample_env, agent, lr=1e-4, minibatch=10):
    return_list = []
    minibatch_grads = []
    for _ in range(minibatch):
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = sample_env.reset()
        done = False

        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = sample_env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
        advantage = agent.update_value(transition_dict)
        single_traj_grads = agent.compute_grads(transition_dict, advantage)
        minibatch_grads.append(single_traj_grads)

    # update policy network
    # need grads to be shape num_agent x list of grads of every layer
    # minibatch_grads = np.asarray(minibatch_grads, dtype=object)
    prev_u = torch.mean(torch.stack(minibatch_grads, dim=1), dim=1)  #np.mean(minibatch_grads) or x.mean() # average across batch
    # prev_u = minibatch_grads.tolist()

    old_para = torch.nn.utils.convert_parameters.parameters_to_vector(agent.actor.parameters())
    new_para = old_para + lr * prev_u
    torch.nn.utils.convert_parameters.vector_to_parameters(new_para, agent.actor.parameters())
    return prev_u


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Momentum_PG_Continuous:
    """ Momentum PG算法 """
    def __init__(self,  state_space, action_space, lmbda, critic_lr, gamma, device, min_isw, beta, actor_lr):
        state_dim = state_space.shape[0]
        action_dim = action_space.shape[0]
        # 策略网络参数不需要优化器更新
        self.actor = PolicyNetContinuous(state_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda  # GAE参数
        self.device = device
        self.min_isw = min_isw
        self.beta = beta
        self.actor_lr = actor_lr

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return [action.item()]

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):  # 计算策略目标
        mu, std = actor(states)
        action_dists = torch.distributions.Normal(mu, std)
        log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def compute_grads(self, transition_dict, advantage):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)

        mu, std = self.actor(states)
        old_action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = old_action_dists.log_prob(actions)

        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        return obj_grad

    def compute_IS_weight(self, action_list, state_list, phi, min_isw):
        mu1, std1 = self.actor(state_list)
        action_dists = torch.distributions.Normal(mu1.detach(), std1.detach())
        log_probs = action_dists.log_prob(action_list)
        prob_tau = torch.prod(log_probs)

        mu2, std2 = phi(state_list)
        old_action_dists = torch.distributions.Normal(mu2.detach(), std2.detach())
        old_policy_log_probs = old_action_dists.log_prob(action_list)
        prob_old_tau = torch.prod(old_policy_log_probs)
        weight = prob_old_tau / (prob_tau + 1e-8)
        weight = np.max((min_isw, weight))
        return weight

    def compute_grad_traj_prev_weights(self, state_list, action_list, phi, advantage):
        mu, std = phi(state_list)
        old_action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_policy_log_probs = old_action_dists.log_prob(action_list)
        old_policy_surrogate_obj = self.compute_surrogate_obj(state_list, action_list, advantage,
                                                   old_policy_log_probs, phi)
        grads = torch.autograd.grad(old_policy_surrogate_obj, phi.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        return obj_grad

    def compute_u(self, grad, prev_u, isw, prev_g, beta):
        grad_surrogate = beta * grad + (1 - beta) * (prev_u + grad - isw * prev_g)
        return grad_surrogate

    def policy_learn(self, transition_dict, advantage, prev_u, phi, lr):  # 更新策略函数
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)

        isw = self.compute_IS_weight(actions, states, phi, self.min_isw)
        prev_g = self.compute_grad_traj_prev_weights(states, actions, phi, advantage)
        obj_grad = self.compute_grads(transition_dict, advantage)
        grad_u = self.compute_u(obj_grad, prev_u, isw, prev_g, self.beta)

        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())
        new_para = old_para + lr * grad_u

        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters())
        return grad_u

    def update_value(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()  # 更新价值函数

        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        return advantage

######################################################################


def set_args(beta=0.2, seed=0):
    parser = argparse.ArgumentParser(description='MPG')
    parser.add_argument('--gamma', type=float, default=0.999, help='discount factor (default: 0.99)')
    parser.add_argument('--critic_lr', type=float, default=2.5e-3, help='critic_lr')
    parser.add_argument('--lmbda', type=float, default=1, help='lambda')
    parser.add_argument('--actor_lr', type=float, default=5e-4, help='actor_lr')
    parser.add_argument('--seed', type=int, default=seed, help='random seed (default: 0)')
    # parser.add_argument('--max_eps_len', type=int, default=20, help='Number of steps per episode')
    parser.add_argument('--num_episodes', type=int, default=5000, help='Number training episodes')
    parser.add_argument('--beta', type=float, default=beta, help='Beta term for surrogate gradient')
    parser.add_argument('--min_isw', type=float, default=0.0, help='Minimum value to set ISW')
    parser.add_argument('--minibatch_size', type=int, default=1, help='Number of trajectory for warm startup')
    args = parser.parse_args()
    return args


def run(beta, env_name, seed):
    args = set_args(beta, seed)
    print(args.seed)
    torch.manual_seed(args.seed)
    num_episodes = args.num_episodes
    gamma = args.gamma
    lmbda = args.lmbda
    beta = args.beta
    min_isw = args.min_isw
    minibatch_size = args.minibatch_size
    critic_lr = args.critic_lr
    actor_lr = args.actor_lr
    device = torch.device("cpu") #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sample_env = gym.make(env_name)
    agent = Momentum_PG_Continuous(sample_env.observation_space, sample_env.action_space, lmbda, critic_lr, gamma, device, min_isw, beta, actor_lr)
    old_policy = copy.deepcopy(agent.actor)
    prev_u = initialization(sample_env, agent, lr=actor_lr, minibatch=minibatch_size)
    env = gym.make(env_name)
    env.seed(args.seed)
    return_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                phi = copy.deepcopy(old_policy)
                # old_agent is now updated agent
                old_policy = copy.deepcopy(agent.actor)
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                advantage = agent.update_value(transition_dict)
                grad_u = agent.policy_learn(transition_dict, advantage, prev_u, phi, agent.actor_lr)
                prev_u = grad_u

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    mv_return_list = moving_average(return_list, 9)
    return return_list, mv_return_list


if __name__ == '__main__':
    env_name = 'MountainCarContinuous-v0'
    betas = [0.8, 0.8]  # [0.2, 0.4, 0.6, 0.8, 1]
    labels = ['beta=0.8', 'beta=0.8']  #['beta=0.2', 'beta=0.4', 'beta=0.6', 'beta=0.8', 'beta=1']
    seeds = [6]
    return_lists = []
    mv_return_lists = []

    for beta, seed in zip(betas, seeds):
        print(f"beta={beta}, seed={seed}")
        return_list, mv_return_list = run(beta, env_name, seed)
        return_lists.append(return_list)
        mv_return_lists.append(mv_return_list)

    plt.figure()
    for return_list, label, seed in zip(return_lists, labels, seeds):
        plt.plot(return_list, label=label)
        # np.save(os.path.join('records/'+label+'_pg_return.npy'), return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.legend()
    plt.title('PG on {}'.format(env_name))
    # plt.savefig('records/pg_discrete_1.jpg')
    plt.show()

    plt.figure()
    for return_list, label, seed in zip(mv_return_lists, labels, seeds):
        plt.plot(return_list, label=label)
        np.save(os.path.join('records/'+env_name+'_'+str(seed)+'_'+'_'+label+'_pg_avg_return.npy'), return_list)

    plt.xlabel('Episodes')
    plt.ylabel('Moving_average Returns')
    plt.title('PG on {}'.format(env_name))
    plt.legend()
    plt.savefig(os.path.join('records/' + env_name + '_' + label + '_pg.jpg'))
    plt.show()

