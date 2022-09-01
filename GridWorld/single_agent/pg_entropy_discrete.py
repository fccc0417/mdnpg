"""
Paper: "A Decentralized Policy Gradient Approach to Multi-Task Reinforcement Learning"
"""
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
from GridWorld.envs.gridworld import GridWorldEnv


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


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PGwithEntropy:
    """ PG with entropy-based regularization."""
    def __init__(self, state_space, action_space, lmbda, critic_lr, gamma, device, entropy_para, actor_lr):
        self.state_dim = state_space.shape[0]
        self.action_dim = action_space.n
        self.actor = PolicyNet(self.state_dim, self.action_dim).to(device)
        self.critic = ValueNet(self.state_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda  # GAE参数
        self.device = device
        self.entropy_para = entropy_para
        self.actor_lr = actor_lr

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def compute_grads(self, transition_dict, advantage):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        return obj_grad

    def policy_learn(self, transition_dict, advantage, lr):
        obj_grad = self.compute_grads(transition_dict, advantage)
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        new_para = old_para + lr * obj_grad
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters())

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
        self.critic_optimizer.step()

        # Add entropy penalty to the advantage function.
        log_probs_all = torch.log(self.actor(states))
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        sum_log_probs = torch.sum(log_probs_all, dim=1)
        entropy_term = 1 / self.action_dim * sum_log_probs + np.log(self.action_dim)
        entropy_term = self.entropy_para * entropy_term
        return advantage + entropy_term


def set_args(seed=0):
    parser = argparse.ArgumentParser(description='Single agent PG with entropy')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--critic_lr', type=float, default=1e-2, help='value learning rate')
    parser.add_argument('--lmbda', type=float, default=0.95, help='lambda')
    parser.add_argument('--entropy_para', type=float, default=0.1, help='entropy parameter')
    parser.add_argument('--actor_lr', type=float, default=2.5e-4, help='policy learning rate')
    parser.add_argument('--seed', type=int, default=seed, help='random seed')
    parser.add_argument('--num_agents', type=int, default=1, help='number of agents')
    parser.add_argument('--max_eps_len', type=int, default=100, help='number of steps per episode')
    parser.add_argument('--num_episodes', type=int, default=2000, help='number training episodes')
    args = parser.parse_args()
    return args


def run(seed):
    args = set_args(seed)
    np.random.seed(seed)
    torch.manual_seed(args.seed)
    print(args.seed)
    num_episodes = args.num_episodes
    entropy_para = args.entropy_para
    gamma = args.gamma
    lmbda = args.lmbda
    critic_lr = args.critic_lr
    actor_lr = args.actor_lr
    max_eps_len = args.max_eps_len
    device = torch.device("cpu")
    env = GridWorldEnv(seed=seed, random_pos=True)
    agent = PGwithEntropy(env.observation_space, env.action_space, lmbda, critic_lr, gamma, device, entropy_para, actor_lr)
    return_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                for t in range(max_eps_len):
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    if done:
                        break
                return_list.append(episode_return)
                advantage = agent.update_value(transition_dict)
                agent.policy_learn(transition_dict, advantage, agent.actor_lr)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    mv_return_list = moving_average(return_list, 9)
    return return_list, mv_return_list, agent


if __name__ == '__main__':
    env_name = 'GridWorld'
    seeds = [0]
    for seed in seeds:
        print(f"seed={seed}")
        return_list, mv_return_list, agent = run(seed)
        np.save(os.path.join('records/' + env_name+'_' + str(seed)+'_' + '_pg_entrpoy_avg_return.npy'), mv_return_list)
        torch.save(agent, os.path.join('agents/'+'pgen_agent_'+str(seed)+'.pth'))

