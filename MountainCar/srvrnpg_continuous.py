"""
Paper: "An Improved Analysis of (Variance-Reduced) Policy Gradient and Natural Policy Gradient Methods"
"""
import argparse
from tqdm import tqdm
import torch
import numpy as np
import gym
import torch.nn.functional as F
import copy
import os


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
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

    prev_u = torch.mean(torch.stack(minibatch_grads, dim=1), dim=1)
    old_para = torch.nn.utils.convert_parameters.parameters_to_vector(agent.actor.parameters())
    new_para = old_para + lr * prev_u
    torch.nn.utils.convert_parameters.vector_to_parameters(new_para, agent.actor.parameters())


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


class SRVR_NPG_Continuous:
    """SRVR-NPG algorithm."""
    def __init__(self, state_space, action_space, lmbda,
                 kl_constraint, critic_lr, gamma, device, min_isw):
        state_dim = state_space.shape[0]
        action_dim = action_space.shape[0]
        self.actor = PolicyNetContinuous(state_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.kl_constraint = kl_constraint
        self.device = device
        self.min_isw = min_isw

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return [action.item()]

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        mu, std = self.actor(states)
        new_action_dists = torch.distributions.Normal(mu, std)

        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
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

    def update_para(self, update_grad, lr):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        new_para = old_para + lr * update_grad
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())

    def policy_learn(self, states_list, grad_u):
        states = torch.tensor(states_list, dtype=torch.float).to(self.device)

        mu, std = self.actor(states)
        old_action_dists = torch.distributions.Normal(mu.detach(), std.detach())

        descent_direction = self.conjugate_gradient(grad_u, states, old_action_dists)

        Hd = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8))
        update_grad = descent_direction * max_coef
        return update_grad

    def compute_pre_grads(self, transition_dict, advantage, phi):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        isw = self.compute_IS_weight(actions, states, phi, self.min_isw)
        prev_g = self.compute_grad_traj_prev_weights(states, actions, phi, advantage)
        return isw * prev_g

    def update_value(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        return advantage


def set_args(seed=0):
    parser = argparse.ArgumentParser(description='SRVR-NPG')
    parser.add_argument('--gamma', type=float, default=0.999, help='discount factor')
    parser.add_argument('--lmbda', type=float, default=1, help='lambda')
    parser.add_argument('--critic_lr', type=float, default=2.5e-3, help='value learning rate')
    parser.add_argument('--kl_constraint', type=float, default=0.05, help='kl constraint')
    parser.add_argument('--actor_lr', type=float, default=2.5e-3, help='policy learning rate')
    parser.add_argument('--init_lr', type=float, default=0, help='actor_lr')
    parser.add_argument('--seed', type=int, default=seed, help='random seed')
    parser.add_argument('--num_agents', type=int, default=1, help='number of agents')
    parser.add_argument('--num_episodes', type=int, default=5000, help='number training episodes')
    parser.add_argument('--min_isw', type=float, default=0.0, help='minimum importance weight')
    parser.add_argument('--batch_size', type=int, default=10, help='N')
    parser.add_argument('--epoch_size', type=int, default=2, help='m')
    parser.add_argument('--minibatch_size', type=int, default=3, help='B')
    args = parser.parse_args()
    return args


def run(env_name, seed=0):
    args = set_args(seed)
    torch.manual_seed(args.seed)
    num_episodes = args.num_episodes
    gamma = args.gamma
    lmbda = args.lmbda
    min_isw = args.min_isw
    actor_lr = args.actor_lr
    init_lr = args.init_lr
    epoch_size = args.epoch_size
    minibatch_size = args.minibatch_size
    batch_size = args.batch_size
    critic_lr = args.critic_lr
    kl_constraint = args.kl_constraint
    device = torch.device("cpu")
    sample_env = gym.make(env_name)
    agent = SRVR_NPG_Continuous(sample_env.observation_space, sample_env.action_space, lmbda, kl_constraint, critic_lr,
                                gamma, device, min_isw)

    initialization(sample_env, agent, lr=init_lr, minibatch=minibatch_size)
    env = gym.make(env_name)
    env.seed(args.seed)
    return_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                old_policy = copy.deepcopy(agent.actor)
                states_list = []
                minibatch_grads = []
                for n in range(batch_size):
                    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                    state = env.reset()
                    done = False
                    while not done:
                        action = agent.take_action(state)
                        states_list.append(state)
                        next_state, reward, done, _ = env.step(action)
                        transition_dict['states'].append(state)
                        transition_dict['actions'].append(action)
                        transition_dict['next_states'].append(next_state)
                        transition_dict['rewards'].append(reward)
                        transition_dict['dones'].append(done)
                        state = next_state

                    advantage = agent.update_value(transition_dict)
                    single_traj_grad = agent.compute_grads(transition_dict, advantage)
                    minibatch_grads.append(single_traj_grad)

                avg_grad = torch.mean(torch.stack(minibatch_grads, dim=1), dim=1)
                prev_u = copy.deepcopy(avg_grad)
                avg_update_grad = agent.policy_learn(states_list, avg_grad)
                agent.update_para(update_grad=avg_update_grad, lr=actor_lr)

                epoch_returns = []
                for m in range(epoch_size):
                    phi = copy.deepcopy(old_policy)
                    old_policy = copy.deepcopy(agent.actor)
                    episode_return = 0
                    states_list = []
                    minibatch_grads = []
                    minibatch_pre_grads = []
                    for b in range(minibatch_size):
                        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                        state = env.reset()
                        done = False
                        while not done:
                            action = agent.take_action(state)
                            states_list.append(state)
                            next_state, reward, done, _ = env.step(action)
                            transition_dict['states'].append(state)
                            transition_dict['actions'].append(action)
                            transition_dict['next_states'].append(next_state)
                            transition_dict['rewards'].append(reward)
                            transition_dict['dones'].append(done)
                            state = next_state
                            episode_return += reward

                        advantage = agent.update_value(transition_dict)
                        single_traj_grad = agent.compute_grads(transition_dict, advantage)
                        single_traj_pre_grad = agent.compute_pre_grads(transition_dict, advantage, phi)
                        minibatch_grads.append(single_traj_grad)
                        minibatch_pre_grads.append(single_traj_pre_grad)

                    avg_return = episode_return / minibatch_size
                    epoch_returns.append(avg_return)
                    avg_grad = torch.mean(torch.stack(minibatch_grads, dim=1), dim=1)
                    avg_pre_grad = torch.mean(torch.stack(minibatch_pre_grads, dim=1), dim=1)
                    avg_grad_u = prev_u + avg_grad - avg_pre_grad
                    avg_update_grad = agent.policy_learn(states_list, avg_grad_u)
                    agent.update_para(update_grad=avg_update_grad, lr=actor_lr)

                return_list.append(np.mean(epoch_returns))

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    mv_return_list = moving_average(return_list, 9)
    return return_list, mv_return_list


if __name__ == '__main__':
    env_name = 'MountainCarContinuous-v0'
    seeds = [0]
    return_lists = []
    mv_return_lists = []

    for seed in seeds:
        print(f"seed={seed}")
        return_list, mv_return_list = run(env_name, seed)
        np.save(os.path.join('records/' + env_name + '_' + str(seed) + '_srvrnpg_avg_return.npy'), mv_return_list)
        # np.save(os.path.join('records/' + env_name + '_' + str(seed) + '_srvrnpg_return.npy'), return_list)
        return_lists.append(return_list)
        mv_return_lists.append(mv_return_list)

