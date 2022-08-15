import argparse
from tqdm import tqdm
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy
import os
# from envs.gridworld_4_test import GridWorldEnv
from envs.gridworld import GridWorldEnv
from envs.init_agent_pos_4_single import *

map_path_0 = "../envs/grid_maps/map_0.npy"
map_path_1 = "../envs/grid_maps/map_1.npy"
map_path_2 = "../envs/grid_maps/map_2.npy"
map_path_3 = "../envs/grid_maps/map_3.npy"
map_path_4 = "../envs/grid_maps/map_4.npy"


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


def initialization(sample_env, agent, max_eps_len=150, lr=1e-4, minibatch=10):
    return_list = []
    minibatch_grads = []
    for _ in range(minibatch):
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = sample_env.reset()
        done = False

        for t in range(max_eps_len): #while not(done): #for t in range(max_eps_len): #
            action = agent.take_action(state)
            next_state, reward, done, _ = sample_env.step(action)
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
        single_traj_grads = agent.compute_grads(transition_dict, advantage)
        minibatch_grads.append(single_traj_grads)

    prev_u = torch.mean(torch.stack(minibatch_grads, dim=1), dim=1)  #np.mean(minibatch_grads) or x.mean() # average across batch

    old_para = torch.nn.utils.convert_parameters.parameters_to_vector(agent.actor.parameters())
    new_para = old_para + lr * prev_u
    torch.nn.utils.convert_parameters.vector_to_parameters(new_para, agent.actor.parameters())
    return prev_u


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


class Momentum_TRPO:
    """ TRPO算法 """
    def __init__(self,  state_space, action_space, lmbda,
                 kl_constraint, alpha, critic_lr, gamma, device, min_isw, beta):
        state_dim = state_space.shape[0]
        action_dim = action_space.n
        # 策略网络参数不需要优化器更新
        self.actor = PolicyNet(state_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda  # GAE参数
        self.kl_constraint = kl_constraint  # KL距离最大限制
        self.alpha = alpha  # 线性搜索参数
        self.device = device
        self.min_isw = min_isw
        self.beta = beta

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        # 计算黑塞矩阵和一个向量的乘积
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists,
                                                 new_action_dists))  # 计算平均KL距离
        kl_grad = torch.autograd.grad(kl,
                                      self.actor.parameters(),
                                      create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # KL距离的梯度先和向量进行点积运算
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                    self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector + 0.1*vector  #+ 0.1*vector

    def conjugate_gradient(self, grad, states, old_action_dists):  # 共轭梯度法求解方程
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):  # 共轭梯度主循环
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

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):  # 计算策略目标
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs,
                    old_action_dists, max_vec):  # 线性搜索
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage,
                                             old_log_probs, self.actor)
        for i in range(15):  # 线性搜索主循环
            coef = self.alpha**i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())
            new_action_dists = torch.distributions.Categorical(
                new_actor(states))
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(old_action_dists,
                                                     new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                 old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
            # if new_obj > 0 and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def compute_grads(self, transition_dict, advantage):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        return obj_grad

    def compute_IS_weight(self, action_list, state_list, phi, min_isw):
        log_probs = torch.log(self.actor(state_list).gather(1, action_list)).detach()
        prob_tau = torch.prod(log_probs)
        old_policy_log_probs = torch.log(phi(state_list).gather(1, action_list)).detach()
        prob_old_tau = torch.prod(old_policy_log_probs)
        weight = prob_old_tau / (prob_tau + 1e-8)
        weight = np.max((min_isw, weight))
        return weight

    def compute_grad_traj_prev_weights(self, state_list, action_list, phi, advantage):
        old_policy_log_probs = torch.log(phi(state_list).gather(1, action_list)).detach()
        old_policy_surrogate_obj = self.compute_surrogate_obj(state_list, action_list, advantage,
                                                   old_policy_log_probs, phi)
        grads = torch.autograd.grad(old_policy_surrogate_obj, phi.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        return obj_grad

    def compute_u(self, grad, prev_u, isw, prev_g, beta):
        grad_surrogate = beta * grad + (1 - beta) * (prev_u + grad - isw * prev_g)
        return grad_surrogate

    def policy_learn(self, transition_dict, advantage, prev_u, phi):  # 更新策略函数
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        old_action_dists = torch.distributions.Categorical(self.actor(states).detach())

        isw = self.compute_IS_weight(actions, states, phi, self.min_isw)
        prev_g = self.compute_grad_traj_prev_weights(states, actions, phi, advantage)
        obj_grad = self.compute_grads(transition_dict, advantage)
        grad_u = self.compute_u(obj_grad, prev_u, isw, prev_g, self.beta)

        # 用共轭梯度法计算x = H^(-1)g
        descent_direction = self.conjugate_gradient(grad_u, states, old_action_dists)

        Hd = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint /
                              (torch.dot(descent_direction, Hd) + 1e-8))
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())
        new_para = self.line_search(states, actions, advantage, old_log_probs, old_action_dists,
                                    descent_direction * max_coef)  # 线性搜索
        # lr = 1e-5
        # new_para = old_para + descent_direction * max_coef * lr
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters())  # 用线性搜索后的参数更新策略
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


def set_args(beta=0.2, seed=0):
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
    parser.add_argument('--lmbda', type=float, default=0.95, help='lambda')
    parser.add_argument('--critic_lr', type=float, default=1e-2, help='critic_lr')
    parser.add_argument('--kl_constraint', type=float, default=8e-5, help='kl_constraint')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')  # 0.002
    parser.add_argument('--actor_lr', type=float, default=1e-4, help='actor_lr')  #3e-3
    parser.add_argument('--seed', type=int, default=seed, help='random seed (default: 0)')
    parser.add_argument('--max_eps_len', type=int, default=200, help='Number of steps per episode')
    parser.add_argument('--num_episodes', type=int, default=2000, help='Number training episodes')
    parser.add_argument('--beta', type=float, default=beta, help='Beta term for surrogate gradient')
    parser.add_argument('--min_isw', type=float, default=0.0, help='Minimum value to set ISW')
    parser.add_argument('--minibatch_size', type=int, default=32, help='Number of trajectory for warm startup')
    parser.add_argument('--random_loc', type=bool, default=True, help='Each episode use random initial location')
    args = parser.parse_args()
    return args


def run(beta, seed):
    args = set_args(beta, seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(args.seed)
    num_episodes = args.num_episodes
    gamma = args.gamma
    lmbda = args.lmbda
    beta = args.beta
    min_isw = args.min_isw
    actor_lr = args.actor_lr
    minibatch_size = args.minibatch_size
    critic_lr = args.critic_lr
    kl_constraint = args.kl_constraint
    max_eps_len = args.max_eps_len
    alpha = args.alpha
    device = torch.device("cpu") #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent_pos = np.random.randint(0, 10, 2)
    env = GridWorldEnv(seed=seed, agent_pos=agent_pos)
    # env = GridWorldEnv(grid_map_path=map_path_4)
    agent = Momentum_TRPO(env.observation_space, env.action_space, lmbda, kl_constraint, alpha, critic_lr, gamma, device, min_isw, beta)
    old_policy = copy.deepcopy(agent.actor)
    prev_u = initialization(env, agent, max_eps_len=max_eps_len, lr=actor_lr, minibatch=minibatch_size)


    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                phi = copy.deepcopy(old_policy)
                # old_agent is now updated agent
                old_policy = copy.deepcopy(agent.actor)
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                if args.random_loc:
                    agent_pos = agent_pos_reset(env)
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
                grad_u = agent.policy_learn(transition_dict, advantage, prev_u, phi)
                prev_u = grad_u

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    mv_return_list = moving_average(return_list, 19)
    return return_list, mv_return_list, agent


if __name__ == '__main__':
    env_name = 'GridWolrld'
    betas = [0.6]  # [0.2, 0.4, 0.6, 0.8, 1] [0.8, 0.8, 0.8]
    labels = ['beta=0.6', 'beta=0.6', 'beta=0.6', 'beta=0.6', 'beta=0.6'] #['beta=0.2', 'beta=0.4', 'beta=0.6', 'beta=0.8', 'beta=1'] ['beta=0.8', 'beta=0.8', 'beta=0.8']
    seeds = [1]  #[6, 7, 8, 9, 10]
    return_lists = []
    mv_return_lists = []

    for beta, seed in zip(betas, seeds):
        print(f"beta={beta}, seed={seed}")
        return_list, mv_return_list, agent = run(beta, seed)
        return_lists.append(return_list)
        mv_return_lists.append(mv_return_list)
        torch.save(agent, os.path.join('agents/'+'npg_agent_'+str(seed)+'.pth'))

    plt.figure()
    for return_list, label, seed in zip(return_lists, labels, seeds):
        plt.plot(return_list, label=label)
        # np.save(os.path.join('records/'+label+'_trpo_return.npy'), return_list)

    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.legend()
    plt.title('TRPO on {}'.format(env_name))
    # plt.savefig('records/trpo_discrete_1.jpg')
    plt.show()

    plt.figure()
    for return_list, label, seed in zip(mv_return_lists, labels, seeds):
        plt.plot(return_list, label=label)
        np.save(os.path.join('records/'+env_name+'_'+str(seed)+'_'+label+'_trpo_avg_return.npy'), return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Moving_average Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.legend()
    plt.savefig(os.path.join('records/'+env_name+ '_'+ label+'_trpo.jpg'))
    plt.show()
