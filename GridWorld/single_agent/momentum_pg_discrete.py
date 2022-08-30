"""
Reference: https://github.com/boyu-ai/Hands-on-RL
"""
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import copy
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


def initialization(sample_env, agent, max_eps_len=150, lr=1e-4, minibatch=10):
    return_list = []
    minibatch_grads = []
    for _ in range(minibatch):
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = sample_env.reset()
        done = False

        for t in range(max_eps_len):
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

    prev_v = torch.mean(torch.stack(minibatch_grads, dim=1), dim=1)
    old_para = torch.nn.utils.convert_parameters.parameters_to_vector(agent.actor.parameters())
    new_para = old_para + lr * prev_v
    torch.nn.utils.convert_parameters.vector_to_parameters(new_para, agent.actor.parameters())
    return prev_v


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


class Momentum_PG:
    """ Momentum-based PG """
    def __init__(self,  state_space, action_space, lmbda, critic_lr, gamma, device, min_isw, beta, actor_lr):
        state_dim = state_space.shape[0]
        action_dim = action_space.n
        self.actor = PolicyNet(state_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.device = device
        self.min_isw = min_isw
        self.beta = beta
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

    def compute_IS_weight(self, action_list, state_list, phi, min_isw):
        probs = self.actor(state_list).gather(1, action_list).detach()
        old_policy_probs = phi(state_list).gather(1, action_list).detach()
        weights = old_policy_probs / (probs + 1e-8)
        weight = torch.prod(weights)
        weight = np.max((min_isw, weight))
        return weight

    def compute_grad_traj_prev_weights(self, state_list, action_list, phi, advantage):
        old_policy_log_probs = torch.log(phi(state_list).gather(1, action_list)).detach()
        old_policy_surrogate_obj = self.compute_surrogate_obj(state_list, action_list, advantage,
                                                   old_policy_log_probs, phi)
        grads = torch.autograd.grad(old_policy_surrogate_obj, phi.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        return obj_grad


    def compute_v(self, grad, prev_v, isw, prev_g, beta):
        grad_surrogate = beta * grad + (1 - beta) * (prev_v + grad - isw * prev_g)
        return grad_surrogate

    def policy_learn(self, transition_dict, advantage, prev_v, phi, lr):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)

        isw = self.compute_IS_weight(actions, states, phi, self.min_isw)
        prev_g = self.compute_grad_traj_prev_weights(states, actions, phi, advantage)
        obj_grad = self.compute_grads(transition_dict, advantage)
        grad_v = self.compute_v(obj_grad, prev_v, isw, prev_g, self.beta)

        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        new_para = old_para + lr * grad_v

        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())
        return grad_v

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


def set_args(beta=0.2, seed=0):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gamma', type=float, default=0.99, help='Single agent momentum PG')
    parser.add_argument('--critic_lr', type=float, default=1e-2, help='value learning rate')
    parser.add_argument('--lmbda', type=float, default=0.95, help='lambda')
    parser.add_argument('--actor_lr', type=float, default=5e-5, help='policy learning rate for initialization')
    parser.add_argument('--seed', type=int, default=seed, help='random seed (default: 0)')
    parser.add_argument('--max_eps_len', type=int, default=100, help='number of steps per episode')
    parser.add_argument('--num_episodes', type=int, default=2000, help='number training episodes')
    parser.add_argument('--beta', type=float, default=beta, help='beta for momentum-based VR')
    parser.add_argument('--min_isw', type=float, default=0.0, help='minimum value to set ISW')
    parser.add_argument('--minibatch_size', type=int, default=32, help='number of trajectory for batch gradients in initialization')
    args = parser.parse_args()
    return args


def run(beta, seed):
    args = set_args(beta, seed)
    print(args.seed)
    np.random.seed(seed)
    torch.manual_seed(args.seed)
    num_episodes = args.num_episodes
    gamma = args.gamma
    lmbda = args.lmbda
    beta = args.beta
    min_isw = args.min_isw
    minibatch_size = args.minibatch_size
    critic_lr = args.critic_lr
    actor_lr = args.actor_lr
    max_eps_len = args.max_eps_len
    device = torch.device("cpu")
    env = GridWorldEnv(seed=seed)
    # env = GridWorldEnv(grid_map_path=map_path_4)
    agent = Momentum_PG(env.observation_space, env.action_space, lmbda, critic_lr, gamma, device, min_isw, beta, actor_lr)
    old_policy = copy.deepcopy(agent.actor)
    prev_v = initialization(env, agent, max_eps_len=max_eps_len, lr=actor_lr, minibatch=minibatch_size)

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
                for t in range(max_eps_len):  #while not done:
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
                grad_v = agent.policy_learn(transition_dict, advantage, prev_v, phi, agent.actor_lr)
                prev_v = grad_v

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    mv_return_list = moving_average(return_list, 19)
    return return_list, mv_return_list, agent


if __name__ == '__main__':
    env_name = 'GridWorld'
    betas = [0.6]
    labels = ['beta=0.6']
    seeds = [0]

    for beta, seed, label in zip(betas, seeds, labels):
        print(f"beta={beta}, seed={seed}")
        return_list, mv_return_list, agent = run(beta, seed)
        np.save(os.path.join('records/'+env_name+'_'+str(seed)+'_'+'_'+label+'_pg_avg_return.npy'), mv_return_list)
        torch.save(agent, os.path.join('agents/'+'pg_agent_'+str(seed)+'.pth'))

