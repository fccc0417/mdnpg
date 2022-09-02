import torch.nn as nn
import torch.nn.functional as F
import torch
from rl_utils_vp import *


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        return self.fc3(x2)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.dense1 = nn.Linear(state_dim, 64)
        self.dense2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x1 = F.relu(self.dense1(x))
        x2 = self.dense2(x1)
        dist = F.softmax(x2, dim=-1)
        return dist


class DualFuncNet(torch.nn.Module):
    def __init__(self, state_dim, action_num):
        super(DualFuncNet, self).__init__()
        input_dim = state_dim+action_num
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        return

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        return self.fc3(x2)


class ValuePropagation:
    def __init__(self, num_agents, state_dim, action_dim, eta=0.1, lmbda=0.01, gamma=0.99, T_dual=4, value_lr=5e-4,
                 policy_lr=5e-4, dual_lr=5e-4):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eta = eta
        self.lmbda = lmbda
        self.gamma = gamma
        self.T_dual = T_dual
        self.value_lr = value_lr
        self.policy_lr = policy_lr
        self.dual_lr = dual_lr

        self.value_net = ValueNet(self.state_dim)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr)

        self.policy_net = PolicyNet(self.state_dim, self.action_dim)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

        self.dual_net = DualFuncNet(self.state_dim, self.num_agents)
        self.dual_optimizer = torch.optim.Adam(self.dual_net.parameters(), lr=self.dual_lr)

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action

    def calc_dual_grad(self, idx, transition_dict):
        dual_obj = -self.calc_dual_obj(idx, transition_dict)
        dual_loss = torch.mean(dual_obj)
        self.dual_optimizer.zero_grad()
        dual_loss.backward()

    @torch.no_grad()
    def update_dual_problem(self):
        self.dual_optimizer.step()

    def calc_primal_grad(self, idx, transition_dict):
        # consensus rho
        primal_obj = self.calc_primal_obj(idx, transition_dict)
        primal_loss = torch.mean(primal_obj)
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        primal_loss.backward()

    @torch.no_grad()
    def update_primal_problem(self):
        self.policy_optimizer.step()
        self.value_optimizer.step()

    def calc_dual_obj(self, idx, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        actions_list = torch.FloatTensor(transition_dict['actions'])
        delta_add_penalty = self.calc_delta_add_penalty(idx, transition_dict).detach()
        sa_pair = torch.cat([states, actions_list], dim=1)
        rho = self.dual_net(sa_pair)
        return -(delta_add_penalty -rho)**2  #self.eta *

    def calc_dual_obj2(self, idx, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        actions_list = torch.FloatTensor(transition_dict['actions'])
        delta_add_penalty = self.calc_delta_add_penalty(idx, transition_dict)
        sa_pair = torch.cat([states, actions_list], dim=1)
        rho = self.dual_net(sa_pair).detach()
        return -self.eta * (delta_add_penalty -rho)**2

    def calc_primal_obj(self, idx, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        delta_add_penalty = self.calc_delta_add_penalty(idx, transition_dict)
        dual_obj = self.calc_dual_obj2(idx, transition_dict)
        primal_obj = (delta_add_penalty - self.value_net(states))**2 + dual_obj
        return primal_obj

    def calc_delta_add_penalty(self, idx, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        actions_list = torch.tensor(transition_dict['actions']).T #TODP
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        log_probs = torch.log(self.policy_net(states).gather(1, actions_list[idx].unsqueeze(1)))
        delta = rewards + self.gamma * self.value_net(next_states) * (1 - dones)
        delta_add_penalty = delta - self.lmbda * self.num_agents * log_probs
        return delta_add_penalty

