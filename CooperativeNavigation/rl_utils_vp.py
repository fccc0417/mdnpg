import numpy as np
import json
import torch


###### Load weight matrix
def load_pi(num_agents, topology):
    """Load the connectivity weight matrix."""
    wsize = num_agents
    if topology == 'dense':
        topo = 1
    elif topology == 'ring':
        topo = 2
    elif topology == 'bipartite':
        topo = 3

    with open('topology/connectivity/%s_%s.json' % (wsize, topo), 'r') as f:
        cdict = json.load(f)  # connectivity dict
    return cdict['pi']


def moving_average(a, window_size):
    """Move average for averaged returns."""
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


@torch.no_grad()
def take_value_param_consensus(agents, pi):
    """Take parameters of value function consensus."""
    layer_1_w = []
    layer_1_b = []
    layer_2_w = []
    layer_2_b = []
    layer_3_w = []
    layer_3_b = []

    for agent in agents:
        layer_1_w.append(agent.value_net.fc1.weight.data)
        layer_1_b.append(agent.value_net.fc1.bias.data)
        layer_2_w.append(agent.value_net.fc2.weight.data)
        layer_2_b.append(agent.value_net.fc2.bias.data)
        layer_3_w.append(agent.value_net.fc3.weight.data)
        layer_3_b.append(agent.value_net.fc3.bias.data)

    for agent_idx, agent in enumerate(agents):
        agent.value_net.fc1.weight.data = torch.sum(
            torch.stack(tuple(layer_1_w)) * torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1), 0).clone()
        agent.value_net.fc1.bias.data = torch.sum(
            torch.stack(tuple(layer_1_b)) * torch.tensor(pi[agent_idx]).unsqueeze(-1), 0).clone()

        agent.value_net.fc2.weight.data = torch.sum(
            torch.stack(tuple(layer_2_w)) * torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1), 0).clone()
        agent.value_net.fc2.bias.data = torch.sum(
            torch.stack(tuple(layer_2_b)) * torch.tensor(pi[agent_idx]).unsqueeze(-1), 0).clone()

        agent.value_net.fc3.weight.data = torch.sum(
            torch.stack(tuple(layer_3_w)) * torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1), 0).clone()
        agent.value_net.fc3.bias.data = torch.sum(
            torch.stack(tuple(layer_3_b)) * torch.tensor(pi[agent_idx]).unsqueeze(-1), 0).clone()
    return agents


@torch.no_grad()
def take_dual_param_consensus(agents, pi):
    """Take parameters of dual function consensus."""
    layer_1_w = []
    layer_1_b = []
    layer_2_w = []
    layer_2_b = []
    layer_3_w = []
    layer_3_b = []

    for agent in agents:
        layer_1_w.append(agent.dual_net.fc1.weight.data)
        layer_1_b.append(agent.dual_net.fc1.bias.data)
        layer_2_w.append(agent.dual_net.fc2.weight.data)
        layer_2_b.append(agent.dual_net.fc2.bias.data)
        layer_3_w.append(agent.dual_net.fc3.weight.data)
        layer_3_b.append(agent.dual_net.fc3.bias.data)

    for agent_idx, agent in enumerate(agents):
        agent.dual_net.fc1.weight.data = torch.sum(
            torch.stack(tuple(layer_1_w)) * torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1), 0).clone()
        agent.dual_net.fc1.bias.data = torch.sum(torch.stack(tuple(layer_1_b)) * torch.tensor(pi[agent_idx]).unsqueeze(-1),
                                           0).clone()

        agent.dual_net.fc2.weight.data = torch.sum(
            torch.stack(tuple(layer_2_w)) * torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1), 0).clone()
        agent.dual_net.fc2.bias.data = torch.sum(torch.stack(tuple(layer_2_b)) * torch.tensor(pi[agent_idx]).unsqueeze(-1),
                                           0).clone()

        agent.dual_net.fc3.weight.data = torch.sum(
            torch.stack(tuple(layer_3_w)) * torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1), 0).clone()
        agent.dual_net.fc3.bias.data = torch.sum(torch.stack(tuple(layer_3_b)) * torch.tensor(pi[agent_idx]).unsqueeze(-1),
                                           0).clone()
    return agents


def calc_grad(obj, model):
    """Calculate gradients for an objective function."""
    grads = torch.autograd.grad(obj, model.parameters(), create_graph=True, retain_graph=True)  # type: ignore
    grads = torch.cat([grad.reshape(-1) for grad in grads]).detach()
    return grads


def compute_discounted_rewards(gamma, rewards, lmbda=1):
    """Calculate cumulative discounted rewards."""
    rewards = rewards.detach().numpy()
    discounted_reward = 0.0
    for reward in rewards[::-1]:
        discounted_reward = gamma * lmbda * discounted_reward + reward
    return torch.tensor(discounted_reward, dtype=torch.float)

