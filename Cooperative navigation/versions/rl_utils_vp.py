import numpy as np
import json
import collections
import random
import torch
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

###### Function related to topology and pi
def load_pi(num_agents, topology):
    wsize = num_agents
    if topology == 'dense':
        topo = 1
    elif topology == 'ring':
        topo = 2
    elif topology == 'bipartite':
        topo = 3

    with open('generate_topology/connectivity/%s_%s.json' % (wsize, topo), 'r') as f:
        cdict = json.load(f)  # connectivity dict.
    return cdict['pi']

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

@torch.no_grad()
def take_value_param_consensus(agents, pi):
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



# for agent, grads in zip(agents, consensus_grad_lists):
#     update_param(agent, grads, lr=args.grad_lr)

# def get_flat_grad(y: torch.Tensor, model, **kwargs):
#     grads = torch.autograd.grad(y, model.parameters(), **kwargs)  # type: ignore
#     return torch.cat([grad.reshape(-1) for grad in grads])
