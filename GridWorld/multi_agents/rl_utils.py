import copy
import torch
import json
import numpy as np


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
        cdict = json.load(f)  # connectivity dict.
    return cdict['pi']


def take_param_consensus(agents, pi):
    """Take parameters consensus."""
    layer_1_w = []
    layer_1_b = []

    layer_2_w = []
    layer_2_b = []

    for agent in agents:
        layer_1_w.append(agent.actor.dense1.weight.data)
        layer_1_b.append(agent.actor.dense1.bias.data)

        layer_2_w.append(agent.actor.dense2.weight.data)
        layer_2_b.append(agent.actor.dense2.bias.data)

    for agent_idx, agent in enumerate(agents):
        agent.actor.dense1.weight.data = torch.sum(
            torch.stack(tuple(layer_1_w)) * torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1), 0).clone()
        agent.actor.dense1.bias.data = torch.sum(torch.stack(tuple(layer_1_b)) * torch.tensor(pi[agent_idx]).unsqueeze(-1),
                                           0).clone()

        agent.actor.dense2.weight.data = torch.sum(
            torch.stack(tuple(layer_2_w)) * torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1), 0).clone()
        agent.actor.dense2.bias.data = torch.sum(torch.stack(tuple(layer_2_b)) * torch.tensor(pi[agent_idx]).unsqueeze(-1),
                                           0).clone()

    return agents


def take_grad_consensus(grad_list_flat, pi):
    """Take gradient consensus."""
    consensus_grads = []
    for j in range(len(grad_list_flat)):
        grad_cons = torch.sum(torch.stack(tuple(grad_list_flat)) * torch.tensor(pi[j]).unsqueeze(-1), 0).clone()
        consensus_grads.append(grad_cons)
    return consensus_grads


def update_param(agent, obj_grad, lr=3e-4):
    """update parameters for an agent"""
    old_para = torch.nn.utils.convert_parameters.parameters_to_vector(agent.actor.parameters())
    new_para = old_para + lr * obj_grad
    torch.nn.utils.convert_parameters.vector_to_parameters(new_para, agent.actor.parameters())


def moving_average(a, window_size):
    """Move average for averaged returns."""
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def compute_advantage(gamma, lmbda, td_delta):
    """Calculate advantage function using GAE."""
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


def update_y(y, v, prev_v):
    """Update gradient estimator y^{t+1} using gradient tracking."""
    next_y = y + v - prev_v
    return next_y


def initialization_gt(sample_envs, agents, pi, lr, minibatch_size, max_eps_len, algo='pg'):
    """Initialization for traning agents."""
    prev_v_list = []
    y_list = []
    states_lists = []  # list of states for all agents

    for idx, (agent, sample_env) in enumerate(zip(agents, sample_envs)):
        minibatch_grads_n = []
        states_list = []  # list of states for each agent        
        print("Initializing for "+ f"agent {idx}" + "...")
        for i in range(minibatch_size):
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state = sample_env.reset()
            done = False
            for t in range(max_eps_len):
                if algo == 'npg':
                    states_list.append(state)
                action = agent.take_action(state)
                next_state, reward, done, _ = sample_env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
                reset = t == max_eps_len - 1
                if done or reset:
                    print("Agent "+ str(idx) + ': Batch Initial Trajectory ' + str(i) + ': { Reward:', episode_return, 'Done:', done , '}')
                    break

            advantage = agent.update_value(transition_dict)
            single_traj_grads = agent.compute_grads(transition_dict, advantage)
            minibatch_grads_n.append(single_traj_grads)

        avg_grads_n = torch.mean(torch.stack(minibatch_grads_n, dim=1), dim=1)
        prev_v = copy.deepcopy(avg_grads_n)
        y_grad = copy.deepcopy(avg_grads_n)
        prev_v_list.append(prev_v)
        y_list.append(y_grad)
        if algo == 'npg':
            states_lists.append(states_list)

    consensus_y_list = take_grad_consensus(y_list, pi)

    if algo == 'npg':
        update_grad_list = []
        for y, agent, states_list in zip(consensus_y_list, agents, states_lists):
            direction_grad = agent.compute_precondition_with_y(states_list, y, None, None)
            update_grad_list.append(direction_grad)
        consensus_grad_list = take_grad_consensus(update_grad_list, pi)
    else:
        consensus_grad_list = copy.deepcopy(consensus_y_list)

    agents = take_param_consensus(agents, pi)

    for agent, y in zip(agents, consensus_grad_list):
        update_param(agent, y, lr=lr)

    return prev_v_list, consensus_y_list

