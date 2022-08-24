import copy
import os
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
    layer_1_w_lists = []
    layer_1_b_lists = []
    layer_2_w_lists = []
    layer_2_b_lists = []
    layer_3_w_lists = []
    layer_3_b_lists = []
    for i in range(len(agents)):
        layer_1_w = []
        layer_1_b = []
        layer_2_w = []
        layer_2_b = []
        layer_3_w = []
        layer_3_b = []
        for j in range(len(agents)):
            layer_1_w.append(agents[j].actors[i].dense1.weight.data)
            layer_1_b.append(agents[j].actors[i].dense1.bias.data)
            layer_2_w.append(agents[j].actors[i].dense2.weight.data)
            layer_2_b.append(agents[j].actors[i].dense2.bias.data)
            layer_3_w.append(agents[j].actors[i].dense3.weight.data)
            layer_3_b.append(agents[j].actors[i].dense3.bias.data)

        layer_1_w_lists.append(layer_1_w)
        layer_1_b_lists.append(layer_1_b)
        layer_2_w_lists.append(layer_2_w)
        layer_2_b_lists.append(layer_2_b)
        layer_3_w_lists.append(layer_3_w)
        layer_3_b_lists.append(layer_3_b)

    for agent_idx, agent in enumerate(agents):
        for i, actor in enumerate(agent.actors):
            actor.dense1.weight.data = torch.sum(
                torch.stack(tuple(layer_1_w_lists[i])) * torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1), 0).clone()
            actor.dense1.bias.data = torch.sum(torch.stack(tuple(layer_1_b_lists[i])) * torch.tensor(pi[agent_idx]).unsqueeze(-1), 0).clone()

            actor.dense2.weight.data = torch.sum(
                torch.stack(tuple(layer_2_w_lists[i])) * torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1), 0).clone()
            actor.dense2.bias.data = torch.sum(torch.stack(tuple(layer_2_b_lists[i])) * torch.tensor(pi[agent_idx]).unsqueeze(-1), 0).clone()

            actor.dense3.weight.data = torch.sum(
                torch.stack(tuple(layer_3_w_lists[i])) * torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1), 0).clone()
            actor.dense3.bias.data = torch.sum(torch.stack(tuple(layer_3_b_lists[i])) * torch.tensor(pi[agent_idx]).unsqueeze(-1), 0).clone()
    '''
    # TEST: When topo is dense, nums=[0]
    numss = []
    for agent in agents:
        nums = []
        for idx, actor in enumerate(agent.actors):
            a = torch.nn.utils.convert_parameters.parameters_to_vector(actor.parameters())
            b = torch.nn.utils.convert_parameters.parameters_to_vector(agents[4].actors[idx].parameters())
            num = np.linalg.norm(a.detach().numpy() - b.detach().numpy(), 2)
            nums.append(num)
        numss.append(nums)
    print(numss)
    '''
    return agents


def take_grad_consensus(grad_lists, pi):
    """Take gradient consensus."""
    re_grad_lists = []
    for i in range(len(grad_lists)):  # for the i-th copy in all agents
        re_grad_list = []
        for j in range(len(grad_lists)):
            re_grad_list.append(grad_lists[j][i])
        re_grad_lists.append(re_grad_list)

    consensus_grad_lists = []
    for idx in range(len(grad_lists)):
        consensus_grad_list = []
        for i in range(len(grad_lists)):  # the i-th copy for idx-agent
            grads = torch.sum(torch.stack(tuple(re_grad_lists[i])) * torch.tensor(pi[idx]).unsqueeze(-1), 0).clone()
            consensus_grad_list.append(grads)
        consensus_grad_lists.append(consensus_grad_list)
    return consensus_grad_lists


def update_y_lists(y_lists, prev_v_lists, v_lists):
    """Update gradient estimator y^{t+1} using gradient tracking."""
    next_y_lists = []
    for y_list, prev_v_list, v_list in zip(y_lists, prev_v_lists, v_lists):
        next_y_list = []
        for y, prev_v, v in zip(y_list, prev_v_list, v_list):
            y_new = y + v - prev_v
            next_y_list.append(y_new)
        next_y_lists.append(next_y_list)
    return next_y_lists


def update_param(agent, v_k_list, lr=3e-4):
    """update parameters for an agent"""
    for idx, actor in enumerate(agent.actors):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(actor.parameters())
        new_para = old_para + lr * v_k_list[idx]
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, actor.parameters())


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


def initialization_gt(sample_envs, agents, pi, lr=3e-4, minibatch_size=1, max_eps_len=20):
    """Initialization for traning agents."""
    prev_v_lists, y_lists = [], []
    for idx, (agent, sample_env) in enumerate(zip(agents, sample_envs)):
        minibatch_grads_n = []
        print("Initializing for " + f"agent {idx}" + "...")
        for i in range(minibatch_size):
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state = sample_env.reset()
            state = np.concatenate(state).ravel()
            for t in range(max_eps_len):
                actions = agent.take_actions(state)
                next_state, rewards, dones, _ = sample_env.step(actions)
                next_state = np.concatenate(next_state).ravel()
                done = all(item == True for item in dones)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(actions)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(rewards[idx])
                transition_dict['dones'].append(dones[idx])
                state = next_state
                episode_return += rewards[idx]
                reset = t == max_eps_len - 1
                if done or reset:
                    print("Agent "+ str(idx) + ': Batch Initial Trajectory ' + str(i) + ': { Reward:', episode_return, 'Done:', done , '}')
                    break

            advantage = agent.update_value(transition_dict)
            old_log_probs_list, log_probs_list = agent.calc_log_probs(transition_dict)
            single_traj_grads = agent.compute_grads(advantage, old_log_probs_list, log_probs_list)
            single_traj_grads = torch.stack(single_traj_grads, dim=0)
            minibatch_grads_n.append(single_traj_grads)

        minibatch_grads_n = torch.stack(minibatch_grads_n, dim=0)
        avg_grads_n = torch.mean(minibatch_grads_n, dim=0)  # grads for the i-th agent and its actors
        prev_v_list = copy.deepcopy(avg_grads_n)
        y_list = copy.deepcopy(prev_v_list)
        prev_v_lists.append(prev_v_list)
        y_lists.append(y_list)

    consensus_y_lists = take_grad_consensus(y_lists, pi)
    agents = take_param_consensus(agents, pi)
    for agent, y_list in zip(agents, consensus_y_lists):
        update_param(agent, y_list, lr=lr)
    return prev_v_lists, consensus_y_lists

