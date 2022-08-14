import copy
import torch
import json
import numpy as np


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


def take_param_consensus(agents, pi):
    layer_1_w = []
    layer_1_b = []

    layer_2_w = []
    layer_2_b = []

    # layer_3_w = []
    # layer_3_b = []

    for agent in agents:
        layer_1_w.append(agent.actor.dense1.weight.data)
        layer_1_b.append(agent.actor.dense1.bias.data)

        layer_2_w.append(agent.actor.dense2.weight.data)
        layer_2_b.append(agent.actor.dense2.bias.data)

        # layer_3_w.append(agent.actor.dense3.weight.data)
        # layer_3_b.append(agent.actor.dense3.bias.data)


    for agent_idx, agent in enumerate(agents):
        agent.actor.dense1.weight.data = torch.sum(
            torch.stack(tuple(layer_1_w)) * torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1), 0).clone()
        agent.actor.dense1.bias.data = torch.sum(torch.stack(tuple(layer_1_b)) * torch.tensor(pi[agent_idx]).unsqueeze(-1),
                                           0).clone()

        agent.actor.dense2.weight.data = torch.sum(
            torch.stack(tuple(layer_2_w)) * torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1), 0).clone()
        agent.actor.dense2.bias.data = torch.sum(torch.stack(tuple(layer_2_b)) * torch.tensor(pi[agent_idx]).unsqueeze(-1),
                                           0).clone()

        # agent.actor.dense3.weight.data = torch.sum(
        #     torch.stack(tuple(layer_3_w)) * torch.tensor(pi[agent_idx]).unsqueeze(-1).unsqueeze(-1), 0).clone()
        # agent.actor.dense3.bias.data = torch.sum(torch.stack(tuple(layer_3_b)) * torch.tensor(pi[agent_idx]).unsqueeze(-1),
        #                                    0).clone()
    return agents


def take_grad_consensus(v_k_list_flat, pi, agents):
    consensus_v_k = []
    for j in range(len(v_k_list_flat)):
        v_k_cons = torch.sum(torch.stack(tuple(v_k_list_flat)) * torch.tensor(pi[j]).unsqueeze(-1), 0).clone()
        consensus_v_k.append(v_k_cons)
    return consensus_v_k


def update_param(agent, obj_grad, lr=3e-4):
    '''update parameters for an agent'''
    old_para = torch.nn.utils.convert_parameters.parameters_to_vector(agent.actor.parameters())
    new_para = old_para + lr * obj_grad
    torch.nn.utils.convert_parameters.vector_to_parameters(new_para, agent.actor.parameters())


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


def update_v(v_k, u_k, prev_u_k):
    next_v_k = v_k + u_k - prev_u_k
    return next_v_k


def get_flat_grad(y: torch.Tensor, model, **kwargs):
    grads = torch.autograd.grad(y, model.parameters(), **kwargs)  # type: ignore
    return torch.cat([grad.reshape(-1) for grad in grads])


def set_from_flat_grads(model, flat_params):
    prev_ind = 0
    grads = []
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        grads.append(flat_params[prev_ind:prev_ind + flat_size]) #.view(param.size())
        prev_ind += flat_size
    return grads


def initialization_gt(sample_envs, agents, pi, lr, minibatch_size, max_eps_len):
    prev_u_list = []
    v_k_list = []

    for idx, (agent, sample_env) in enumerate(zip(agents, sample_envs)):
        minibatch_grads_n = []
        print("Initializing for "+ f"agent {idx}" + "...")
        for i in range(minibatch_size):
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
                reset = t == max_eps_len - 1
                if done or reset:
                    print("Agent "+ str(idx) + ': Batch Initial Trajectory ' + str(i) + ': { Reward:', episode_return, 'Done:', done , '}')
                    break

            advantage = agent.update_value(transition_dict)
            single_traj_grads = agent.compute_grads(transition_dict, advantage)
            # single_traj_grads = torch.stack(single_traj_grads, dim=0)
            minibatch_grads_n.append(single_traj_grads)

        # minibatch_grads_nn = torch.stack(minibatch_grads_n, dim=0)
        # avg_grads_n1 = torch.mean(minibatch_grads_nn, dim=0)
        avg_grads_n = torch.mean(torch.stack(minibatch_grads_n, dim=1), dim=1)
        # s = np.linalg.norm(avg_grads_n - avg_grads_n1, 2)
        prev_u = copy.deepcopy(avg_grads_n)
        v_k = copy.deepcopy(avg_grads_n)
        prev_u_list.append(prev_u)
        v_k_list.append(v_k)

    agents = take_param_consensus(agents, pi)
    for agent, u_k in zip(agents, prev_u_list):
        update_param(agent, u_k, lr=lr)
    return prev_u_list, v_k_list

