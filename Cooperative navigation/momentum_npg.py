from nets import PolicyNet, ValueNet
import torch.nn.functional as F
from rl_utils import *


class MomentumNPG:
    """ momentum-based NPG algorithm """
    def __init__(self, num_agents, state_dim, action_dim, lmbda, kl_constraint, alpha,
                 critic_lr, gamma, device, min_isw, beta):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        # 策略网络参数不需要优化器更新
        self.actors = []
        for _ in range(self.num_agents):
            self.actors.append(PolicyNet(self.state_dim, self.action_dim).to(device))
        self.critic = ValueNet(self.state_dim).to(device)  # value net for the i-th agent
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda  # GAE参数
        self.kl_constraint = kl_constraint  # KL距离最大限制
        self.alpha = alpha  # 线性搜索参数
        self.device = device
        self.min_isw = min_isw
        self.beta = beta

    def take_actions(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        actions = []
        for actor in self.actors:
            probs = actor(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            actions.append(action)
        return actions

    def calc_log_probs(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions_list = torch.tensor(transition_dict['actions']).T.to(self.device)
        # actions_list = torch.tensor(transition_dict['actions']).view(self.num_agents, -1).to(self.device)
        old_log_probs_list = []
        log_probs_list = []
        for idx, actor in enumerate(self.actors):
            # probs = actor(states)
            # probs4action = actor(states).gather(1, actions_list[idx].unsqueeze(1)).detach()
            old_log_probs = torch.log(actor(states).gather(1, actions_list[idx].unsqueeze(1))).detach()
            old_log_probs_list.append(old_log_probs)
            log_probs = torch.log(actor(states).gather(1, actions_list[idx].unsqueeze(1)))
            log_probs_list.append(log_probs)
        return old_log_probs_list, log_probs_list

    def compute_grads(self, advantage, old_log_probs_list, log_probs_list):
        obj_grad_list = []
        for actor, old_log_probs, log_probs in zip(self.actors, old_log_probs_list, log_probs_list):
            ratio = torch.exp(log_probs - old_log_probs)
            surrogate_obj = torch.mean(ratio * advantage)
            grads = torch.autograd.grad(surrogate_obj, actor.parameters())
            obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
            obj_grad_list.append(obj_grad)
        # obj_grad = [grad.view(-1) for grad in grads]
        return obj_grad_list

    def compute_IS_weight(self, actions_list, states, phis, min_isw):
        # test = torch.log(self.actor(state_list).gather(1, action_list))
        weight_list = []
        for idx, (actor, phi) in enumerate(zip(self.actors, phis)):
            log_probs = torch.log(actor(states).gather(1, actions_list[idx].unsqueeze(1))).detach()
            prob_tau = torch.prod(log_probs)
            old_policy_log_probs = torch.log(phi(states).gather(1, actions_list[idx].unsqueeze(1))).detach()
            prob_old_tau = torch.prod(old_policy_log_probs)
            weight = prob_old_tau / (prob_tau + 1e-8)
            weight = np.max((min_isw, weight))
            weight_list.append(weight)
        return weight_list

    def compute_grad_traj_prev_weights(self, states, actions_list, phis, advantage):
        obj_grad_list = []
        for idx, phi in enumerate(phis):
            old_policy_log_probs = torch.log(phi(states).gather(1, actions_list[idx].unsqueeze(1))).detach()
            log_probs = torch.log(phi(states).gather(1, actions_list[idx].unsqueeze(1)))
            ratio = torch.exp(log_probs - old_policy_log_probs)
            old_policy_surrogate_obj = torch.mean(ratio * advantage)
            grads = torch.autograd.grad(old_policy_surrogate_obj, phi.parameters())
            obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
            obj_grad_list.append(obj_grad)
        return obj_grad_list

    def compute_u_k(self, transition_dict, advantage, prev_u_list, phis, beta):  # 更新策略函数
        old_log_probs_list, log_probs_list = self.calc_log_probs(transition_dict)
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # actions_list = torch.tensor(transition_dict['actions']).view(self.num_agents, -1).to(self.device)
        actions_list = torch.tensor(transition_dict['actions']).T.to(self.device)

        isw_list = self.compute_IS_weight(actions_list, states, phis, self.min_isw)
        prev_g_list = self.compute_grad_traj_prev_weights(states, actions_list, phis, advantage)
        grad_list = self.compute_grads(advantage, old_log_probs_list, log_probs_list)

        grad_u_list = []
        for grad, prev_u, prev_g, isw in zip(grad_list, prev_u_list, prev_g_list, isw_list):
            grad_u = beta * grad + (1 - beta) * (prev_u + grad - isw * prev_g)
            grad_u_list.append(grad_u)
        return grad_u_list

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

    # def update_value(self, transition_dict):
    #     rewards = torch.tensor(transition_dict['rewards'],
    #                            dtype=torch.float).view(-1, 1).to(self.device)
    #     advantage = compute_advantage(self.gamma, self.lmbda, rewards).to(self.device)
    #     return advantage

    def hessian_matrix_vector_product(self, states, old_action_dists, vector, idx):
        # 计算黑塞矩阵和一个向量的乘积
        new_action_dists = torch.distributions.Categorical(self.actors[idx](states))
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists,
                                                 new_action_dists))  # 计算平均KL距离
        kl_grad = torch.autograd.grad(kl,
                                      self.actors[idx].parameters(),
                                      create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # KL距离的梯度先和向量进行点积运算
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                    self.actors[idx].parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists, idx):  # 共轭梯度法求解方程
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):  # 共轭梯度主循环
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p, idx)
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

    def compute_precondition_with_v(self, states_list, v_k_list):
        states = torch.tensor(states_list, dtype=torch.float).to(self.device)

        vec_grad_list = []
        for idx, v_k in enumerate(v_k_list):
            old_action_dists = torch.distributions.Categorical(self.actors[idx](states).detach())
            # 用共轭梯度法计算x = H^(-1)g
            descent_direction = self.conjugate_gradient(v_k, states, old_action_dists, idx)

            Hd = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction, idx)
            max_coef = torch.sqrt(2 * self.kl_constraint /
                                  (torch.dot(descent_direction, Hd) + 1e-8))

            min_coef = 1e-7
            max_coef = max_coef if max_coef > min_coef else min_coef
            vec_grad = max_coef * descent_direction
            vec_grad_list.append(vec_grad)
            # print(f'\nmax_coef={max_coef}')
        return vec_grad_list

        # new_para = self.line_search(states, actions, advantage, old_log_probs, old_action_dists,
        #                             descent_direction * max_coef)  # 线性搜索
        # torch.nn.utils.convert_parameters.vector_to_parameters(
        #     new_para, self.actor.parameters())  # 用线性搜索后的参数更新策略


    '''
    def line_search(self, states, actions, advantage, old_log_probs,
                    old_action_dists, max_vec):  # 线性搜索
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage,
                                             old_log_probs, self.actor)
        for i in range(15): 
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
                return new_para
        return old_para
    '''