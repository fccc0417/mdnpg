from nets import PolicyNet, ValueNet
import torch.nn.functional as F
from rl_utils import *
import copy


class MomentumNPG:
    """ momentum-based NPG algorithm """
    def __init__(self,  state_space, action_space, lmbda,
                 kl_constraint, alpha, critic_lr, gamma, device, min_isw, beta):
        self.state_dim = state_space.shape[0]
        self.action_dim = action_space.n
        self.actor = PolicyNet(self.state_dim, self.action_dim).to(device)
        self.critic = ValueNet(self.state_dim).to(device)
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
        return grad2_vector + 0.1 * vector
        # return grad2_vector + 0.01 * vector

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
                # print(new_rdotr)
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        # print(new_rdotr)
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
        for i in range(20):  # 线性搜索主循环
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
                return coef * max_vec
        return torch.zeros_like(old_para)  #old_para

    def compute_grads(self, transition_dict, advantage):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # obj_grad = [grad.view(-1) for grad in grads]
        return obj_grad

    def compute_IS_weight(self, action_list, state_list, phi, min_isw):
        # datas = torch.log(self.actor(state_list).gather(1, action_list))
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
        # obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        return obj_grad

    def compute_v(self, transition_dict, advantage, prev_v, phi, beta):  # 更新策略函数
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        isw = self.compute_IS_weight(actions, states, phi, self.min_isw)
        prev_g = self.compute_grad_traj_prev_weights(states, actions, phi, advantage)
        grad = self.compute_grads(transition_dict, advantage)
        grad_v = beta * grad + (1 - beta) * (prev_v + grad - isw * prev_g)
        return grad_v

    def compute_precondition_with_y(self, states_list, y, transition_dict, advantage):
        states = torch.tensor(states_list, dtype=torch.float).to(self.device)
        # actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        old_action_dists = torch.distributions.Categorical(self.actor(states).detach())

        # 用共轭梯度法计算x = H^(-1)g
        descent_direction = self.conjugate_gradient(y, states, old_action_dists)

        Hd = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint /
                              (torch.dot(descent_direction, Hd) + 1e-8))

        # min_coef = 1e-7
        # max_coef = max_coef if max_coef > min_coef else min_coef
        # vec_grad = max_coef * descent_direction

        states_2 = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions_2 = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        old_log_probs_2 = torch.log(self.actor(states_2).gather(1, actions_2)).detach()
        old_action_dists_2 = torch.distributions.Categorical(self.actor(states_2).detach())
        vec_grad = self.line_search(states_2, actions_2, advantage, old_log_probs_2, old_action_dists_2,
                                    descent_direction * max_coef)  # 线性搜索
        return vec_grad


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
