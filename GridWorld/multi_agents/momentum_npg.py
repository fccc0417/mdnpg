from nets import PolicyNet, ValueNet
import torch.nn.functional as F
from rl_utils import *
import copy


class MomentumNPG:
    """A momentum-based NPG agent.
    Attributes:
        actor: policy network for the i-th agent.
        critic: value network for the i-th agent.
        gamma: discount factor.
        lmbda: lambda for GAE.
        kl_constraint: KL constraint for NPG.
        alpha: used for line search.
        min_isw: minimum importance weight.
        beta: beta for momentum-based variance reduction
    """
    def __init__(self,  state_space, action_space, lmbda, kl_constraint, alpha, critic_lr,
                 gamma, device, min_isw, beta):
        self.state_dim = state_space.shape[0]
        self.action_dim = action_space.n
        self.actor = PolicyNet(self.state_dim, self.action_dim).to(device)
        self.critic = ValueNet(self.state_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.kl_constraint = kl_constraint
        self.alpha = alpha
        self.device = device
        self.min_isw = min_isw
        self.beta = beta

    def take_action(self, state):
        """Giving a state, take an action using policy network."""
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        """Calculate product between Hessian of KL and gradient vector."""
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector + 0.1 * vector
        # return grad2_vector + 0.01 * vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        """Conjugate gradient method for the calculation of $H^t_j \times y^{t+1}_j$.
        See: https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/npg.py
        """
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):
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

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        """Calculate objective function for policy update."""
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs, old_action_dists, max_vec):
        """Line search for adjust the update step size."""
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        for i in range(20):
            coef = self.alpha**i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
            new_action_dists = torch.distributions.Categorical(new_actor(states))
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists,new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage,old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return coef * max_vec
        return torch.zeros_like(old_para)

    def compute_grads(self, transition_dict, advantage):
        """Calculate gradients for the advantage function."""
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        return obj_grad

    def compute_IS_weight(self, action_list, state_list, phi, min_isw):
        """Calculate importance weight."""
        log_probs = torch.log(self.actor(state_list).gather(1, action_list)).detach()
        prob_tau = torch.prod(log_probs)
        old_policy_log_probs = torch.log(phi(state_list).gather(1, action_list)).detach()
        prob_old_tau = torch.prod(old_policy_log_probs)
        weight = prob_old_tau / (prob_tau + 1e-8)
        weight = np.max((min_isw, weight))
        return weight

    def compute_grad_traj_prev_weights(self, state_list, action_list, phi, advantage):
        """Calculate gradients of old policy using the trajectory generated by new policy"""
        old_policy_log_probs = torch.log(phi(state_list).gather(1, action_list)).detach()
        old_policy_surrogate_obj = self.compute_surrogate_obj(state_list, action_list, advantage,
                                                              old_policy_log_probs, phi)
        grads = torch.autograd.grad(old_policy_surrogate_obj, phi.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        return obj_grad

    def compute_v(self, transition_dict, advantage, prev_v, phi, beta):
        """Generate gradient estimator based on momentum-based variance reduction."""
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        isw = self.compute_IS_weight(actions, states, phi, self.min_isw)
        prev_g = self.compute_grad_traj_prev_weights(states, actions, phi, advantage)
        grad = self.compute_grads(transition_dict, advantage)
        grad_v = beta * grad + (1 - beta) * (prev_v + grad - isw * prev_g)
        return grad_v

    def compute_precondition_with_y(self, states_list, y, transition_dict, advantage):
        """Calculate precondition for NPG."""
        states = torch.tensor(states_list, dtype=torch.float).to(self.device)
        old_action_dists = torch.distributions.Categorical(self.actor(states).detach())

        # 用共轭梯度法计算x = H^(-1)g
        descent_direction = self.conjugate_gradient(y, states, old_action_dists)

        Hd = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8))

        states_2 = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions_2 = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        old_log_probs_2 = torch.log(self.actor(states_2).gather(1, actions_2)).detach()
        old_action_dists_2 = torch.distributions.Categorical(self.actor(states_2).detach())

        # line search
        vec_grad = self.line_search(states_2, actions_2, advantage, old_log_probs_2, old_action_dists_2,
                                    descent_direction * max_coef)
        return vec_grad


    def update_value(self, transition_dict):
        """Update value network and calculate advantage functions."""
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
