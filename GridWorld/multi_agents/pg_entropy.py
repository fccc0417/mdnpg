from nets import PolicyNet, ValueNet
import torch.nn.functional as F
from rl_utils import *


class PGwithEntropy:
    """ momentum-based NPG algorithm """
    def __init__(self, state_space, action_space, lmbda, critic_lr, gamma, entropy_para, device):
        self.state_dim = state_space.shape[0]
        self.action_dim = action_space.n
        self.actor = PolicyNet(self.state_dim, self.action_dim).to(device)
        self.critic = ValueNet(self.state_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma  # discount factor
        self.lmbda = lmbda  # lambda for GAE
        self.entropy_para = entropy_para  # for regularization
        self.device = device

    def take_action(self, state):
        """Giving a state, take an action using policy network."""
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        """Calculate objective function."""
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def compute_grads(self, transition_dict, advantage):
        """Calculate gradients for the objective function."""
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        return obj_grad

    def update_value(self, transition_dict):
        """Update value network and calculate advantage functions."""
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
        self.critic_optimizer.step()

        # Add penalty term to the objective function.
        log_probs_all = torch.log(self.actor(states))
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        sum_log_probs = torch.sum(log_probs_all, dim=1)
        entropy_term = 1 / self.action_dim * sum_log_probs + np.log(self.action_dim)
        entropy_term = self.entropy_para * entropy_term

        return advantage + entropy_term
