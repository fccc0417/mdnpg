import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(torch.nn.Module):
    """Policy neural network."""
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.dense1 = nn.Linear(state_dim, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, action_dim)
    def forward(self, x):
        x1 = F.relu(self.dense1(x))   #torch.tanh
        x2 = F.relu(self.dense2(x1))
        x3 = self.dense3(x2)
        dist = F.softmax(x3, dim=-1)
        return dist


class ValueNet(torch.nn.Module):
    """Value neural network."""
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        return self.fc3(x2)

