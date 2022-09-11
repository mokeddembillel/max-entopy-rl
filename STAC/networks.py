import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal



def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class MLPFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, out_dim, hidden_sizes, activation=nn.Identity):
        super().__init__()
        # print('activation ', activation)
        self.net = mlp([obs_dim + act_dim] + list(hidden_sizes) + [out_dim], activation)

    def forward(self, obs, act):
        out = self.net(torch.cat([obs, act], dim=-1))
        return torch.squeeze(out, -1) # Critical to ensure q has right shape.


class MLPSquashedGaussian(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Identity):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, obs):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std


    




