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

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        print('activation ', activation)
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPSquashedGaussian(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, num_particles, log_std_min, log_std_max):#, wandb):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.num_particles = num_particles
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std


    def act(self, obs, deterministic, with_logprob):#, wandb=None):
        if num_particles is None:
            num_particles = self.num_particles
        
        mu, sigma = self.forward(obs)

        pi_distribution = Normal(self.mu, self.std)
        
        if deterministic:
            pi_action = self.mu
        else:
            pi_action = pi_distribution.rsample()
        
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
            
            if num_particles>0:
                logp_pi = logp_pi.view(-1,num_particles).mean(-1)
        else:
            logp_pi = None
        
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action.reshape(-1,pi_action.size()[-1]), logp_pi




