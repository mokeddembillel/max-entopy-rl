import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math
import torch.autograd as autograd
import torch.optim as optim


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





class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, num_svgd_particles):#, wandb):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.num_svgd_particles = num_svgd_particles
        #self.wandb = wandb

    def forward(self, obs, deterministic=False, with_logprob=True, num_svgd_particles=None):#, wandb=None):
        if num_svgd_particles is None:
            num_svgd_particles = self.num_svgd_particles
        
        #print('***************obs: ', obs.size() )
        net_out = self.net(obs)
        self.mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        self.std = torch.exp(log_std)

        #print('mu: ', self.mu)
        #print('std: ',self.std)
        # Pre-squash distribution and sample
        pi_distribution = Normal(self.mu, self.std)
        
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = self.mu
        else:
            pi_action = pi_distribution.rsample()
        
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
            
            #print(logp_pi)
            if num_svgd_particles>0:
                logp_pi = logp_pi.view(-1,num_svgd_particles).mean(-1)
        else:
            logp_pi = None
        
        #import pdb; pdb.set_trace()
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action.reshape(-1,pi_action.size()[-1]), logp_pi




