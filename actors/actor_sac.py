from networks import MLPSquashedGaussian
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal

class ActorSac(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes, activation, test_deterministic):
        super(ActorSac, self).__init__()
        self.num_particles = 1
        self.act_limit = act_limit
        self.test_deterministic = test_deterministic
        self.policy_net = MLPSquashedGaussian(obs_dim, act_dim, hidden_sizes, activation)
    
    # def forward(self):
    #     return self.net.forward

    def log_prob(self, pi_distribution, pi_action):
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
        return logp_pi

    def act(self, obs, deterministic=None, with_logprob=None):#, wandb=None):
        mu, sigma = self.policy_net(obs)

        pi_distribution = Normal(mu, sigma)
        
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        
        logp_pi = self.log_prob(pi_distribution, pi_action) if with_logprob else None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action.reshape(-1,pi_action.size()[-1]), logp_pi
