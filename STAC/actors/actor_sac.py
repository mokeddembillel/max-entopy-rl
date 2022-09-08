from networks import MLPSquashedGaussian
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal

class ActorSac(torch.nn.Module):
    def __init__(self, actor, obs_dim, act_dim, act_limit, hidden_sizes, activation=torch.nn.Identity, test_deterministic=False):
        super(ActorSac, self).__init__()
        self.num_particles = 1
        self.actor = actor
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.test_deterministic = test_deterministic
        self.policy_net = MLPSquashedGaussian(obs_dim, act_dim, hidden_sizes, activation)
    

    def log_prob(self, pi_distribution, pi_action):
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
        return logp_pi

    def act(self, obs, deterministic=None, with_logprob=None, loss_q=None):
        self.mu, self.sigma = self.policy_net(obs)

        pi_distribution = Normal(self.mu, self.sigma)
        
        if deterministic:
            a = self.mu
        else:
            a = pi_distribution.rsample()
        
        logp_pi = self.log_prob(pi_distribution, a) if with_logprob else None
        
        a = torch.tanh(a)
        a = self.act_limit * a
        
        return a.reshape(-1, self.act_dim), logp_pi
