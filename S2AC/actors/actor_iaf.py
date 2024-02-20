from networks import MLPSquashedGaussian
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
import time
class ActorIaf(torch.nn.Module):
    def __init__(self, actor, obs_dim, act_dim, act_limit, hidden_sizes, device, batch_size, 
                 activation=torch.nn.Identity, test_action_selection=False, latent_dim=2):
        super(ActorIaf, self).__init__()
        self.batch_size = batch_size
        self.num_particles = 1
        self.actor = actor
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.test_action_selection = test_action_selection
        self.device = device 
        self.policy_net = MLPSquashedGaussian(obs_dim, act_dim, hidden_sizes, activation)
        self.qz_net = MLPSquashedGaussian(obs_dim, act_dim, hidden_sizes, activation)
        self.down_net = MLPSquashedGaussian(act_dim, act_dim, hidden_sizes, activation)
        self.steps_debug = 0

    def log_prob(self, pi_distribution, pi_action):
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
        return logp_pi

    def act(self, obs, action_selection=None, with_logprob=None, loss_q_=None, itr=None, with_kl=False):
        start_time = time.time()
        self.mu, self.sigma = self.policy_net(obs)
        self.qz_mu, self.qz_sigma = self.qz_net(obs)


        prior = Normal(self.mu, self.sigma)
        posterior = Normal(self.mu+self.qz_mu, self.sigma+self.qz_sigma)
        z = posterior.rsample()
        x = self.down_net(z)
        z = (z - x[0] * 0.1) / (x[1] * 0.1)
        logqs = posterior.log_prob(z) + x[1]*0.1
        logps = prior.log_prob(z)
        kl = torch.clamp((logqs - logps).mean()*1e-3, -100, 100)
        if action_selection == 'max':
            a = self.qz_mu
        else:
            a = posterior.rsample()
        
        logp_pi = self.log_prob(posterior, a) if with_logprob else None
        
        a = torch.tanh(a)
        a = self.act_limit * a
        if with_kl:
            return a.reshape(-1, self.act_dim), logp_pi, kl
        return a.reshape(-1, self.act_dim), logp_pi
