import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from networks import MLPSquashedGaussian
from torch.distributions import Normal
from networks import mlp

class RBF(torch.nn.Module):
    def __init__(self, parametrized=False, act_dim=None, obs_dim=None, hidden_sizes=None, activation=torch.nn.ReLU):
        super(RBF, self).__init__()
        self.parametrized = parametrized

        if parametrized:
            self.log_std_layer = mlp([obs_dim] + list(hidden_sizes) + [act_dim] , activation, activation)
            self.log_std_min = 2
            self.log_std_max = -20
            

    def forward(self, input_1, input_2,  h_min=1e-3):
        _, out_dim1 = input_1.size()[-2:]
        _, out_dim2 = input_2.size()[-2:]
        num_particles = input_2.size()[-2]
        assert out_dim1 == out_dim2
        
        # Compute the pairwise distances of left and right particles.
        diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)
        dist_sq = diff.pow(2).sum(-1)
        dist_sq = dist_sq.unsqueeze(-1)
        
        if self.parametrized == False:
            # Get median.
            median_sq = torch.median(dist_sq.detach().reshape(-1, num_particles*num_particles), dim=1)[0]
            median_sq = median_sq.reshape(-1,1,1,1)
            
            h = median_sq / (2 * np.log(num_particles + 1.))
            sigma = torch.sqrt(h)
        else:
            log_std = self.log_std_layer(net_out)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            sigma = torch.exp(log_std)
        

        gamma = 1.0 / (1e-8 + 2 * sigma**2) 
        kappa = (-gamma * dist_sq).exp()
        kappa_grad = -2. * (diff * gamma) * kappa
        return kappa.squeeze(-1), diff, gamma, kappa_grad

class ActorSvgd(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, num_svgd_particles, num_svgd_steps, svgd_lr, q1, q2):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.num_particles = num_svgd_particles
        self.num_svgd_steps = num_svgd_steps
        self.svgd_lr = svgd_lr
        
        self.Kernel = RBF()
        self.q1 = q1
        self.q2 = q2

    def svgd_optim(self, x, dx): 
        dx = dx.view(x.size())
        x = x + self.svgd_lr * dx
        return x

    def sampler(self, obs, a, with_logprob=True):
        logp = 0

        def phi(X):
            nonlocal logp

            X = X.requires_grad_(True)
            
            log_prob1 = self.q1(obs, X)
            log_prob2 = self.q1(obs, X)
            log_prob = torch.min(log_prob1, log_prob2)

            score_func = autograd.grad(log_prob.sum(), X, retain_graph=True, create_graph=True)[0]
            
            X = X.reshape(-1, self.num_particles, self.act_dim)
            score_func = score_func.reshape(X.size())
            K_XX, K_diff, K_gamma, K_grad = self.Kernel(X, X.detach())
            phi = (K_XX.detach().matmul(score_func) - K_grad.sum(2)) / self.num_particles 
            
            # compute the entropy
            if with_logprob:
                #import pdb; pdb.set_trace()
                tmp1 = (K_grad * score_func.reshape(-1,1,self.num_particles,self.act_dim)).sum(-1).mean(-1)
                tmp2 = -2 * K_gamma.view(-1,1) * ((-K_grad * K_diff).sum(-1) - self.act_dim * K_XX).mean(-1)
                logp -= self.svgd_lr*(tmp1+tmp2)
            
            return phi, log_prob 
        
        for t in range(self.num_svgd_steps):
            phi_, q_s_a= phi(a)
            a = self.svgd_optim(a, phi_)
            a = torch.clamp(a, -self.act_limit, self.act_limit).detach()

        return a, logp, q_s_a

    
        


class ActorSvgdNonParam(ActorSvgd):
    def __init__(self, obs_dim, act_dim, act_limit, num_svgd_particles, num_svgd_steps, svgd_lr, device, test_deterministic, batch_size, q1, q2):
        ActorSvgd.__init__(self, obs_dim, act_dim, act_limit, num_svgd_particles, num_svgd_steps, svgd_lr, q1, q2)
        
        self.test_deterministic = test_deterministic
        self.batch_size = batch_size

        self.a0 = torch.normal(0, 1, size=(2 * batch_size * num_svgd_particles, self.act_dim)).to(device)

    def act(self, obs, deterministic=False, with_logprob=True):
        # sample from a Gaussian
        a0 = self.a0[torch.randint(len(self.a0), (len(obs),))]
        a0 = self.act_limit * torch.tanh(a0) 
        
        # run svgd
        a, logp_a, q_s_a = self.sampler(obs, a0.detach(), with_logprob) 
        
        # compute the entropy 
        if with_logprob:
            logp0 = (self.act_dim/2) * np.log(2 * np.pi) + (self.act_dim/2)+ (2*(np.log(2) - a0 - F.softplus(-2*a0))).sum(axis=-1).view(-1,self.num_particles)
            logp_a = (logp0 + logp_a).mean(-1)
        
        # at test time
        if (with_logprob == False) and (deterministic == True):
            a = a.view(-1, self.num_particles, self.act_dim)[:,q_s_a.view(-1, self.num_particles).argmax(-1),:]
         
        elif (with_logprob == False) and (deterministic == False):
            a = a.view(-1, self.num_particles, self.act_dim)[:,np.random.randint(self.num_particles),:]

        return a, logp_a


class ActorSvgdP0Param(ActorSvgd):
    def __init__(self, obs_dim, act_dim, act_limit, num_svgd_particles, num_svgd_steps, svgd_lr, test_deterministic, hidden_sizes, q1, q2, activation=torch.nn.ReLU):
        ActorSvgd.__init__(self, obs_dim, act_dim, act_limit, num_svgd_particles, num_svgd_steps, svgd_lr, q1, q2)
        self.test_deterministic = test_deterministic
        self.policy_net = MLPSquashedGaussian(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False, with_logprob=True):
        # sample from a gaussian
        mu, sigma = self.policy_net(obs)
        a0 = Normal(mu, sigma).rsample()
        a0 = self.act_limit * torch.tanh(a0)
        
        # run svgd
        a, logp_a, q_s_a = self.sampler(obs, a0, with_logprob) 

        # compute the entropy 
        if with_logprob:
            logp0 = (self.act_dim/2) * np.log(2 * np.pi) + (self.act_dim/2) + (2*(np.log(2) - a0 - F.softplus(-2*a0))).sum(axis=-1).view(-1,self.num_particles)
            logp_a = (logp0 + logp_a).mean(-1)
        
        # at test time
        if (with_logprob == False) and (deterministic == True):
            a = a.view(-1, self.num_particles, self.act_dim)[:,q_s_a.view(-1, self.num_particles).argmax(-1),:]
        
        elif (with_logprob == False) and (deterministic == False):
            a = a.view(-1, self.num_particles, self.act_dim)[:,np.random.randint(self.num_particles),:]
        
        return a, logp_a


class ActorSvgdP0KernelParam(ActorSvgd):
    def __init__(self, obs_dim, act_dim, act_limit, num_svgd_particles, num_svgd_steps, svgd_lr, device, test_deterministic, hidden_sizes, q1, q2, activation=torch.nn.ReLU):
        ActorSvgd.__init__(self, obs_dim, act_dim, num_svgd_particles, num_svgd_steps, svgd_lr, q1, q2)
        self.test_deterministic = test_deterministic
        self.policy_net = MLPSquashedGaussian(obs_dim, act_dim, hidden_sizes, activation)
        self.Kernel = RBF(parametrized=True)

    def act(self, obs, deterministic=False, with_logprob=True):

        return a, logp_a


