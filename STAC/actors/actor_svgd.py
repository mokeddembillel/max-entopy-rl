import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from networks import MLPSquashedGaussian
from torch.distributions import Normal, Categorical
from actors.kernels import RBF

class ActorSvgd(torch.nn.Module):
    def __init__(self, actor, obs_dim, act_dim, act_limit, num_svgd_particles, num_svgd_steps, svgd_lr, test_deterministic, batch_size, device, hidden_sizes, q1, q2, activation=torch.nn.ReLU):
        super().__init__()
        self.actor = actor
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.num_particles = num_svgd_particles
        self.num_svgd_steps = num_svgd_steps
        self.svgd_lr = svgd_lr
        
        self.q1 = q1
        self.q2 = q2

        self.test_deterministic = test_deterministic
        self.batch_size = batch_size

        if actor == "ActorSvgdNonParam":
            self.a0 = torch.normal(0, 1, size=(2 * batch_size * num_svgd_particles, self.act_dim)).to(device)
        else:
            self.p0 = MLPSquashedGaussian(obs_dim, act_dim, hidden_sizes, activation)

        if actor == "svgd_p0_kernel_pram":
            self.Kernel = RBF(True, act_dim, hidden_sizes)
        else:
            self.Kernel = RBF(num_particles=self.num_particles)

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

    def act(self, obs, deterministic=False, with_logprob=True):
        if self.actor == "ActorSvgdNonParam":
            a0 = self.a0[torch.randint(len(self.a0), (len(obs),))]
            a0 = self.act_limit * torch.tanh(a0) 
        else:
            mu, sigma = self.p0(obs)
            a0 = Normal(mu, sigma).rsample()
            a0 = self.act_limit * torch.tanh(a0)

        # run svgd
        a, logp_a, q_s_a = self.sampler(obs, a0.detach(), with_logprob) 
        
        # compute the entropy 
        if with_logprob:
            logp0 = (self.act_dim/2) * np.log(2 * np.pi) + (self.act_dim/2)+ (2*(np.log(2) - a0 - F.softplus(-2*a0))).sum(axis=-1).view(-1,self.num_particles)
            logp_a = (logp0 + logp_a).mean(-1)
        
        a = a.view(-1, self.num_particles, self.act_dim)
        # at test time
        if (with_logprob == False) and (deterministic == True):
            a = a.view(-1, self.num_particles, self.act_dim)[:,q_s_a.view(-1, self.num_particles).argmax(-1),:]
         
        elif (with_logprob == False) and (deterministic == False):
            beta = 1
            max_q_value = torch.max(q_s_a, dim=1, keepdim=True)[0] # (-1, 1)
            soft_max_porbs = torch.exp(beta * q_s_a - max_q_value)
            dist = Categorical(soft_max_porbs/ torch.sum(soft_max_porbs, dim=1, keepdim=True))
            a = torch.gather(q_s_a, 1, dist.sample().unsqueeze(-1))
        return a, logp_a

