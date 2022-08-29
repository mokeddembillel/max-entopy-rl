import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma

    def forward(self, input_1, input_2,  h_min=1e-3):
        k_fix, out_dim1 = input_1.size()[-2:]
        k_upd, out_dim2 = input_2.size()[-2:]
        num_particles = input_2.size()[-2]
        assert out_dim1 == out_dim2
        
        # Compute the pairwise distances of left and right particles.
        diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)
        dist_sq = diff.pow(2).sum(-1)
        dist_sq = dist_sq.unsqueeze(-1)
        
        # Get median.
        median_sq = torch.median(dist_sq.detach().reshape(-1, num_particles*num_particles), dim=1)[0]
        median_sq = median_sq.reshape(-1,1,1,1)
        
        h = median_sq / (2 * np.log(num_particles + 1.))
        sigma = torch.sqrt(h)
        gamma = 1.0 / (1e-8 + 2 * sigma**2) 
        
        kappa = (-gamma * dist_sq).exp()
        kappa_grad = -2. * (diff * gamma) * kappa
        return kappa.squeeze(-1), diff, gamma, kappa_grad

class ActorSvgd(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, num_svgd_particles, num_svgd_steps, svgd_lr, test_deterministic, device):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.num_particles = num_svgd_particles
        self.num_svgd_steps = num_svgd_steps
        self.svgd_lr = svgd_lr
        self.test_deterministic = test_deterministic
        self.device = device
        self.Kernel = RBF()

    def svgd_optim(self, x, dx): 
        dx = dx.view(x.size())
        x = x + self.svgd_lr * dx
        return x

    def sampler(self, obs, a, with_logprob=True, debug=False, itr=None):
        logp = 0

        def phi(X):
            nonlocal logp
            X = X.requires_grad_(True)
            log_prob = self.q1(obs, X)
            score_func = autograd.grad(log_prob.sum(), X, retain_graph=True, create_graph=True)[0]
            
            X = X.reshape(-1, self.num_particles, self.act_dim)
            score_func = score_func.reshape(X.size())
            K_XX, K_diff, K_gamma, K_grad = self.Kernel(X, X.detach())
            phi = (K_XX.detach().matmul(score_func) - K_grad.sum(2)) / self.num_particles 
            
            # compute the entropy
            if with_logprob:
                #import pdb; pdb.set_trace()
                line_4 = (K_grad * score_func.reshape(-1,1,self.num_particles,self.act_dim)).sum(-1).mean(-1)
                line_5 = -2 * K_gamma.view(-1,1) * ((-K_grad * K_diff).sum(-1) - self.act_dim * K_XX).mean(-1)
                logp -= self.svgd_lr*(line_4+line_5)
            
            return phi 
        
        for t in range(self.num_svgd_steps):
            a = self.svgd_optim(a, phi(a))
            a = torch.clamp(a, -self.act_limit, self.act_limit).detach()

        return a, logp
        


class ActorSvgdNonParam(ActorSvgd):
    def __init__(self, obs_dim, act_dim, act_limit, num_svgd_particles, num_svgd_steps, svgd_lr, test_deterministic, device):
        ActorSvgd.__init__(self, obs_dim, act_dim, act_limit, num_svgd_particles, num_svgd_steps, svgd_lr, test_deterministic, device)

    def act(self, obs):
        # sample from a Gaussian
        a0 = torch.normal(0, 1, size=(len(obs) * self.num_particles, self.act_dim)).to(self.device)
        a0 = self.act_limit * torch.tanh(a0) 

        # entropy of a gaussian followed by tanh
        logp0 = (self.act_dim/2) * np.log(2 * np.pi) + (self.act_dim/2)
        logp0 += (2*(np.log(2) - a0 - F.softplus(-2*a0))).sum(axis=-1)

        # run svgd
        a, logp = self.svgd_sampler(obs, a0.detach()) 

        # compute the entropy 
        logp_a = logp0 + logp.mean(-1)

        return a, logp_a.detach()


class ActorSvgdP0Param(ActorSvgd):
    def __init__(self, obs_dim, act_dim, act_limit, num_svgd_particles, num_svgd_steps, svgd_lr, test_deterministic):
        ActorSvgd.__init__(self, obs_dim, act_dim, num_svgd_particles, num_svgd_steps, svgd_lr, test_deterministic)
    def act(self, obs):
        return ActorSvgd.act(self, obs)


class ActorSvgdP0KernelParam(ActorSvgd):
    def __init__(self, obs_dim, act_dim, act_limit, num_svgd_particles, num_svgd_steps, svgd_lr, test_deterministic):
        ActorSvgd.__init__(self, obs_dim, act_dim, num_svgd_particles, num_svgd_steps, svgd_lr, test_deterministic)
    def act(self, obs):
        return ActorSvgd.act(self, obs)


